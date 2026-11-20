"""
A2C training (Advantage Actor-Critic), built on TorchRL.

A2C has an actor (the policy) and a critic (an estimate of how good a state
is). The advantage, the reward compared with what the critic expected, tells
the actor how much better an action was than usual. Each batch of experience is
used for one gradient step.
"""

import random
import time
from pathlib import Path

import numpy as np
import torch
from environment.custom_env import BloodDroneEnv
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.envs import Compose, GymWrapper, RewardScaling, RewardSum, TransformedEnv
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss
from torchrl.objectives.value import GAE


# ---------------------------------------------------------------------------
# Helpers (the same in all four training files)
# ---------------------------------------------------------------------------

ENV_SETTINGS = {"grid_size": 10, "max_steps": 200}


COMMON_DEFAULTS = {
    "hidden": (64, 64),         # two hidden layers of 64 units, ReLU
    "total_frames": 100_000,    # environment steps used for training
    "seed": 0,
    "eval_every": 5_000,        # frames between evaluation checkpoints
    "eval_episodes": 10,        # episodes per checkpoint
    "final_eval_episodes": 50,  # episodes for the final evaluation
}


CHECKPOINT_SEED = 500


FINAL_EVAL_SEED = 1000


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed=None, reward_scale=1.0):
    """
    The drone environment as a TorchRL environment.

    reward_scale shrinks the reward the learner sees (the reported episode
    reward is always the real one). Policy-gradient methods learn better when
    rewards are small numbers.
    """

    transforms = [RewardSum()]
    if reward_scale != 1.0:
        transforms.append(RewardScaling(loc=0.0, scale=reward_scale))

    env = TransformedEnv(
        GymWrapper(
            BloodDroneEnv(**ENV_SETTINGS),
            categorical_action_encoding=True,
        ),
        Compose(*transforms),
    )
    if seed is not None:
        env.set_seed(seed)
    return env


def env_sizes():
    """(observation size, number of actions)."""

    env = BloodDroneEnv(**ENV_SETTINGS)
    return env.observation_space.shape[0], env.action_space.n


def make_mlp(in_features, out_features, hidden):
    """Fully connected network with ReLU activations."""

    return MLP(
        in_features=in_features,
        out_features=out_features,
        num_cells=list(hidden),
        activation_class=nn.ReLU,
    )


def finished_episode_rewards(batch):
    """Total rewards of the episodes that ended inside this batch."""

    done = batch["next", "done"].reshape(-1)
    rewards = batch["next", "episode_reward"].reshape(-1)
    return rewards[done].tolist()


def greedy_action(network):
    """Function that picks the best action for an observation."""

    def act(observation):
        with torch.no_grad():
            scores = network(torch.as_tensor(observation, dtype=torch.float32))
            return int(scores.argmax().item())

    return act


def evaluate(act, episodes=10, seed_start=FINAL_EVAL_SEED):
    """
    Fly complete missions and measure how well the agent does.

    Returns the metrics used in the report: average reward, delivery success,
    steps per delivery, battery and package integrity at delivery, tower hits.
    """

    env = BloodDroneEnv(**ENV_SETTINGS)
    rewards, delivered, complete, steps, hits = [], [], [], [], []
    delivery_steps, delivery_battery, delivery_integrity = [], [], []

    for k in range(episodes):
        observation, _ = env.reset(seed=seed_start + k)
        total = 0.0
        done = False
        was_delivered = False

        while not done:
            observation, reward, terminated, truncated, _ = env.step(act(observation))
            total += reward
            done = terminated or truncated

            if env.delivered and not was_delivered:
                was_delivered = True
                delivery_steps.append(env.steps)
                delivery_battery.append(env.battery)
                delivery_integrity.append(env.package_integrity)

        rewards.append(total)
        delivered.append(float(env.delivered))
        complete.append(float(env.mission_complete))
        steps.append(env.steps)
        hits.append(env.obstacle_hits)

    def mean(values):
        return float(np.mean(values)) if len(values) else None

    return {
        "episodes": episodes,
        "avg_reward": mean(rewards),
        "std_reward": float(np.std(rewards)),
        "delivery_success": mean(delivered),
        "return_rate": mean(complete),
        "avg_steps": mean(steps),
        "avg_delivery_steps": mean(delivery_steps),
        "avg_battery_at_delivery": mean(delivery_battery),
        "avg_integrity_at_delivery": mean(delivery_integrity),
        "avg_tower_hits": mean(hits),
    }


def save_checkpoint(path, algorithm, network, hidden):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": algorithm,
            "hidden": list(hidden),
            "state_dict": network.state_dict(),
        },
        path,
    )


class RunLog:
    """Collects the learning curves and results of one training run."""

    def __init__(self, algorithm, name, config):
        self.algorithm = algorithm
        self.name = name
        self.config = {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()}
        self.history = []     # one entry per collected batch
        self.evals = []       # one entry per evaluation checkpoint
        self.start = time.time()
        self.next_eval = config["eval_every"]

    def add(self, **values):
        self.history.append({**values, "time": time.time() - self.start})

    def due_for_eval(self, frames):
        if frames >= self.next_eval:
            self.next_eval += self.config["eval_every"]
            return True
        return False

    def add_eval(self, frames, metrics):
        self.evals.append({"frames": frames, **metrics})

    def stability(self):
        """Standard deviation of training reward over the last 20% of training."""

        rewards = [h["train_reward"] for h in self.history if h.get("train_reward") is not None]
        tail = rewards[int(len(rewards) * 0.8):] or rewards
        return float(np.std(tail)) if tail else None


def log_line(verbose, algorithm, frames, total, train_reward, extra=""):
    if verbose:
        reward = "n/a" if train_reward is None else f"{train_reward:8.1f}"
        print(f"[{algorithm}] frames {frames:>7}/{total} | train reward {reward} {extra}")


# ---------------------------------------------------------------------------
# A2C
# ---------------------------------------------------------------------------

DEFAULTS = {
    **COMMON_DEFAULTS,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "entropy_coeff": 0.01,
    "value_coeff": 0.5,
    "grad_clip": 1.0,
    "reward_scale": 1.0,       # the learner sees reward * reward_scale
    "frames_per_batch": 200,   # environment steps per policy update
}


def train_a2c(config=None, run_name="a2c_default", model_path=None,
              verbose=True):
    """Train the drone with A2C and return the final evaluation."""

    cfg = {**DEFAULTS, **(config or {})}
    seed_everything(cfg["seed"])

    probe = make_env(reward_scale=cfg["reward_scale"])
    obs_size, n_actions = env_sizes()

    # Actor: observation -> action probabilities. Critic: observation -> value.
    actor_network = make_mlp(obs_size, n_actions, cfg["hidden"])
    actor = ProbabilisticActor(
        TensorDictModule(actor_network, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"],
        spec=probe.action_spec,
        distribution_class=Categorical,
        return_log_prob=True,
    )
    critic = ValueOperator(
        make_mlp(obs_size, 1, cfg["hidden"]), in_keys=["observation"]
    )

    advantage = GAE(
        gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic
    )
    loss_module = A2CLoss(
        actor,
        critic,
        entropy_bonus=True,
        entropy_coeff=cfg["entropy_coeff"],
        critic_coeff=cfg["value_coeff"],
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])

    collector = Collector(
        lambda: make_env(cfg["seed"], cfg["reward_scale"]),
        actor,
        frames_per_batch=cfg["frames_per_batch"],
        total_frames=cfg["total_frames"],
        device="cpu",
    )

    log = RunLog("a2c", run_name, cfg)
    frames = 0
    episodes = 0

    for batch in collector:
        frames += batch.numel()
        episode_rewards = finished_episode_rewards(batch)
        episodes += len(episode_rewards)

        with torch.no_grad():
            advantage(batch)

        loss_td = loss_module(batch)
        loss = (
            loss_td["loss_objective"]
            + loss_td["loss_critic"]
            + loss_td["loss_entropy"]
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["grad_clip"])
        optimizer.step()

        train_reward = (
            sum(episode_rewards) / len(episode_rewards) if episode_rewards else None
        )
        entropy_value = loss_td["entropy"].mean().item()
        log.add(
            frames=frames,
            episodes=episodes,
            train_reward=train_reward,
            loss=loss.item(),
            entropy=entropy_value,
        )

        if log.due_for_eval(frames):
            metrics = evaluate(greedy_action(actor_network), cfg["eval_episodes"], CHECKPOINT_SEED)
            log.add_eval(frames, metrics)
            log_line(
                verbose, "A2C", frames, cfg["total_frames"], train_reward,
                f"| eval reward {metrics['avg_reward']:7.1f} | "
                f"success {metrics['delivery_success']:.0%} | entropy {entropy_value:.2f}",
            )

    collector.shutdown()

    final = evaluate(greedy_action(actor_network), cfg["final_eval_episodes"], FINAL_EVAL_SEED)
    if model_path:
        save_checkpoint(model_path, "a2c", actor_network, cfg["hidden"])

    if verbose:
        print(f"\nA2C finished: reward {final['avg_reward']:.1f}, "
              f"delivery success {final['delivery_success']:.0%}")
    return {"final": final, "log": log, "network": actor_network}


if __name__ == "__main__":
    train_a2c(model_path="models/a2c/a2c.pt")
