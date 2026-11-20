"""
DQN training (value-based), built on TorchRL.

DQN learns a Q-value for every action: the reward the drone expects to collect
if it takes that action now and plays well afterwards. It uses
    - epsilon-greedy exploration (random actions early, best actions later),
    - a replay buffer of past experience,
    - a target network that is copied from the online network every few updates.
"""

import random
import time
from pathlib import Path

import numpy as np
import torch
from environment.custom_env import BloodDroneEnv
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import Compose, GymWrapper, RewardScaling, RewardSum, TransformedEnv
from torchrl.modules import EGreedyModule, MLP, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.objectives.utils import ValueEstimators


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
# DQN
# ---------------------------------------------------------------------------

DEFAULTS = {
    **COMMON_DEFAULTS,
    "lr": 1e-3,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_min": 0.05,
    "eps_decay": 0.995,        # multiplied in after every finished episode
    "buffer_size": 10_000,
    "batch_size": 64,
    "target_update": 250,      # gradient steps between target network copies
    "grad_clip": 1.0,
    "frames_per_batch": 100,   # environment steps collected per iteration
    "updates_per_batch": 50,   # gradient steps per iteration
    "warmup": 1_000,           # random-ish experience before learning starts
}


def train_dqn(config=None, run_name="dqn_default", model_path=None,
              verbose=True):
    """Train the drone with DQN and return the final evaluation."""

    cfg = {**DEFAULTS, **(config or {})}
    seed_everything(cfg["seed"])

    # Environment, Q-network and epsilon-greedy exploration.
    probe = make_env()
    spec = probe.action_spec
    obs_size, n_actions = env_sizes()

    q_network = make_mlp(obs_size, n_actions, cfg["hidden"])
    value_module = TensorDictModule(
        q_network, in_keys=["observation"], out_keys=["action_value"]
    )
    actor = QValueActor(value_module, spec=spec)
    exploration = EGreedyModule(
        spec=spec,
        eps_init=cfg["eps_start"],
        eps_end=cfg["eps_min"],
        annealing_num_steps=1,
    )
    policy = TensorDictSequential(actor, exploration)

    # Loss: mean squared error between the Q-value and the TD target.
    loss_module = DQNLoss(
        actor, action_space=spec, loss_function="l2", delay_value=True
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg["gamma"])
    target_updater = HardUpdate(
        loss_module, value_network_update_interval=cfg["target_update"]
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg["buffer_size"]),
        batch_size=cfg["batch_size"],
    )

    collector = Collector(
        lambda: make_env(cfg["seed"]),
        policy,
        frames_per_batch=cfg["frames_per_batch"],
        total_frames=cfg["total_frames"],
        device="cpu",
    )

    log = RunLog("dqn", run_name, cfg)
    epsilon = cfg["eps_start"]
    frames = 0
    episodes = 0

    for batch in collector:
        frames += batch.numel()
        buffer.extend(batch.reshape(-1))

        # Decay epsilon once per finished episode.
        episode_rewards = finished_episode_rewards(batch)
        for _ in episode_rewards:
            episodes += 1
            epsilon = max(cfg["eps_min"], epsilon * cfg["eps_decay"])
        exploration.eps.fill_(epsilon)

        losses, q_values = [], []
        if len(buffer) >= cfg["warmup"]:
            for _ in range(cfg["updates_per_batch"]):
                sample = buffer.sample()
                loss_td = loss_module(sample)
                loss = loss_td["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["grad_clip"])
                optimizer.step()
                target_updater.step()

                losses.append(loss.item())
                q_values.append(sample["action_value"].mean().item()
                                if "action_value" in sample.keys() else 0.0)

        train_reward = (
            sum(episode_rewards) / len(episode_rewards) if episode_rewards else None
        )
        log.add(
            frames=frames,
            episodes=episodes,
            train_reward=train_reward,
            loss=sum(losses) / len(losses) if losses else None,
            epsilon=epsilon,
        )

        if log.due_for_eval(frames):
            metrics = evaluate(greedy_action(q_network), cfg["eval_episodes"], CHECKPOINT_SEED)
            log.add_eval(frames, metrics)
            log_line(
                verbose, "DQN", frames, cfg["total_frames"], train_reward,
                f"| eval reward {metrics['avg_reward']:7.1f} | "
                f"success {metrics['delivery_success']:.0%} | eps {epsilon:.2f}",
            )

    collector.shutdown()

    final = evaluate(greedy_action(q_network), cfg["final_eval_episodes"], FINAL_EVAL_SEED)
    if model_path:
        save_checkpoint(model_path, "dqn", q_network, cfg["hidden"])

    if verbose:
        print(f"\nDQN finished: reward {final['avg_reward']:.1f}, "
              f"delivery success {final['delivery_success']:.0%}")
    return {"final": final, "log": log, "network": q_network}


if __name__ == "__main__":
    train_dqn(model_path="models/dqn/dqn.pt")
