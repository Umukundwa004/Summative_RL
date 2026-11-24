"""
DQN training with TorchRL.

DQN learns a score (Q-value) for every action: how much reward it expects
from that action. It explores at random at first, then mostly picks the
best-scoring action. It learns from a memory of past steps.
"""

import random
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import Compose, GymWrapper, RewardScaling, RewardSum, TransformedEnv
from torchrl.modules import EGreedyModule, MLP, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.objectives.utils import ValueEstimators

from environment.custom_env import BloodDroneEnv, mask_scores

ENV_SETTINGS = {"grid_size": 10, "max_steps": 300}
TRAIN_PRACTICE = 0.8      # chance of an easy start during training (fades out, see the environment)
FINAL_EVAL_SEED = 1000          # final evaluation missions (the same maps every time)
CHECKPOINT_SEED = 500     # checkpoint evaluation missions


DEFAULTS = {
    "hidden": (64, 64),         # two hidden layers of 64 units
    "total_frames": 300_000,    # environment steps for training
    "seed": 0,
    "eval_every": 5_000,        # frames between checkpoints
    "eval_episodes": 10,
    "final_eval_episodes": 50,
    "lr": 0.001,
    "gamma": 0.99,
    "eps_start": 1.0,           # chance of a random action at the start
    "eps_min": 0.05,
    "eps_decay": 0.995,         # applied after each finished episode
    "buffer_size": 10_000,
    "batch_size": 64,
    "target_update": 250,       # learning steps between copies of the target network
    "grad_clip": 1.0,
    "frames_per_batch": 100,
    "updates_per_batch": 50,
    "warmup": 1_000,            # steps of experience before learning starts
}


class MaskedActor(nn.Module):
    """A network that gives impossible actions a very low score."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, observation):
        return mask_scores(self.net(observation), observation)


def make_env(seed=None, reward_scale=1.0):
    """The environment as TorchRL sees it. A reward scale makes learning easier."""
    transforms = [RewardSum()]
    if reward_scale != 1.0:
        transforms.append(RewardScaling(loc=0.0, scale=reward_scale))
    env = TransformedEnv(
        GymWrapper(BloodDroneEnv(**ENV_SETTINGS, practice=TRAIN_PRACTICE),
                   categorical_action_encoding=True),
        Compose(*transforms),
    )
    if seed is not None:
        env.set_seed(seed)
    return env


def make_mlp(in_features, out_features, hidden):
    return MLP(in_features=in_features, out_features=out_features,
               num_cells=list(hidden), activation_class=nn.ReLU)


def greedy_action(network):
    """A function: observation -> the best possible action."""
    def act(observation):
        with torch.no_grad():
            scores = network(torch.as_tensor(observation, dtype=torch.float32))
            return int(scores.argmax().item())
    return act


def evaluate(act, episodes=10, seed_start=FINAL_EVAL_SEED):
    """Fly complete missions and measure them. Returns the report numbers."""
    env = BloodDroneEnv(**ENV_SETTINGS)
    rewards, delivered, complete, steps, hits, rounds = [], [], [], [], [], []
    delivery_steps, delivery_battery, delivery_integrity = [], [], []

    for k in range(episodes):
        observation, _ = env.reset(seed=seed_start + k)
        total, done, was_delivered = 0.0, False, False
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
        rounds.append(env.rounds_done)
        steps.append(env.steps)
        hits.append(env.obstacle_hits)

    def mean(values):
        return float(np.mean(values)) if values else None

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
        "avg_rounds_done": mean(rounds),
    }


def save_checkpoint(path, algorithm, network, hidden):
    """Save the network (without the mask wrapper) so the viewer can load it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plain = network.net if isinstance(network, MaskedActor) else network
    torch.save({"algorithm": algorithm, "hidden": list(hidden),
                "state_dict": plain.state_dict()}, path)


def train_dqn(config=None, run_name="dqn_default", model_path=None, verbose=True):
    """Train the drone with DQN. Returns the final evaluation."""
    cfg = {**DEFAULTS, **(config or {})}
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # The Q-network: one score per action, with impossible actions masked.
    spec = make_env().action_spec
    env_probe = BloodDroneEnv(**ENV_SETTINGS)
    obs_size, n_actions = env_probe.observation_space.shape[0], env_probe.action_space.n
    q_network = MaskedActor(make_mlp(obs_size, n_actions, cfg["hidden"]))

    value_module = TensorDictModule(q_network, in_keys=["observation"], out_keys=["action_value"])
    actor = QValueActor(value_module, spec=spec)
    exploration = EGreedyModule(spec=spec, eps_init=cfg["eps_start"],
                                eps_end=cfg["eps_min"], annealing_num_steps=1)
    policy = TensorDictSequential(actor, exploration)

    loss_module = DQNLoss(actor, action_space=spec, loss_function="l2", delay_value=True)
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg["gamma"])
    target_updater = HardUpdate(loss_module, value_network_update_interval=cfg["target_update"])
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]),
                                    batch_size=cfg["batch_size"])

    collector = Collector(lambda: make_env(cfg["seed"]), policy,
                          frames_per_batch=cfg["frames_per_batch"],
                          total_frames=cfg["total_frames"], device="cpu")

    epsilon = cfg["eps_start"]
    frames, next_eval, evals = 0, cfg["eval_every"], []

    for batch in collector:
        frames += batch.numel()
        buffer.extend(batch.reshape(-1))

        # Episodes that ended in this batch: their total rewards.
        done = batch["next", "done"].reshape(-1)
        episode_rewards = batch["next", "episode_reward"].reshape(-1)[done].tolist()
        for _ in episode_rewards:
            epsilon = max(cfg["eps_min"], epsilon * cfg["eps_decay"])
        exploration.eps.fill_(epsilon)

        if len(buffer) >= cfg["warmup"]:
            for _ in range(cfg["updates_per_batch"]):
                sample = buffer.sample()
                loss = loss_module(sample)["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["grad_clip"])
                optimizer.step()
                target_updater.step()

        if frames >= next_eval:
            next_eval += cfg["eval_every"]
            metrics = evaluate(greedy_action(q_network), cfg["eval_episodes"], CHECKPOINT_SEED)
            evals.append({"frames": frames, **metrics})
            if verbose:
                print(f"[DQN] frames {frames:>7}/{cfg['total_frames']} | "
                      f"eval reward {metrics['avg_reward']:7.1f} | "
                      f"success {metrics['delivery_success']:.0%} | eps {epsilon:.2f}")

    collector.shutdown()
    final = evaluate(greedy_action(q_network), cfg["final_eval_episodes"], FINAL_EVAL_SEED)
    if model_path:
        save_checkpoint(model_path, "dqn", q_network, cfg["hidden"])
    if verbose:
        print(f"\nDQN finished: reward {final['avg_reward']:.1f}, "
              f"delivery success {final['delivery_success']:.0%}")
    return {"final": final, "evals": evals, "network": q_network}


if __name__ == "__main__":
    train_dqn(model_path="models/dqn/dqn.pt")
