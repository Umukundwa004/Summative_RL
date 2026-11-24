"""
REINFORCE training with TorchRL.

REINFORCE is the simplest policy-gradient method. After each batch of
complete episodes, it makes actions that led to a high total reward more
likely, and actions that led to a low total reward less likely.
"""

import random
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.envs import Compose, GymWrapper, RewardScaling, RewardSum, TransformedEnv
from torchrl.modules import MLP, ProbabilisticActor
from torchrl.objectives.value.functional import reward2go

from environment.custom_env import BloodDroneEnv

from training.dqn_training import (
    CHECKPOINT_SEED, ENV_SETTINGS, FINAL_EVAL_SEED, MaskedActor,
    evaluate, greedy_action, make_env, make_mlp, save_checkpoint,
)

DEFAULTS = {
    "hidden": (128, 128),       # two hidden layers of 128 units
    "total_frames": 300_000,    # environment steps for training
    "seed": 0,
    "eval_every": 5_000,
    "eval_episodes": 10,
    "final_eval_episodes": 50,
    "lr": 0.001,
    "gamma": 0.99,
    "normalize_returns": True,  # scale the returns to mean 0, spread 1: steadier learning
    "entropy_coeff": 0.01,      # bonus for trying different actions
    "grad_clip": 1.0,
    "reward_scale": 1.0,        # the learner sees reward * reward_scale
    "frames_per_batch": 256,    # environment steps per policy update
}


def train_reinforce(config=None, run_name="reinforce_default", model_path=None, verbose=True):
    """Train the drone with REINFORCE. Returns the final evaluation."""
    cfg = {**DEFAULTS, **(config or {})}
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    spec = make_env(reward_scale=cfg["reward_scale"]).action_spec
    env_probe = BloodDroneEnv(**ENV_SETTINGS)
    obs_size, n_actions = env_probe.observation_space.shape[0], env_probe.action_space.n

    # The policy network: observation -> action scores (softmax gives probabilities).
    policy_network = MaskedActor(make_mlp(obs_size, n_actions, cfg["hidden"]))
    actor = ProbabilisticActor(
        TensorDictModule(policy_network, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"], spec=spec, distribution_class=Categorical, return_log_prob=True,
    )
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=cfg["lr"])

    collector = Collector(lambda: make_env(cfg["seed"], cfg["reward_scale"]), actor,
                          frames_per_batch=cfg["frames_per_batch"],
                          total_frames=cfg["total_frames"], device="cpu")

    frames, next_eval, evals = 0, cfg["eval_every"], []

    for batch in collector:
        frames += batch.numel()
        done = batch["next", "done"].reshape(-1)
        episode_rewards = batch["next", "episode_reward"].reshape(-1)[done].tolist()

        # Learn only from complete episodes: stop at the last one that finished.
        finished = done.nonzero()
        entropy_value = None
        if len(finished) > 0:
            end = int(finished[-1]) + 1
            observations = batch["observation"][:end]
            actions = batch["action"][:end]
            rewards = batch["next", "reward"][:end]
            ends = batch["next", "done"][:end]

            # The reward from each step onward, discounted.
            returns = reward2go(rewards, ends, gamma=cfg["gamma"], time_dim=-2).squeeze(-1)
            if cfg["normalize_returns"] and returns.numel() > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            distribution = Categorical(logits=policy_network(observations))
            entropy = distribution.entropy().mean()
            loss = -(distribution.log_prob(actions) * returns).mean() - cfg["entropy_coeff"] * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_network.parameters(), cfg["grad_clip"])
            optimizer.step()
            entropy_value = entropy.item()

        if frames >= next_eval:
            next_eval += cfg["eval_every"]
            metrics = evaluate(greedy_action(policy_network), cfg["eval_episodes"], CHECKPOINT_SEED)
            evals.append({"frames": frames, **metrics})
            if verbose:
                entropy_text = "n/a" if entropy_value is None else f"{entropy_value:.2f}"
                print(f"[REINFORCE] frames {frames:>7}/{cfg['total_frames']} | "
                      f"eval reward {metrics['avg_reward']:7.1f} | "
                      f"success {metrics['delivery_success']:.0%} | "
                      f"finished {metrics['return_rate']:.0%} | "
                      f"rounds {metrics['avg_rounds_done']:.2f} | entropy {entropy_text}")

    collector.shutdown()
    final = evaluate(greedy_action(policy_network), cfg["final_eval_episodes"], FINAL_EVAL_SEED)
    if model_path:
        save_checkpoint(model_path, "reinforce", policy_network, cfg["hidden"])
    if verbose:
        print(f"\nREINFORCE finished: reward {final['avg_reward']:.1f}, "
              f"delivery success {final['delivery_success']:.0%}")
    return {"final": final, "evals": evals, "network": policy_network}


if __name__ == "__main__":
    train_reinforce(model_path="models/reinforce/reinforce.pt")
