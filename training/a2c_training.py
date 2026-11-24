"""
A2C training with TorchRL.

A2C is an actor-critic method. The actor (the policy) chooses actions, and the
critic estimates how good a situation is. Each update uses the most recent
steps only, so learning is quick.
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
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss
from torchrl.objectives.value import GAE

from environment.custom_env import BloodDroneEnv

from training.dqn_training import (
    CHECKPOINT_SEED, ENV_SETTINGS, FINAL_EVAL_SEED, MaskedActor,
    evaluate, greedy_action, make_env, make_mlp, save_checkpoint,
)

DEFAULTS = {
    "hidden": (64, 64),         # two hidden layers of 64 units
    "total_frames": 300_000,    # environment steps for training
    "seed": 0,
    "eval_every": 5_000,
    "eval_episodes": 10,
    "final_eval_episodes": 50,
    "lr": 0.0003,
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "entropy_coeff": 0.01,      # bonus for trying different actions
    "value_coeff": 0.5,         # weight of the critic's loss
    "grad_clip": 1.0,
    "reward_scale": 1.0,        # the learner sees reward * reward_scale
    "frames_per_batch": 200,    # environment steps per policy update
}


def train_a2c(config=None, run_name="a2c_default", model_path=None, verbose=True):
    """Train the drone with A2C. Returns the final evaluation."""
    cfg = {**DEFAULTS, **(config or {})}
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    spec = make_env(reward_scale=cfg["reward_scale"]).action_spec
    env_probe = BloodDroneEnv(**ENV_SETTINGS)
    obs_size, n_actions = env_probe.observation_space.shape[0], env_probe.action_space.n

    # The actor chooses actions. The critic estimates how good a situation is.
    actor_network = MaskedActor(make_mlp(obs_size, n_actions, cfg["hidden"]))
    actor = ProbabilisticActor(
        TensorDictModule(actor_network, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"], spec=spec, distribution_class=Categorical, return_log_prob=True,
    )
    critic = ValueOperator(make_mlp(obs_size, 1, cfg["hidden"]), in_keys=["observation"])

    advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic)
    loss_module = A2CLoss(actor, critic, entropy_bonus=True,
                          entropy_coeff=cfg["entropy_coeff"], critic_coeff=cfg["value_coeff"])
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])

    collector = Collector(lambda: make_env(cfg["seed"], cfg["reward_scale"]), actor,
                          frames_per_batch=cfg["frames_per_batch"],
                          total_frames=cfg["total_frames"], device="cpu")

    frames, next_eval, evals = 0, cfg["eval_every"], []

    for batch in collector:
        frames += batch.numel()
        done = batch["next", "done"].reshape(-1)
        episode_rewards = batch["next", "episode_reward"].reshape(-1)[done].tolist()

        with torch.no_grad():
            advantage(batch)

        loss_td = loss_module(batch)
        loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["grad_clip"])
        optimizer.step()
        entropy_value = loss_td["entropy"].mean().item()

        if frames >= next_eval:
            next_eval += cfg["eval_every"]
            metrics = evaluate(greedy_action(actor_network), cfg["eval_episodes"], CHECKPOINT_SEED)
            evals.append({"frames": frames, **metrics})
            if verbose:
                print(f"[A2C] frames {frames:>7}/{cfg['total_frames']} | "
                      f"eval reward {metrics['avg_reward']:7.1f} | "
                      f"success {metrics['delivery_success']:.0%} | "
                      f"finished {metrics['return_rate']:.0%} | "
                      f"rounds {metrics['avg_rounds_done']:.2f} | entropy {entropy_value:.2f}")

    collector.shutdown()
    final = evaluate(greedy_action(actor_network), cfg["final_eval_episodes"], FINAL_EVAL_SEED)
    if model_path:
        save_checkpoint(model_path, "a2c", actor_network, cfg["hidden"])
    if verbose:
        print(f"\nA2C finished: reward {final['avg_reward']:.1f}, "
              f"delivery success {final['delivery_success']:.0%}")
    return {"final": final, "evals": evals, "network": actor_network}


if __name__ == "__main__":
    train_a2c(model_path="models/a2c/a2c.pt")
