"""
Export one mission per trained model as JSON for the 3D frontend.

Each trained network from models/<name>/<name>.pt is run for one mission and
saved to frontend/public/episodes/<name>.json. The frontend lists these four
models in its drop-down.

Examples:
    python export_episode.py              # export all four models
    python export_episode.py --model dqn  # export one model
"""

import argparse
import json
from pathlib import Path

from environment.custom_env import ACTION_NAMES, BloodDroneEnv


def model_path(name):
    return Path("models") / name / f"{name}.pt"


def load_policy(path):
    """Load a saved model and return a function: observation -> best action."""

    import torch
    from torch import nn
    from torchrl.modules import MLP

    checkpoint = torch.load(path, map_location="cpu")
    env = BloodDroneEnv()
    network = MLP(
        in_features=env.observation_space.shape[0],
        out_features=env.action_space.n,
        num_cells=list(checkpoint["hidden"]),
        activation_class=nn.ReLU,
    )
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()

    def act(observation):
        with torch.no_grad():
            scores = network(torch.as_tensor(observation, dtype=torch.float32))
            return int(scores.argmax().item())

    return act


OUT_DIR = Path("frontend/public/episodes")

# id -> label shown in the frontend
MODELS = {
    "dqn": "DQN",
    "reinforce": "REINFORCE",
    "ppo": "PPO",
    "a2c": "A2C",
}


def frame(env, action, reward, event):
    return {
        "step": env.steps,
        "x": env.drone_position[0],
        "y": env.drone_position[1],
        "action": action,
        "action_name": ACTION_NAMES[action] if action is not None else "Start",
        "reward": round(float(reward), 2),
        "total_reward": round(float(env.total_reward), 2),
        "event": event,
        "battery": round(float(env.battery), 1),
        "temperature": round(float(env.package_temperature), 2),
        "integrity": round(float(env.package_integrity), 1),
        "wind": round(float(env.wind), 2),
        "rain": round(float(env.rain), 2),
        "collected": bool(env.package_collected),
        "delivered": bool(env.delivered),
        "complete": bool(env.mission_complete),
        "hits": int(env.obstacle_hits),
    }


def run_episode(act, seed=0):
    env = BloodDroneEnv()
    observation, _ = env.reset(seed=seed)

    frames = [frame(env, None, 0.0, "start")]
    done = False

    while not done:
        action = act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(frame(env, action, reward, info["event"]))
        done = terminated or truncated

    layout = {
        "grid_size": env.grid_size,
        "bank": list(env.bank),
        "hospital": list(env.hospital),
        "charger": list(env.charger),
        "obstacles": sorted(list(c) for c in env.obstacles),
    }
    return {"layout": layout, "frames": frames}


def export(name):
    try:
        act = load_policy(model_path(name))
    except Exception as error:
        print(f"  {name:<10} skipped ({type(error).__name__}: {error})")
        return False

    episode = run_episode(act)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(episode), encoding="utf-8")

    last = episode["frames"][-1]
    print(
        f"  {name:<10} steps={last['step']:<4} reward={last['total_reward']:<8} "
        f"hits={last['hits']} delivered={last['delivered']} "
        f"returned={last['complete']}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Export missions for the 3D frontend")
    parser.add_argument(
        "--model", choices=list(MODELS), default=None,
        help="export one model (default: all four)",
    )
    args = parser.parse_args()

    print(f"Writing to {OUT_DIR}")
    for name in [args.model] if args.model else list(MODELS):
        export(name)

    # The drop-down lists only the four models that have an exported episode.
    index = [
        {"id": name, "label": label}
        for name, label in MODELS.items()
        if (OUT_DIR / f"{name}.json").exists()
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
