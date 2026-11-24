"""
Export missions from the trained models as JSON for the 3D viewer.

Each model in models/<name>/<name>.pt flies one complete mission per seed.
The missions are saved to frontend/public/episodes/.

Examples:
    python export_episode.py                          # all models, seeds 0-3
    python export_episode.py --model dqn --seeds 4 5  # one model, seeds 4 and 5
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchrl.modules import MLP

from environment.custom_env import ACTION_NAMES, BloodDroneEnv, mask_scores, rain_level

OUT_DIR = Path("frontend/public/episodes")
MODELS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}


def model_path(name):
    return Path("models") / name / f"{name}.pt"


def load_policy(path):
    """Load a saved model. Returns a function: observation -> best possible action."""
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
        # The same rule as in training: impossible actions are never chosen.
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32)
            return int(mask_scores(network(obs), obs).argmax().item())

    return act


def layout_of(env):
    """The map of one mission."""
    return {
        "grid_size": env.grid_size,
        "bank": list(env.bank),
        "charger": list(env.charger),
        "hospital": list(env.hospital),
        "hospitals": [list(h) for h in env.hospital_options],
        "shelters": sorted(list(c) for c in env.shelters),
        "obstacles": sorted([x, y, h] for (x, y), h in env.obstacles.items()),  # height 4 = building
    }


def frame(env, action, reward, event):
    """Everything the viewer needs for one step."""
    return {
        "step": env.steps,
        "x": env.drone_position[0],
        "y": env.drone_position[1],
        "altitude": env.altitude,
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
        "weather_level": rain_level(env.rain),
        "storms": [
            {"x": s["x"], "y": s["y"], "radius": s["radius"],
             "intensity": round(min(s["age"] / 4, (s["life"] - s["age"]) / 4, 1.0), 2)}
            for s in env.storms
        ],
        "hospital": list(env.hospital),
        "collected": bool(env.package_collected),
        "round_delivered": bool(env.round_delivered),
        "rounds_done": int(env.rounds_done),
        "delivered": bool(env.delivered),
        "complete": bool(env.mission_complete),
        "hits": int(env.obstacle_hits),
    }


def run_episode(act, seed=0):
    """Fly one complete mission with the trained policy."""
    env = BloodDroneEnv()
    observation, _ = env.reset(seed=seed)
    layout = layout_of(env)
    frames = [frame(env, None, 0.0, "start")]

    done = False
    while not done:
        action = act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(frame(env, action, reward, info["event"]))
        done = terminated or truncated

    last = frames[-1]
    summary = {
        "seed": seed,
        "total_reward": last["total_reward"],
        "steps": last["step"],
        "rounds_done": last["rounds_done"],
        "mission_complete": last["complete"],
        "outcome": last["event"],   # how it ended, e.g. "returned" or "crash"
    }
    return {"summary": summary, "layout": layout, "frames": frames}


def episode_path(name, seed, out_dir=OUT_DIR):
    """Seed 0 keeps the plain name (dqn.json); other seeds get their own file."""
    suffix = "" if seed == 0 else f"-seed-{seed}"
    return out_dir / f"{name}{suffix}.json"


def build_episode_index(out_dir=OUT_DIR):
    """List every exported episode, in the order the viewer shows them."""
    index = []
    for name, label in MODELS.items():
        seeds = []
        if episode_path(name, 0, out_dir).exists():
            seeds.append((0, name))
        prefix = f"{name}-seed-"
        for path in out_dir.glob(f"{prefix}*.json"):
            try:
                seeds.append((int(path.stem[len(prefix):]), path.stem))
            except ValueError:
                continue
        for seed, episode_id in sorted(set(seeds)):
            index.append({"id": episode_id, "label": f"{label} (seed {seed})"})
    return index


def export(name, seeds):
    try:
        act = load_policy(model_path(name))
    except Exception as error:
        print(f"  {name:<10} skipped ({type(error).__name__}: {error})")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        episode = run_episode(act, seed)
        episode_path(name, seed).write_text(json.dumps(episode), encoding="utf-8")
        s = episode["summary"]
        print(f"  {name:<10} seed={seed:<4} steps={s['steps']:<4} "
              f"total reward={s['total_reward']:<8} rounds={s['rounds_done']} outcome={s['outcome']}")


def main():
    parser = argparse.ArgumentParser(description="Export missions for the 3D viewer")
    parser.add_argument("--model", choices=list(MODELS), default=None, help="one model (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3], help="episode seeds")
    args = parser.parse_args()

    print(f"Writing to {OUT_DIR}")
    for name in [args.model] if args.model else list(MODELS):
        export(name, list(dict.fromkeys(args.seeds)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.json").write_text(json.dumps(build_episode_index(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
