"""
Watch a trained drone fly its mission.

Usage:
    uv run python play.py --model dqn          # best trained DQN
    uv run python play.py --model ppo --fps 8
    uv run python play.py --model random       # random actions, no model needed
    uv run python play.py --model manual       # you fly: arrows, C, R, SPACE

The window shows the drone, towers, blood bank, hospital and charging station.
The terminal prints every action with its reward, so you can follow how the
agent behaves.
"""

import argparse
from pathlib import Path

import pygame

from environment.custom_env import ACTION_NAMES, BloodDroneEnv
from environment.rendering import draw_environment, window_size


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


# Keys used in manual mode.
KEY_ACTIONS = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_RIGHT: 2,
    pygame.K_LEFT: 3,
    pygame.K_c: 4,
    pygame.K_r: 5,
    pygame.K_SPACE: 6,
}


def main():
    parser = argparse.ArgumentParser(description="Watch the blood delivery drone")
    parser.add_argument(
        "--model", default="dqn",
        choices=["dqn", "reinforce", "ppo", "a2c", "random", "manual"],
        help="trained model, random actions, or manual control",
    )
    parser.add_argument("--fps", type=int, default=5, help="steps per second")
    parser.add_argument("--episodes", type=int, default=0,
                        help="stop after this many missions (0 = keep going)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    env = BloodDroneEnv()

    act = None
    if args.model not in ("random", "manual"):
        path = model_path(args.model)
        if not Path(path).exists():
            raise SystemExit(
                f"No trained model at {path}. Train one first: uv run main.py"
            )
        act = load_policy(path)

    pygame.init()
    screen = pygame.display.set_mode(window_size(env))
    pygame.display.set_caption(f"Blood Delivery Drone - {args.model.upper()}")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    observation, _ = env.reset(seed=args.seed)
    last_action, last_reward = None, 0.0
    missions = 0
    running = True

    print(f"\nModel: {args.model.upper()}  |  map {env.grid_size}x{env.grid_size}  |  "
          f"deadline {env.max_steps} steps")
    print("-" * 78)
    draw_environment(screen, env, font, last_action, last_reward)

    while running:
        manual_action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                manual_action = KEY_ACTIONS.get(event.key)

        if args.model == "manual":
            if manual_action is None:
                clock.tick(30)
                continue
            action = manual_action
        elif args.model == "random":
            action = env.action_space.sample()
        else:
            action = act(observation)

        observation, reward, terminated, truncated, info = env.step(action)
        last_action, last_reward = action, reward

        print(
            f"step {env.steps:>3} | {ACTION_NAMES[action]:<18} | reward {reward:>+7.1f} | "
            f"total {info['total_reward']:>+8.1f} | battery {env.battery:>5.1f} | "
            f"integrity {env.package_integrity:>5.1f} | {info['event']}"
        )
        draw_environment(screen, env, font, last_action, last_reward)
        clock.tick(args.fps if args.model != "manual" else 30)

        if terminated or truncated:
            missions += 1
            if env.mission_complete:
                result = "MISSION COMPLETE: blood delivered and drone back at base"
            elif env.delivered:
                result = "Blood delivered, but the drone did not return"
            else:
                result = f"MISSION FAILED ({info['event']})"
            print("-" * 78)
            print(f"{result} | total reward {info['total_reward']:+.1f} | "
                  f"tower hits {env.obstacle_hits}\n")

            if args.episodes and missions >= args.episodes:
                break

            pygame.time.wait(1500)
            observation, _ = env.reset()
            last_action, last_reward = None, 0.0
            draw_environment(screen, env, font, last_action, last_reward)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
