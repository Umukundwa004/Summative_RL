"""
Watch a drone fly its mission in a window.

Usage:
    uv run python play.py --model dqn           # trained DQN
    uv run python play.py --model ppo --fps 8   # trained PPO, faster
    uv run python play.py --model random        # random actions, no model needed
    uv run python play.py --model manual        # you fly: arrows, C, R, SPACE

The terminal prints every action and its reward.
"""

import argparse
from pathlib import Path

import pygame

from environment.custom_env import ACTION_NAMES, BloodDroneEnv
from environment.rendering import draw_environment, window_size
from export_episode import load_policy, model_path

# Keys for manual mode.
KEY_ACTIONS = {
    pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_RIGHT: 2, pygame.K_LEFT: 3,
    pygame.K_c: 4, pygame.K_r: 5, pygame.K_SPACE: 6,
}


def choose_action(args, env, observation, act):
    """Pick the next action, depending on the mode."""
    if args.model == "random":
        return env.action_space.sample()
    return act(observation)


def main():
    parser = argparse.ArgumentParser(description="Watch the blood delivery drone")
    parser.add_argument("--model", default="dqn",
                        choices=["dqn", "reinforce", "ppo", "a2c", "random", "manual"])
    parser.add_argument("--fps", type=int, default=5, help="steps per second")
    parser.add_argument("--episodes", type=int, default=0, help="stop after this many missions (0 = forever)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    env = BloodDroneEnv()
    act = None
    if args.model not in ("random", "manual"):
        if not Path(model_path(args.model)).exists():
            raise SystemExit(f"No trained model at {model_path(args.model)}. Train one first: uv run main.py")
        act = load_policy(model_path(args.model))

    pygame.init()
    screen = pygame.display.set_mode(window_size(env))
    pygame.display.set_caption(f"Blood Delivery Drone - {args.model.upper()}")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

    observation, _ = env.reset(seed=args.seed)
    last_action, last_reward = None, 0.0
    missions = 0
    print(f"\nModel: {args.model.upper()}  |  deadline {env.max_steps} steps")
    print("-" * 78)
    draw_environment(screen, env, font, last_action, last_reward)

    running = True
    while running:
        key_action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                key_action = KEY_ACTIONS.get(event.key)

        if args.model == "manual":
            if key_action is None:
                clock.tick(30)
                continue
            action = key_action
        else:
            action = choose_action(args, env, observation, act)

        observation, reward, terminated, truncated, info = env.step(action)
        last_action, last_reward = action, reward
        print(f"step {env.steps:>3} | {ACTION_NAMES[action]:<18} | reward {reward:>+7.1f} | "
              f"total {info['total_reward']:>+8.1f} | battery {env.battery:>5.1f} | "
              f"integrity {env.package_integrity:>5.1f} | {info['event']}")
        draw_environment(screen, env, font, last_action, last_reward)
        clock.tick(30 if args.model == "manual" else args.fps)

        if terminated or truncated:
            missions += 1
            if env.mission_complete:
                result = "MISSION COMPLETE: blood delivered and drone back at base"
            elif env.delivered:
                result = "Blood delivered, but the drone did not return"
            else:
                result = f"MISSION FAILED ({info['event']})"
            print("-" * 78)
            print(f"{result} | total reward {info['total_reward']:+.1f} | tower hits {env.obstacle_hits}\n")
            if args.episodes and missions >= args.episodes:
                break

            pygame.time.wait(1500)
            observation, _ = env.reset()
            last_action, last_reward = None, 0.0
            draw_environment(screen, env, font, last_action, last_reward)

    pygame.quit()


if __name__ == "__main__":
    main()
