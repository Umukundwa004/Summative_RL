"""
See the environment: print the map in the terminal and show / save the 2D picture.

    uv run render_environment.py              # map in the terminal + a window (close it to quit)
    uv run render_environment.py --save       # only write assets/environment_2d.png
    uv run render_environment.py --seed 4     # a different storm position
"""

import argparse
import os
from pathlib import Path

from environment.custom_env import BloodDroneEnv

IMAGE_PATH = Path("assets") / "environment_2d.png"


def print_map(env):
    """Print the map as text.  B bank, H hospital, C charger, # tower, D drone, ~ rain."""
    print()
    print("  " + " ".join(str(x) for x in range(env.grid_size)))
    for y in range(env.grid_size):
        row = []
        for x in range(env.grid_size):
            cell = (x, y)
            if cell == env.drone_position:
                row.append("D")
            elif cell == env.bank:
                row.append("B")
            elif cell == env.hospital:
                row.append("H")
            elif cell == env.charger:
                row.append("C")
            elif cell in env.obstacles:
                row.append("#")
            elif env.weather_at(cell)[1] > 1.0:
                row.append("~")
            else:
                row.append(".")
        print(f"{y} " + " ".join(row))
    print()
    print("B bank   H hospital   C charging station   # tower   ~ rain   D drone")
    print(f"Wind {env.wind:.1f}   Rain {env.rain:.1f}   Storms: {len(env.storms)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Show the blood delivery drone environment")
    parser.add_argument("--save", action="store_true", help="only save the picture, do not open a window")
    parser.add_argument("--seed", type=int, default=1, help="random seed (changes where the storm is)")
    args = parser.parse_args()

    if args.save:
        os.environ["SDL_VIDEODRIVER"] = "dummy"   # draw without opening a window

    env = BloodDroneEnv()
    env.reset(seed=args.seed)
    for storm in env.storms:
        storm["age"] = 6                      # let the storm grow, so the rain is visible
    env.wind, env.rain = env.weather_at(env.drone_position)
    print_map(env)

    import pygame
    from environment.rendering import draw_environment, window_size

    pygame.init()
    screen = pygame.display.set_mode(window_size(env))
    pygame.display.set_caption("Blood delivery drone: environment")
    font = pygame.font.SysFont("consolas", 16)
    draw_environment(screen, env, font)

    IMAGE_PATH.parent.mkdir(exist_ok=True)
    pygame.image.save(screen, str(IMAGE_PATH))
    print(f"Saved {IMAGE_PATH}")

    if not args.save:
        print("Close the window to quit.")
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
