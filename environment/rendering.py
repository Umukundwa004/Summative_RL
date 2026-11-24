"""
2D rendering of the drone environment with pygame.

play.py uses this to show the simulation in a window. The 3D version of the
same simulation is in the frontend/ folder (Three.js).
"""

import pygame

from environment.custom_env import ACTION_NAMES

CELL = 60
PANEL = 110

COLORS = {
    "ground": (225, 230, 235),
    "tower": (200, 60, 60),
    "bank": (59, 130, 246),
    "hospital": (248, 250, 252),
    "charger": (34, 197, 94),
    "drone": (245, 158, 11),
    "text": (25, 30, 35),
    "good": (22, 150, 80),
    "bad": (200, 50, 50),
}


def window_size(env):
    return env.grid_size * CELL, env.grid_size * CELL + PANEL


def draw_environment(screen, env, font, last_action=None, last_reward=0.0):
    """Draw the map, the drone and a status panel."""

    screen.fill((245, 247, 250))

    for y in range(env.grid_size):
        for x in range(env.grid_size):
            cell = (x, y)
            rect = pygame.Rect(x * CELL, y * CELL, CELL - 2, CELL - 2)

            color = COLORS["ground"]
            if cell in env.obstacles:
                color = COLORS["tower"]
            elif cell == env.bank:
                color = COLORS["bank"]
            elif cell == env.hospital:
                color = COLORS["hospital"]
            elif cell == env.charger:
                color = COLORS["charger"]
            pygame.draw.rect(screen, color, rect)

            # Blue tint where it is raining (rain is local, not map-wide).
            cell_rain = env.weather_at(cell)[1]
            if cell_rain > 1.0:
                shade = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                shade.fill((59, 111, 201, int(min(cell_rain / 8, 1.0) * 150)))
                screen.blit(shade, rect)

            if cell == env.hospital:
                pygame.draw.rect(screen, (220, 38, 38), (rect.centerx - 3, rect.y + 12, 6, CELL - 26))
                pygame.draw.rect(screen, (220, 38, 38), (rect.x + 12, rect.centery - 3, CELL - 26, 6))
            elif cell in env.obstacles:
                pygame.draw.polygon(
                    screen, (255, 255, 255),
                    [(rect.centerx, rect.y + 8), (rect.x + 14, rect.bottom - 8),
                     (rect.right - 14, rect.bottom - 8)], 3,
                )

    # Labels.
    for cell, text in ((env.bank, "BANK"), (env.hospital, ""), (env.charger, "CHARGE")):
        if text:
            label = font.render(text, True, (255, 255, 255))
            screen.blit(label, (cell[0] * CELL + 4, cell[1] * CELL + CELL - 22))

    # The drone (carrying a red package until delivery).
    dx, dy = env.drone_position
    center = (dx * CELL + CELL // 2, dy * CELL + CELL // 2)
    pygame.draw.circle(screen, COLORS["drone"], center, CELL // 3)
    pygame.draw.circle(screen, (30, 30, 30), center, CELL // 3, 2)
    if env.package_collected and not env.delivered:
        pygame.draw.rect(screen, (220, 38, 38), (center[0] - 7, center[1] + 8, 14, 12))

    # Status panel.
    top = env.grid_size * CELL + 8
    action = "-" if last_action is None else ACTION_NAMES[last_action]
    reward_color = COLORS["good"] if last_reward >= 0 else COLORS["bad"]

    lines = [
        (f"Step {env.steps}/{env.max_steps}    Action: {action}", COLORS["text"]),
        (f"Reward {last_reward:+.1f}    Total {env.total_reward:+.1f}", reward_color),
        (f"Battery {env.battery:.0f}%    Integrity {env.package_integrity:.0f}%    "
         f"Tower hits {env.obstacle_hits}/3", COLORS["text"]),
        (f"Wind {env.wind:.1f}   Rain {env.rain:.1f}   Blood temp {env.package_temperature:.1f} C",
         COLORS["text"]),
        (f"Event: {env.last_event}", COLORS["text"]),
    ]
    for i, (text, color) in enumerate(lines):
        screen.blit(font.render(text, True, color), (10, top + i * 20))

    pygame.display.flip()
