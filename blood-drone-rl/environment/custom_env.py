"""
Blood Delivery Drone Environment

A simulation of a medical drone (similar to Zipline in Rwanda) that
must collect blood at the blood bank, deliver it to the hospital and
fly safely back to base.

Along the way the drone must:
    - avoid many obstacles (towers, buildings, no-fly zones),
    - recharge at the charging station when its battery is low,
    - wait out storms, because wind and rain make flying dangerous,
    - keep the blood package at a safe temperature,
    - finish before the deadline.

Actions (7):
    0 = Move north       4 = Collect / Deliver
    1 = Move south       5 = Recharge
    2 = Move east        6 = Wait
    3 = Move west

All four reinforcement learning algorithms use this environment.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ACTION_NAMES = [
    "Move north",
    "Move south",
    "Move east",
    "Move west",
    "Collect / Deliver",
    "Recharge",
    "Wait",
]

# (dx, dy) for the four movement actions. North is y - 1.
MOVES = {
    0: (0, -1),
    1: (0, 1),
    2: (1, 0),
    3: (-1, 0),
}

# Reward table.
REWARD_COLLECT = 15.0
REWARD_DELIVER = 100.0
REWARD_RETURN = 15.0
REWARD_PROGRESS = 2.0       # multiplied by the distance improvement
REWARD_RECHARGE = 5.0
REWARD_STEP = -0.1          # small cost for every step
PENALTY_INVALID = -3.0
PENALTY_OBSTACLE = -20.0
PENALTY_DEADLINE = -40.0
PENALTY_UNSAFE_TEMP = -60.0
PENALTY_CRASH = -100.0
PENALTY_BATTERY = -100.0

MAX_OBSTACLE_HITS = 3       # the third collision destroys the drone


class BloodDroneEnv(gym.Env):
    """Autonomous blood delivery drone on a 10 x 10 map."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size=10, max_steps=200):

        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps

        # Fixed map layout.
        self.bank = (0, 0)
        self.hospital = (9, 9)
        self.charger = (5, 5)

        # Towers, buildings and no-fly zones. A path always exists
        # between the bank, the charging station and the hospital.
        self.obstacles = {
            (2, 1), (2, 2), (2, 3),
            (4, 0), (4, 1),
            (1, 5), (2, 5), (3, 5),
            (4, 3), (5, 2), (6, 2), (6, 3),
            (7, 5), (7, 6), (8, 6),
            (4, 7), (5, 7), (3, 8), (6, 8),
            (8, 3), (8, 2),
        }

        # Battery and package settings.
        self.battery_capacity = 100.0
        self.recharge_amount = 25.0
        self.safe_temp_min = 2.0
        self.safe_temp_max = 8.0
        self.max_temp = 10.0

        self.action_space = spaces.Discrete(7)

        # 18 observations: the 15 features of the design, where the single
        # "obstacle in next direction" value is replaced by four values
        # (north, south, east, west) so the agent knows every direction.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(18,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_distance(self, a, b):
        """Manhattan distance between two cells."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_blocked(self, cell):
        """True when the cell is outside the map or contains an obstacle."""
        x, y = cell
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return True
        return cell in self.obstacles

    def current_target(self):
        """Where the drone should go next, based on the mission stage."""
        if not self.package_collected:
            return self.bank
        if not self.delivered:
            return self.hospital
        return self.bank

    def mission_stage(self):
        """0 = collect blood, 0.5 = carrying blood, 1 = returning to base."""
        if not self.package_collected:
            return 0.0
        if not self.delivered:
            return 0.5
        return 1.0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self):

        x, y = self.drone_position
        span = self.grid_size - 1
        max_distance = 2.0 * span
        remaining = max(self.max_steps - self.steps, 1)

        to_hospital = self.get_distance(self.drone_position, self.hospital)
        to_bank = self.get_distance(self.drone_position, self.bank)
        to_charger = self.get_distance(self.drone_position, self.charger)

        # How tight the deadline is for reaching the current target.
        urgency = min(
            self.get_distance(self.drone_position, self.current_target())
            / remaining,
            1.0,
        )

        temperature = (self.package_temperature - 0.0) / self.max_temp

        obstacle_ahead = [
            float(self.is_blocked((x + dx, y + dy)))
            for dx, dy in MOVES.values()
        ]

        observation = [
            x / span,
            y / span,
            self.battery / self.battery_capacity,
            float(self.package_collected),
            to_hospital / max_distance,
            to_bank / max_distance,
            to_charger / max_distance,
            urgency,
            remaining / self.max_steps,
            self.wind / 10.0,
            self.rain / 10.0,
            float(np.clip(temperature, 0.0, 1.0)),
            max(self.package_integrity, 0.0) / 100.0,
            *obstacle_ahead,
            self.mission_stage(),
        ]

        return np.array(observation, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.drone_position = self.bank
        self.battery = self.battery_capacity

        self.package_collected = False
        self.delivered = False
        self.mission_complete = False

        self.package_temperature = 4.0
        self.package_integrity = 100.0

        self.steps = 0
        self.obstacle_hits = 0
        self.total_reward = 0.0

        # Weather: a base level plus occasional storms.
        self.wind = float(self.np_random.uniform(0.5, 3.0))
        self.rain = float(self.np_random.uniform(0.0, 2.0))
        self.storm_steps_left = 0

        self.last_event = "start"

        return self.get_observation(), {}

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def update_weather(self):
        """Random weather with occasional storms (strong wind and rain)."""

        if self.storm_steps_left == 0 and self.np_random.random() < 0.03:
            self.storm_steps_left = int(self.np_random.integers(8, 16))

        if self.storm_steps_left > 0:
            self.storm_steps_left -= 1
            wind_target, rain_target = 8.5, 7.0
        else:
            wind_target, rain_target = 2.0, 1.0

        self.wind = float(np.clip(
            self.wind + 0.35 * (wind_target - self.wind)
            + self.np_random.uniform(-0.6, 0.6), 0.0, 10.0))
        self.rain = float(np.clip(
            self.rain + 0.35 * (rain_target - self.rain)
            + self.np_random.uniform(-0.4, 0.4), 0.0, 10.0))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):

        action = int(action)
        self.steps += 1

        reward = REWARD_STEP
        event = "move"
        terminated = False

        old_target_distance = self.get_distance(
            self.drone_position, self.current_target()
        )

        at_charger = self.drone_position == self.charger
        battery_cost = 0.0

        if action in MOVES:
            reward_delta, event, battery_cost = self.move_drone(action)
            reward += reward_delta

        elif action == 4:
            reward_delta, event, battery_cost = self.handle_package()
            reward += reward_delta

        elif action == 5:
            reward_delta, event = self.handle_recharge(at_charger)
            reward += reward_delta
            battery_cost = 0.2

        else:
            # Waiting: costs very little battery and lets storms pass.
            event = "wait"
            battery_cost = 0.2

        self.battery = max(self.battery - battery_cost, 0.0)

        # Reward flying toward the current target.
        if action in MOVES and event == "move":
            new_target_distance = self.get_distance(
                self.drone_position, self.current_target()
            )
            reward += REWARD_PROGRESS * (
                old_target_distance - new_target_distance
            )

        self.update_weather()
        self.update_package()

        # ---------------- Termination checks ----------------

        if (
            self.delivered
            and not self.mission_complete
            and self.drone_position == self.bank
        ):
            self.mission_complete = True
            reward += REWARD_RETURN
            event = "returned"
            terminated = True

        if self.obstacle_hits >= MAX_OBSTACLE_HITS or self.package_integrity <= 0:
            reward += PENALTY_CRASH
            event = "crash"
            terminated = True

        if self.package_temperature >= self.max_temp and not self.delivered:
            reward += PENALTY_UNSAFE_TEMP
            event = "unsafe_temp"
            terminated = True

        if self.battery <= 0 and self.drone_position != self.charger:
            if not self.mission_complete:
                reward += PENALTY_BATTERY
                event = "battery_dead"
            terminated = True

        truncated = self.steps >= self.max_steps and not terminated

        if truncated:
            reward += PENALTY_DEADLINE
            event = "deadline"

        self.total_reward += reward
        self.last_event = event

        info = {
            "event": event,
            "delivered": self.delivered,
            "mission_complete": self.mission_complete,
            "battery": self.battery,
            "package_temperature": self.package_temperature,
            "package_integrity": self.package_integrity,
            "wind": self.wind,
            "rain": self.rain,
            "steps": self.steps,
            "obstacle_hits": self.obstacle_hits,
            "total_reward": self.total_reward,
        }

        return self.get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Parts of a step
    # ------------------------------------------------------------------

    def move_drone(self, action):
        """Move one cell. Returns (reward, event, battery cost)."""

        # Strong wind can push the drone in a random direction.
        gust_chance = max(self.wind - 6.0, 0.0) * 0.1
        if self.np_random.random() < gust_chance:
            action = int(self.np_random.integers(0, 4))

        dx, dy = MOVES[action]
        new_position = (
            self.drone_position[0] + dx,
            self.drone_position[1] + dy,
        )

        # Flying uses more energy in wind and rain.
        cost = 1.8 + self.wind * 0.12 + self.rain * 0.08

        if self.battery <= 0:
            return PENALTY_INVALID, "invalid", 0.0

        x, y = new_position
        outside = (
            x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size
        )

        if outside:
            return PENALTY_INVALID, "invalid", 0.0

        if new_position in self.obstacles:
            self.obstacle_hits += 1
            self.package_integrity -= 15.0
            return PENALTY_OBSTACLE, "obstacle", cost + 2.0

        self.drone_position = new_position
        return 0.0, "move", cost

    def handle_package(self):
        """Collect blood at the bank or deliver it at the hospital."""

        if (
            self.drone_position == self.bank
            and not self.package_collected
        ):
            self.package_collected = True
            return REWARD_COLLECT, "collected", 0.5

        if (
            self.drone_position == self.hospital
            and self.package_collected
            and not self.delivered
        ):
            self.delivered = True
            return REWARD_DELIVER, "delivered", 0.5

        return PENALTY_INVALID, "invalid", 0.0

    def handle_recharge(self, at_charger):
        """Recharge only at the charging station."""

        if not at_charger or self.battery >= self.battery_capacity - 5.0:
            return PENALTY_INVALID, "invalid"

        self.battery = min(
            self.battery + self.recharge_amount, self.battery_capacity
        )
        return REWARD_RECHARGE, "recharged"

    def update_package(self):
        """Blood warms up while carried; unsafe temperatures damage it."""

        if self.package_collected and not self.delivered:
            self.package_temperature += (
                0.04 + self.rain * 0.01 + self.wind * 0.006
            )
        else:
            # Before collection the package is in the cold box.
            self.package_temperature = max(
                self.package_temperature - 0.2, 4.0
            )

        if (
            self.package_collected
            and not self.delivered
            and not (
                self.safe_temp_min <= self.package_temperature
                <= self.safe_temp_max
            )
        ):
            self.package_integrity -= 2.0

    # ------------------------------------------------------------------
    # Text rendering
    # ------------------------------------------------------------------

    def render(self):
        """Return a text picture of the map."""

        rows = []
        for y in range(self.grid_size):
            row = ""
            for x in range(self.grid_size):
                cell = (x, y)
                if cell == self.drone_position:
                    row += "D "
                elif cell == self.bank:
                    row += "B "
                elif cell == self.hospital:
                    row += "H "
                elif cell == self.charger:
                    row += "C "
                elif cell in self.obstacles:
                    row += "# "
                else:
                    row += ". "
            rows.append(row)

        text = "\n".join(rows)
        print(text)
        return text
