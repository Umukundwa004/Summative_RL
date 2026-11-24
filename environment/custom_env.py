"""
Blood Delivery Drone Environment.

The drone starts at the blood bank, collects blood, delivers it to a hospital,
recharges at the charging station and flies back to the bank. One episode is
one complete mission. Towers, buildings, the hospital, the weather and the
battery are different in every episode.

Actions (9):
    0 north        4 collect / deliver
    1 south        5 recharge
    2 east         6 wait
    3 west         7 climb
                   8 descend
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_NAMES = [
    "Move north", "Move south", "Move east", "Move west",
    "Collect / Deliver", "Recharge", "Wait", "Climb", "Descend",
]

# (dx, dy) for the four moves. North is y - 1.
MOVES = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}

# ---- Altitude ----
MAX_ALTITUDE = 3
CLIMB_COST = 2.0
DESCEND_COST = 0.3
ALTITUDE_COST = 0.3          # extra battery per move, for each level of altitude

# ---- Towers and buildings ----
# An obstacle of height h blocks the drone at altitudes 0 to h.
# So a drone must be higher than an obstacle to fly over it.
LOW_TOWER, MEDIUM_TOWER, HIGH_TOWER, BUILDING = 1, 2, 3, 4
TOWER_HEIGHTS = [LOW_TOWER, MEDIUM_TOWER, HIGH_TOWER]

# ---- Battery and mission ----
ROUNDS = 1                   # one episode = collect, deliver, recharge, return
LOW_BATTERY = 0.35           # below this the drone goes to the charger
MOVE_COST = 1.0              # battery for one move on a calm day
ENERGY_SAFETY = 1.25         # safety margin on the battery estimate

# ---- Weather (rain on a 0-10 scale) ----
RAIN_MODERATE = 3.0
RAIN_HEAVY = 6.0
RAIN_SEVERE = 8.0
LIGHT, MODERATE, HEAVY, SEVERE = 0, 1, 2, 3

# ---- Rewards and penalties ----
REWARD_COLLECT = 15.0
REWARD_DELIVER = 100.0
REWARD_ROUND = 15.0          # back at the bank after the recharge
REWARD_SAFE_WAIT = 0.5       # waiting landed on a shelter in a severe storm
REWARD_STEP = -0.1           # small cost for every step
POTENTIAL_SCALE = 1.0        # shaping: reward for getting closer to the target
SHAPING_GAMMA = 0.99
PENALTY_INVALID = -3.0
PENALTY_WAIT = -0.5
PENALTY_OBSTACLE = -20.0
PENALTY_DEADLINE = -40.0
PENALTY_UNSAFE_TEMP = -60.0
PENALTY_CRASH = -100.0
PENALTY_BATTERY = -100.0
PENALTY_HEAVY_RAIN = -2.0    # per step in heavy rain, outside a shelter
PENALTY_SEVERE_STORM = -10.0 # per step in a severe storm, not landed on a shelter
MAX_OBSTACLE_HITS = 3

# ---- Action mask (the last 9 observation values) ----
MASK_START = 32
MASK_BLOCK = 1e9             # score given to an impossible action


def mask_scores(scores, observation):
    """Give impossible actions a very low score, so they are never chosen.

    Works for numpy arrays and torch tensors."""
    mask = observation[..., MASK_START:MASK_START + 9]
    return scores * mask + (mask - 1.0) * MASK_BLOCK


def rain_level(rain):
    """Rain value (0-10) to a level: 0 light, 1 moderate, 2 heavy, 3 severe."""
    if rain >= RAIN_SEVERE:
        return SEVERE
    if rain >= RAIN_HEAVY:
        return HEAVY
    if rain >= RAIN_MODERATE:
        return MODERATE
    return LIGHT


class BloodDroneEnv(gym.Env):
    """Blood delivery drone on a 10 x 10 map."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size=10, max_steps=300, practice=0.0, practice_episodes=2000):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Practice: a chance that an episode starts near the hospital with the blood
        # on board. The chance fades to 0 over the first practice_episodes episodes.
        self.practice = practice
        self.practice_episodes = practice_episodes
        self.episodes_started = 0

        # Fixed places.
        self.bank = (0, 0)
        self.charger = (5, 5)
        self.hospital_options = [(9, 9), (9, 0), (0, 9), (9, 5)]

        # Battery and package.
        self.battery_capacity = 100.0
        self.recharge_amount = 25.0
        self.safe_temp_min = 2.0
        self.safe_temp_max = 8.0
        self.max_temp = 10.0

        self.action_space = spaces.Discrete(9)
        # 43 values, all from 0 to 1. See get_observation for the list.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(43,), dtype=np.float32)

        self.reset()

    # ------------------------------------------------------------------
    # Map and distances
    # ------------------------------------------------------------------

    def get_distance(self, a, b):
        """Straight-line (Manhattan) distance between two cells."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def in_map(self, cell):
        """True if the cell is inside the map."""
        x, y = cell
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_blocked(self, cell, altitude=0):
        """True if a drone at this altitude cannot be in this cell."""
        if not self.in_map(cell):
            return True
        height = self.obstacles.get(cell)
        return height is not None and altitude <= height

    def height_of(self, cell):
        """Obstacle height in a cell, from 0 to 1. Outside the map counts as a building."""
        if not self.in_map(cell):
            return 1.0
        return self.obstacles.get(cell, 0) / BUILDING

    def flight_distances(self, start):
        """Steps from start to every cell that can be reached by flying at the top altitude."""
        queue = deque([start])
        dist = {start: 0}
        while queue:
            cell = queue.popleft()
            for dx, dy in MOVES.values():
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt not in dist and not self.is_blocked(nxt, MAX_ALTITUDE):
                    dist[nxt] = dist[cell] + 1
                    queue.append(nxt)
        return dist

    def path_distance(self):
        """Walking distance (around towers) from the drone to its current target."""
        return self.dist_maps[self.current_target()].get(self.drone_position, self.grid_size * 4)

    def potential(self):
        """Shaping potential: minus the distance to the target.

        The target depends only on the mission stage, so loops add up to zero reward."""
        return -POTENTIAL_SCALE * self.path_distance()

    # ------------------------------------------------------------------
    # Mission rules
    # ------------------------------------------------------------------

    def energy_for_steps(self, steps):
        """Battery for a number of cells, at a middle altitude, with a safety margin."""
        return steps * (MOVE_COST + ALTITUDE_COST * 2) * ENERGY_SAFETY

    def energy_needed(self):
        """Battery for the bank, hospital and back to the bank. Needed before collecting."""
        steps = self.flight_dist.get(self.hospital, self.grid_size * 2)
        return self.energy_for_steps(2 * steps)

    def ready_for_round(self):
        """Charged, and enough battery for the round plus the trip from the charger."""
        trip_back = self.dist_maps[self.bank].get(self.charger, self.grid_size * 4)
        needed = self.energy_needed() + self.energy_for_steps(trip_back)
        return self.charged_since_return and self.battery >= needed

    def current_target(self):
        """Where the drone should go next: bank, hospital, charger, then bank again."""
        if self.battery < LOW_BATTERY * self.battery_capacity:
            return self.charger
        if not self.package_collected:
            if self.charged_since_return and self.battery >= self.energy_needed():
                return self.bank
            return self.charger
        if not self.round_delivered:
            return self.hospital
        if not self.recharged_after_delivery:
            return self.charger
        return self.bank

    def mission_stage(self):
        """0 = collect blood, 0.5 = carrying blood, 1 = after delivery."""
        if not self.package_collected:
            return 0.0
        if not self.round_delivered:
            return 0.5
        return 1.0

    def action_mask(self):
        """Nine numbers: 1.0 if the action is possible now, 0.0 if not."""
        mask = np.zeros(9, dtype=np.float32)
        x, y = self.drone_position
        alt = self.altitude
        on_ground = alt == 0

        for action, (dx, dy) in MOVES.items():
            mask[action] = 0.0 if self.is_blocked((x + dx, y + dy), alt) else 1.0

        can_collect = (
            self.drone_position == self.bank
            and not self.package_collected
            and self.rounds_done < ROUNDS
            and self.charged_since_return
            and self.battery >= self.energy_needed()
            and on_ground
        )
        can_deliver = (
            self.drone_position == self.hospital
            and self.package_collected
            and not self.round_delivered
            and on_ground
        )
        can_recharge = (
            self.drone_position == self.charger
            and on_ground
            and (not self.round_delivered or not self.recharged_after_delivery)
            and self.battery < self.battery_capacity
        )
        mask[4] = 1.0 if (can_collect or can_deliver) else 0.0
        mask[5] = 1.0 if can_recharge else 0.0
        mask[6] = 1.0
        mask[7] = 1.0 if alt < MAX_ALTITUDE else 0.0
        mask[8] = 1.0 if alt > 0 and not self.is_blocked(self.drone_position, alt - 1) else 0.0
        return mask

    # ------------------------------------------------------------------
    # Observation: what the drone can see, all from 0 to 1
    # ------------------------------------------------------------------

    def get_observation(self):
        x, y = self.drone_position
        span = self.grid_size - 1
        max_distance = 2.0 * span
        remaining = max(self.max_steps - self.steps, 1)
        target = self.current_target()
        around = [(x + dx, y + dy) for dx, dy in MOVES.values()]

        observation = [
            x / span,                                     # 1 drone x
            y / span,                                     # 2 drone y
            self.battery / self.battery_capacity,         # 3 battery
            float(self.package_collected),                # 4 blood on board
            self.get_distance(self.drone_position, self.hospital) / max_distance,  # 5
            self.get_distance(self.drone_position, self.bank) / max_distance,      # 6
            self.get_distance(self.drone_position, self.charger) / max_distance,   # 7
            min(self.get_distance(self.drone_position, target) / remaining, 1.0),  # 8 urgency
            remaining / self.max_steps,                   # 9 time left
            self.wind / 10.0,                             # 10 wind
            self.rain / 10.0,                             # 11 rain
            self.package_temperature / self.max_temp,     # 12 temperature
            max(self.package_integrity, 0.0) / 100.0,     # 13 integrity
            *[float(self.is_blocked(c, self.altitude)) for c in around],   # 14-17 blocked
            self.mission_stage(),                         # 18 mission stage
            *[self.weather_at(c)[1] / 10.0 if self.in_map(c)
              else 0.0 for c in around],                  # 19-22 rain around
            self.altitude / MAX_ALTITUDE,                 # 23 altitude
            rain_level(self.rain) / SEVERE,               # 24 weather level
            float(self.drone_position in self.shelters),  # 25 on a shelter
            self.rounds_done / ROUNDS,                    # 26 rounds finished
            float(self.ready_for_round()),                # 27 ready for a round
            *[self.height_of(c) for c in around],         # 28-31 tower heights
            self.path_distance() / 40.0,                  # 32 walking distance to target
            *self.action_mask(),                          # 33-41 possible actions
            target[0] / span,                             # 42 target x
            target[1] / span,                             # 43 target y
        ]
        return np.clip(np.array(observation, dtype=np.float32), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Reset and random map
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.build_map()
        self.hospital = self.hospital_options[int(self.np_random.integers(len(self.hospital_options)))]
        self.flight_dist = self.flight_distances(self.bank)

        self.drone_position = self.bank
        self.altitude = 0
        self.battery = float(self.np_random.uniform(40.0, self.battery_capacity))

        self.package_collected = False
        self.round_delivered = False
        self.recharged_after_delivery = False
        self.delivered = False
        self.mission_complete = False
        self.rounds_done = 0
        self.charged_since_return = True
        self.package_temperature = 4.0
        self.package_integrity = 100.0

        self.steps = 0
        self.obstacle_hits = 0
        self.total_reward = 0.0

        self.base_wind = float(self.np_random.uniform(0.5, 3.0))
        self.storms = []
        self.spawn_storm()
        self.wind, self.rain = self.weather_at(self.drone_position)
        self.last_event = "start"

        # Practice start: near the hospital, with the blood on board, fully charged.
        chance = self.practice * max(0.0, 1.0 - self.episodes_started / self.practice_episodes)
        self.episodes_started += 1
        if chance > 0 and self.np_random.random() < chance:
            near = [c for c, d in self.dist_maps[self.hospital].items()
                    if 1 <= d <= 12 and c not in self.obstacles]
            if near:
                self.drone_position = near[int(self.np_random.integers(len(near)))]
                self.battery = self.battery_capacity
                self.package_collected = True
                self.wind, self.rain = self.weather_at(self.drone_position)

        return self.get_observation(), {}

    def build_map(self):
        """Place towers, buildings and two landing pads at random.

        Tries again until every important place can be reached by flying."""
        protected = {self.bank, self.charger, *self.hospital_options}
        free_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                      if (x, y) not in protected]

        for _ in range(500):
            towers = int(self.np_random.integers(6, 10))
            buildings = int(self.np_random.integers(3, 6))
            picked = self.np_random.choice(len(free_cells), size=towers + buildings + 2, replace=False)
            cells = [free_cells[i] for i in picked]

            self.obstacles = {}
            for cell in cells[:towers]:
                self.obstacles[cell] = int(self.np_random.choice(TOWER_HEIGHTS))
            for cell in cells[towers:towers + buildings]:
                self.obstacles[cell] = BUILDING
            self.shelters = {self.charger, *cells[towers + buildings:]}

            self.flight_dist = self.flight_distances(self.bank)
            important = [self.charger, *self.hospital_options, *self.shelters]
            if all(cell in self.flight_dist for cell in important):
                self.dist_maps = {cell: self.flight_distances(cell)
                                  for cell in [self.bank, self.charger, *self.hospital_options]}
                return
        raise RuntimeError("Could not build a map where every place can be reached.")

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def spawn_storm(self):
        """A storm patch, about 5 cells wide, somewhere on the map."""
        self.storms.append({
            "x": int(self.np_random.integers(0, self.grid_size)),
            "y": int(self.np_random.integers(0, self.grid_size)),
            "radius": 2.5,
            "age": 0,
            "life": int(self.np_random.integers(20, 40)),
        })

    def weather_at(self, cell):
        """(wind, rain) in one cell: calm weather plus any storms."""
        wind, rain = self.base_wind, 0.0
        for storm in self.storms:
            distance = ((cell[0] - storm["x"]) ** 2 + (cell[1] - storm["y"]) ** 2) ** 0.5
            if distance > storm["radius"]:
                continue
            # Strongest in the middle, fading at the edge and over the storm's life.
            strength = 1.0 - distance / (storm["radius"] + 0.5)
            strength *= min(storm["age"] / 4.0, (storm["life"] - storm["age"]) / 4.0, 1.0)
            rain += 10.0 * max(strength, 0.0)
            wind += 6.0 * max(strength, 0.0)
        return min(wind, 10.0), min(rain, 10.0)

    def update_weather(self):
        """Storms move, age and disappear. New ones appear now and then."""
        for storm in self.storms:
            storm["age"] += 1
            if storm["age"] % 3 == 0:
                dx, dy = list(MOVES.values())[int(self.np_random.integers(0, 4))]
                storm["x"] = int(np.clip(storm["x"] + dx, 0, self.grid_size - 1))
                storm["y"] = int(np.clip(storm["y"] + dy, 0, self.grid_size - 1))
        self.storms = [s for s in self.storms if s["age"] < s["life"]]

        if len(self.storms) < 2 and self.np_random.random() < 0.05:
            self.spawn_storm()
        self.wind, self.rain = self.weather_at(self.drone_position)

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def step(self, action):
        action = int(action)
        self.steps += 1
        phi_before = self.potential()

        # Each handler returns (reward change, event name, battery used).
        if action in MOVES:
            change, event, cost = self.move_drone(action)
        elif action == 4:
            change, event, cost = self.handle_package()
        elif action == 5:
            change, event, cost = self.handle_recharge()
        elif action == 6:
            change, event, cost = self.handle_wait()
        elif action == 7:
            change, event, cost = self.handle_climb()
        else:
            change, event, cost = self.handle_descend()

        reward = REWARD_STEP + change
        self.battery = max(self.battery - cost, 0.0)
        self.update_weather()
        self.update_package()

        # Weather: penalties for flying in rain or storms.
        level = rain_level(self.rain)
        sheltered = self.drone_position in self.shelters
        if level == HEAVY and not sheltered:
            reward += PENALTY_HEAVY_RAIN
        if level == SEVERE and not (sheltered and self.altitude == 0):
            reward += PENALTY_SEVERE_STORM

        terminated = False
        # Back at the bank after delivering and recharging: the mission is complete.
        if (self.round_delivered and self.recharged_after_delivery
                and self.drone_position == self.bank and self.altitude == 0):
            self.rounds_done += 1
            reward += REWARD_ROUND
            event = "returned"
            self.mission_complete = True
            terminated = True
            self.package_collected = False
            self.round_delivered = False
            self.recharged_after_delivery = False
            self.charged_since_return = False

        # Shaping after the stage changes above (stages are part of the state).
        reward += SHAPING_GAMMA * self.potential() - phi_before

        # End of the episode: crash, bad blood, empty battery, or the deadline.
        if self.obstacle_hits >= MAX_OBSTACLE_HITS or self.package_integrity <= 0:
            reward += PENALTY_CRASH
            event, terminated = "crash", True

        carrying = self.package_collected and not self.round_delivered
        if self.package_temperature >= self.max_temp and carrying:
            reward += PENALTY_UNSAFE_TEMP
            event, terminated = "unsafe_temp", True

        if self.battery <= 0 and self.drone_position != self.charger and not self.mission_complete:
            reward += PENALTY_BATTERY
            event, terminated = "battery_dead", True

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
            "rounds_done": self.rounds_done,
            "battery": self.battery,
            "altitude": self.altitude,
            "package_temperature": self.package_temperature,
            "package_integrity": self.package_integrity,
            "wind": self.wind,
            "rain": self.rain,
            "weather_level": rain_level(self.rain),
            "storms": [dict(s) for s in self.storms],
            "steps": self.steps,
            "obstacle_hits": self.obstacle_hits,
            "total_reward": self.total_reward,
        }
        return self.get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # The actions. Each returns (reward change, event name, battery used).
    # ------------------------------------------------------------------

    def move_drone(self, action):
        # Strong wind can push the drone in a random direction.
        gust_chance = max(self.wind - 6.0, 0.0) * 0.1
        if self.np_random.random() < gust_chance:
            action = int(self.np_random.integers(0, 4))

        dx, dy = MOVES[action]
        new_position = (self.drone_position[0] + dx, self.drone_position[1] + dy)
        cost = MOVE_COST + ALTITUDE_COST * self.altitude + self.wind * 0.25 + self.rain * 0.3
        if rain_level(self.rain) == MODERATE:
            cost += 0.5                     # moderate rain: fly carefully

        if self.battery <= 0:
            return PENALTY_INVALID, "invalid", 0.0
        if not self.in_map(new_position):
            return PENALTY_INVALID, "invalid", 0.0
        if self.is_blocked(new_position, self.altitude):
            self.obstacle_hits += 1
            self.package_integrity -= 15.0
            return PENALTY_OBSTACLE, "obstacle", cost + 2.0

        self.drone_position = new_position
        return 0.0, "move", cost

    def handle_package(self):
        """Collect the blood at the bank, or deliver it at the hospital."""
        if self.altitude != 0:
            return PENALTY_INVALID, "invalid", 0.0

        if self.drone_position == self.bank and not self.package_collected and self.rounds_done < ROUNDS:
            if not self.charged_since_return:
                return PENALTY_INVALID, "must_charge_first", 0.0
            if self.battery < self.energy_needed():
                return PENALTY_INVALID, "not_enough_battery", 0.0
            self.package_collected = True
            self.package_integrity = 100.0
            return REWARD_COLLECT, "collected", 0.5

        if self.drone_position == self.hospital and self.package_collected and not self.round_delivered:
            self.round_delivered = True
            self.recharged_after_delivery = False
            self.delivered = True
            return REWARD_DELIVER, "delivered", 0.5

        return PENALTY_INVALID, "invalid", 0.0

    def handle_recharge(self):
        """Recharge at the charger on the ground. Recharging earns no reward."""
        if self.drone_position != self.charger or self.altitude != 0:
            return PENALTY_INVALID, "invalid", 0.0
        self.battery = min(self.battery + self.recharge_amount, self.battery_capacity)
        self.charged_since_return = True
        if self.round_delivered:
            self.recharged_after_delivery = True
        return 0.0, "recharged", 0.0

    def handle_wait(self):
        reward = PENALTY_WAIT
        if rain_level(self.rain) == SEVERE and self.drone_position in self.shelters and self.altitude == 0:
            reward += REWARD_SAFE_WAIT
        return reward, "wait", 0.4 + 0.3 * self.altitude

    def handle_climb(self):
        if self.altitude >= MAX_ALTITUDE:
            return PENALTY_INVALID, "invalid", 0.0
        self.altitude += 1
        return 0.0, "climbed", CLIMB_COST

    def handle_descend(self):
        if self.altitude == 0:
            return PENALTY_INVALID, "invalid", 0.0
        if self.is_blocked(self.drone_position, self.altitude - 1):
            return PENALTY_INVALID, "invalid", 0.0   # would land inside a tower
        self.altitude -= 1
        return 0.0, "descended", DESCEND_COST

    def update_package(self):
        """Blood warms up while carried, faster in bad weather. Otherwise it cools."""
        carrying = self.package_collected and not self.round_delivered
        if carrying:
            self.package_temperature += 0.04 + self.rain * 0.01 + self.wind * 0.006
        else:
            self.package_temperature = max(self.package_temperature - 0.2, 4.0)

        if carrying and not (self.safe_temp_min <= self.package_temperature <= self.safe_temp_max):
            self.package_integrity -= 2.0

    # ------------------------------------------------------------------
    # Text picture of the map
    # ------------------------------------------------------------------

    def render(self):
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
                elif cell in self.shelters:
                    row += "S "
                elif cell in self.obstacles:
                    height = self.obstacles[cell]
                    row += "# " if height >= BUILDING else f"{height} "
                else:
                    row += ". "
            rows.append(row)
        text = "\n".join(rows)
        print(text)
        return text
