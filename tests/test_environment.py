"""Tests for the blood delivery drone environment (version 2)."""

import numpy as np
import pytest

from environment.custom_env import (
    BUILDING,
    CLIMB_COST,
    HEAVY,
    LIGHT,
    MODERATE,
    MOVES,
    SEVERE,
    BloodDroneEnv,
    rain_level,
)

NORTH, SOUTH, EAST, WEST, COLLECT, RECHARGE, WAIT, CLIMB, DESCEND = range(9)


@pytest.fixture
def env():
    """An empty, calm map with a full battery, so each test controls the situation."""
    env = BloodDroneEnv()
    env.reset(seed=0)
    env.storms = []
    env.base_wind = 0.0
    env.wind = 0.0
    env.rain = 0.0
    env.obstacles = {}
    env.shelters = {env.charger}
    env.flight_dist = env.flight_distances(env.bank)
    env.battery = 100.0
    env.charged_since_return = True
    return env


def test_spaces():
    env = BloodDroneEnv()
    assert env.action_space.n == 9
    assert env.observation_space.shape == (43,)


def test_reset_returns_valid_observation():
    env = BloodDroneEnv()
    observation, _ = env.reset(seed=1)
    assert observation.shape == (43,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert env.drone_position == env.bank
    assert env.altitude == 0
    assert 40.0 <= env.battery <= 100.0
    assert not env.package_collected


def test_every_important_place_can_be_reached():
    # Check many random maps: the bank, charger, hospitals and landing pads
    # must always be reachable by flying.
    for seed in range(30):
        env = BloodDroneEnv()
        env.reset(seed=seed)
        reachable = env.flight_distances(env.bank)
        for cell in [env.charger, *env.hospital_options, *env.shelters]:
            assert cell in reachable, f"seed {seed}: {cell} cannot be reached"


def test_random_maps_have_different_tower_heights():
    heights = set()
    for seed in range(10):
        env = BloodDroneEnv()
        env.reset(seed=seed)
        heights.update(env.obstacles.values())
    assert {1, 2, 3, BUILDING} <= heights


def test_same_seed_gives_same_map():
    a, b = BloodDroneEnv(), BloodDroneEnv()
    a.reset(seed=11)
    b.reset(seed=11)
    assert a.obstacles == b.obstacles
    assert a.hospital == b.hospital
    assert a.shelters == b.shelters


def test_different_seeds_give_different_maps():
    a, b = BloodDroneEnv(), BloodDroneEnv()
    a.reset(seed=11)
    b.reset(seed=12)
    assert a.obstacles != b.obstacles


def test_collect_at_bank_gives_reward(env):
    _, reward, terminated, _, info = env.step(COLLECT)
    assert env.package_collected
    assert info["event"] == "collected"
    assert not terminated


def test_collect_away_from_bank_is_invalid(env):
    env.step(EAST)
    _, reward, _, _, info = env.step(COLLECT)
    assert info["event"] == "invalid"
    assert reward < 0


def test_leaving_the_map_is_invalid(env):
    _, reward, _, _, info = env.step(NORTH)  # the bank is in the top-left corner
    assert info["event"] == "invalid"
    assert env.drone_position == env.bank
    assert reward < 0


def test_not_enough_battery_blocks_collecting(env):
    env.battery = 10.0
    _, reward, _, _, info = env.step(COLLECT)
    assert info["event"] == "not_enough_battery"
    assert not env.package_collected
    assert reward < 0


def test_must_land_before_loading_or_unloading(env):
    env.package_collected = True
    env.drone_position = env.hospital
    env.altitude = 1
    _, _, _, _, info = env.step(COLLECT)
    assert info["event"] == "invalid"
    assert not env.delivered


# ---------------------------------------------------------------- altitude


def test_ground_drone_is_blocked_by_a_low_tower(env):
    env.obstacles = {(3, 0): 1}          # a low tower
    env.drone_position = (2, 0)
    _, reward, _, _, info = env.step(EAST)
    assert info["event"] == "obstacle"
    assert env.drone_position == (2, 0)
    assert env.obstacle_hits == 1
    assert reward <= -20


def test_climbing_lets_the_drone_fly_over_a_low_tower(env):
    env.obstacles = {(3, 0): 1}
    env.drone_position = (2, 0)
    env.step(CLIMB)
    env.step(CLIMB)                       # altitude 2 is higher than the tower
    _, _, _, _, info = env.step(EAST)
    assert info["event"] == "move"
    assert env.drone_position == (3, 0)


def test_high_tower_cannot_be_flown_over(env):
    env.obstacles = {(3, 0): 3}          # high tower: even altitude 3 is not enough
    env.drone_position = (2, 0)
    env.altitude = 3
    _, _, _, _, info = env.step(EAST)
    assert info["event"] == "obstacle"
    assert env.drone_position == (2, 0)


def test_building_always_blocks_the_drone(env):
    env.obstacles = {(3, 0): BUILDING}
    env.drone_position = (2, 0)
    env.altitude = 3
    _, _, _, _, info = env.step(EAST)
    assert info["event"] == "obstacle"


def test_climb_has_a_limit_and_costs_battery(env):
    for _ in range(3):
        _, _, _, _, info = env.step(CLIMB)
        assert info["event"] == "climbed"
    assert env.altitude == 3
    _, _, _, _, info = env.step(CLIMB)
    assert info["event"] == "invalid"
    assert env.battery == pytest.approx(100.0 - 3 * CLIMB_COST)


def test_cannot_descend_into_a_tower(env):
    env.obstacles = {(3, 0): 2}          # medium tower
    env.drone_position = (3, 0)
    env.altitude = 3
    _, _, _, _, info = env.step(DESCEND)
    assert info["event"] == "invalid"    # altitude 2 would be inside the tower
    assert env.altitude == 3


def test_descend_at_ground_is_invalid(env):
    _, _, _, _, info = env.step(DESCEND)
    assert info["event"] == "invalid"


def test_higher_flight_uses_more_battery(env):
    env.altitude = 0
    env.step(EAST)
    low_cost = 100.0 - env.battery

    env.battery = 100.0
    env.drone_position = env.bank
    env.altitude = 3
    env.step(EAST)
    high_cost = 100.0 - env.battery

    assert high_cost > low_cost


def test_three_collisions_crash_the_drone(env):
    env.obstacles = {(3, 0): 1}
    env.drone_position = (2, 0)
    terminated = False
    for _ in range(3):
        _, reward, terminated, _, info = env.step(EAST)
    assert terminated
    assert info["event"] == "crash"
    assert reward <= -100


def test_moving_toward_target_beats_moving_away(env):
    env.step(COLLECT)                    # target becomes the hospital
    _, closer, _, _, _ = env.step(EAST)
    env.drone_position = env.bank
    _, farther, _, _, _ = env.step(WEST)
    assert closer > farther


# ---------------------------------------------------------------- battery


def test_recharge_only_works_at_the_charger(env):
    env.battery = 40.0
    _, _, _, _, info = env.step(RECHARGE)
    assert info["event"] == "invalid"
    assert env.battery == 40.0           # a refused recharge costs no battery

    env.drone_position = env.charger
    _, _, _, _, info = env.step(RECHARGE)
    assert info["event"] == "recharged"
    assert env.battery == pytest.approx(65.0)


def test_recharge_must_be_on_the_ground(env):
    env.battery = 40.0
    env.drone_position = env.charger
    env.altitude = 1
    _, _, _, _, info = env.step(RECHARGE)
    assert info["event"] == "invalid"


def test_recharging_gives_no_reward_so_it_cannot_be_farmed(env):
    # Recharging again and again must not earn reward overall.
    env.drone_position = env.charger
    env.battery = 10.0
    total = 0.0
    for _ in range(10):
        _, reward, _, _, _ = env.step(RECHARGE)
        total += reward
    assert total <= 0.0


def test_going_back_and_forth_does_not_earn_progress_again(env):
    # Potential-based shaping: a loop that returns to the same cell adds up to zero.
    env.step(COLLECT)                    # carrying blood: target is the hospital
    loop = [EAST] * 3 + [WEST] * 3
    total = sum(env.step(action)[1] for action in loop)
    assert abs(total) < 0.5
    # Repeating the same loop does not earn anything either.
    again = sum(env.step(action)[1] for action in loop)
    assert abs(again) < 0.5


def test_wait_costs_little_battery(env):
    before = env.battery
    env.step(WAIT)
    assert 0 < before - env.battery < 1.0


def test_low_battery_sends_the_drone_to_the_charger(env):
    env.package_collected = True
    assert env.current_target() == env.hospital
    env.battery = 20.0
    assert env.current_target() == env.charger


def test_drone_charges_first_when_battery_is_too_low_for_a_round(env):
    # The far hospital (9, 9) makes a round plus the trip back need about 92 battery.
    env.hospital = (9, 9)
    env.battery = 60.0                   # above the low-battery level, but not enough for a round
    assert env.current_target() == env.charger
    env.battery = 100.0
    assert env.current_target() == env.bank


# ---------------------------------------------------------------- rounds


def test_one_episode_collect_deliver_recharge_return(env):
    # One episode: collect at the bank, deliver, recharge at the charger, return to the bank.
    _set_hospital(env, (9, 0))
    env.drone_position = env.bank
    assert env.step(COLLECT)[4]["event"] == "collected"
    env.drone_position = env.hospital
    assert env.step(COLLECT)[4]["event"] == "delivered"
    env.drone_position = env.bank
    _, _, terminated, _, info = env.step(WAIT)
    assert info["event"] == "invalid" or not terminated   # not finished yet: no recharge
    env.drone_position = env.charger
    assert env.step(RECHARGE)[4]["event"] == "recharged"
    env.drone_position = env.bank
    _, _, terminated, _, info = env.step(WAIT)
    assert info["event"] == "returned"
    assert terminated
    assert env.mission_complete


def test_delivery_without_recharge_does_not_finish_the_episode(env):
    _set_hospital(env, (9, 0))
    env.drone_position = env.bank
    env.step(COLLECT)
    env.drone_position = env.hospital
    env.step(COLLECT)
    env.drone_position = env.bank
    _, _, terminated, _, info = env.step(WAIT)
    assert not terminated
    assert not env.mission_complete


def test_delivery_success_is_recorded(env):
    env.drone_position = env.bank
    env.step(COLLECT)
    env.drone_position = env.hospital
    _, reward, _, _, info = env.step(COLLECT)
    assert env.delivered
    assert info["event"] == "delivered"
    assert reward > 90


# ---------------------------------------------------------------- weather


@pytest.mark.parametrize(
    "rain, level",
    [(0.0, LIGHT), (2.9, LIGHT), (4.0, MODERATE), (7.0, HEAVY), (9.0, SEVERE)],
)
def test_rain_levels(rain, level):
    assert rain_level(rain) == level


def test_rain_is_local_not_map_wide():
    env = BloodDroneEnv()
    env.reset(seed=2)
    env.storms = [{"x": 8, "y": 8, "radius": 2.5, "age": 10, "life": 50}]
    _, rain_inside = env.weather_at((8, 8))
    _, rain_far = env.weather_at((0, 0))
    assert rain_inside > 5.0
    assert rain_far == 0.0


def test_heavy_rain_outside_a_shelter_is_penalised(env):
    env.storms = [{"x": 1, "y": 0, "radius": 2.5, "age": 10, "life": 50}]
    env.drone_position = (2, 0)          # not a shelter
    _, reward, _, _, info = env.step(WAIT)
    assert info["weather_level"] == HEAVY
    assert reward < -2.5                 # step -0.1, wait -0.5, heavy rain -2


def test_severe_storm_needs_landing_on_a_shelter(env):
    env.storms = [{"x": 5, "y": 5, "radius": 2.5, "age": 10, "life": 50}]
    env.drone_position = env.charger
    env.altitude = 2                     # flying in the storm
    _, reward_flying, _, _, info = env.step(WAIT)
    assert info["weather_level"] == SEVERE
    assert reward_flying < -10

    env.altitude = 0                     # landed on the shelter
    _, reward_landed, _, _, _ = env.step(WAIT)
    assert reward_landed > -5


def test_same_seed_gives_same_weather():
    a, b = BloodDroneEnv(), BloodDroneEnv()
    a.reset(seed=7)
    b.reset(seed=7)
    for _ in range(20):
        a.step(WAIT)
        b.step(WAIT)
    assert a.wind == pytest.approx(b.wind)
    assert a.rain == pytest.approx(b.rain)


# ---------------------------------------------------------------- time limits


def test_deadline_truncates_the_episode():
    env = BloodDroneEnv(max_steps=20)
    env.reset(seed=0)
    env.storms = []
    env.base_wind = 0.0
    env.wind = 0.0
    env.rain = 0.0
    env.obstacles = {}
    truncated = False
    for _ in range(env.max_steps):
        _, reward, terminated, truncated, info = env.step(WAIT)
        if terminated or truncated:
            break
    assert truncated
    assert info["event"] == "deadline"
    assert reward < -30


def test_observation_stays_in_range_during_random_play():
    env = BloodDroneEnv()
    observation, _ = env.reset(seed=3)
    env.action_space.seed(3)
    for _ in range(2000):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            observation, _ = env.reset()


# ---------------------------------------------------------------- hand-written routes
# These tests follow a fixed list of actions, so they show the rules work
# before any learning is involved.


def _set_hospital(env, cell):
    """Move the hospital and recompute the walking distances."""
    if cell not in env.hospital_options:
        env.hospital_options.append(cell)     # so the distance table includes it
    env.hospital = cell
    env.dist_maps = {c: env.flight_distances(c) for c in [env.bank, env.charger, *env.hospital_options]}
    env.flight_dist = env.flight_distances(env.bank)


def test_manual_route_bank_hospital_charger_bank(env):
    # Bank (0,0) -> hospital (9,0) -> charger (5,5) -> bank (0,0), with the blood delivered and recharged.
    _set_hospital(env, (9, 0))
    route = ([COLLECT] + [EAST] * 9 + [COLLECT]                 # collect, fly to hospital, deliver
             + [WEST] * 4 + [SOUTH] * 5 + [RECHARGE]              # fly to the charger and recharge
             + [WEST] * 5 + [NORTH] * 5)                          # fly back to the bank
    events = [env.step(action)[4]["event"] for action in route]
    assert events[0] == "collected"
    assert events[10] == "delivered"
    assert events[20] == "recharged"
    assert events[-1] == "returned"
    assert env.mission_complete


def test_manual_route_around_a_high_tower(env):
    # A high tower at (2,0) cannot be flown over, so the route goes around it.
    env.obstacles = {(2, 0): 3}
    _set_hospital(env, (3, 0))
    route = [COLLECT, SOUTH, EAST, EAST, EAST, NORTH, COLLECT]
    events = [env.step(action)[4]["event"] for action in route]
    assert events[0] == "collected"
    assert events[-1] == "delivered"
    assert env.obstacle_hits == 0


def test_manual_route_over_a_low_tower_by_climbing(env):
    # A low tower at (2,0) is flown over: climb to level 2, then go straight through.
    env.obstacles = {(2, 0): 1}
    _set_hospital(env, (3, 0))
    route = [COLLECT, EAST, CLIMB, CLIMB, EAST, EAST, DESCEND, DESCEND, COLLECT]
    events = [env.step(action)[4]["event"] for action in route]
    assert events[0] == "collected"
    assert events[1] == "move"               # (1,0), not yet blocked
    assert events[4:6] == ["move", "move"]    # over the tower at altitude 2
    assert events[-1] == "delivered"
    assert env.obstacle_hits == 0


def test_mask_never_offers_an_impossible_action(env):
    # On the ground at the bank, Descend and Climb-into-the-map-edge are impossible.
    mask = env.action_mask()
    assert mask[8] == 0.0                     # cannot descend on the ground
    assert mask[NORTH] == 0.0                 # off the map
    assert mask[WAIT] == 1.0                  # waiting is always possible
