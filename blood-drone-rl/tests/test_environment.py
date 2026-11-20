"""Tests for the blood delivery drone environment."""

from collections import deque

import numpy as np
import pytest

from environment.custom_env import BloodDroneEnv, MOVES

NORTH, SOUTH, EAST, WEST, COLLECT, RECHARGE, WAIT = range(7)


@pytest.fixture
def env():
    env = BloodDroneEnv()
    env.reset(seed=0)
    # Calm weather so the tests are predictable.
    env.wind = 0.0
    env.rain = 0.0
    return env


def path_exists(env, start, goal):
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for dx, dy in MOVES.values():
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt not in seen and not env.is_blocked(nxt):
                seen.add(nxt)
                queue.append(nxt)
    return False


def test_spaces():
    env = BloodDroneEnv()
    assert env.action_space.n == 7
    assert env.observation_space.shape == (18,)


def test_reset_returns_valid_observation(env):
    observation, _ = env.reset(seed=1)
    assert observation.shape == (18,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert env.drone_position == env.bank
    assert env.battery == 100.0
    assert not env.package_collected


def test_map_is_solvable(env):
    assert path_exists(env, env.bank, env.charger)
    assert path_exists(env, env.charger, env.hospital)
    assert path_exists(env, env.hospital, env.bank)
    assert env.bank not in env.obstacles
    assert env.hospital not in env.obstacles
    assert env.charger not in env.obstacles


def test_collect_at_bank_gives_reward(env):
    _, reward, terminated, _, info = env.step(COLLECT)
    assert env.package_collected
    assert info["event"] == "collected"
    assert reward > 10
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


def test_obstacle_collision_is_penalised(env):
    env.drone_position = (3, 0)  # a tower is at (4, 0)
    _, reward, _, _, info = env.step(EAST)
    assert info["event"] == "obstacle"
    assert reward <= -20
    assert env.drone_position == (3, 0)
    assert env.obstacle_hits == 1


def test_three_collisions_crash_the_drone(env):
    env.drone_position = (3, 0)
    terminated = False
    for _ in range(3):
        _, reward, terminated, _, info = env.step(EAST)
    assert terminated
    assert info["event"] == "crash"
    assert reward <= -100


def test_moving_toward_target_beats_moving_away(env):
    env.step(COLLECT)  # target becomes the hospital
    _, closer, _, _, _ = env.step(EAST)
    _, farther, _, _, _ = env.step(WEST)
    assert closer > farther


def test_recharge_only_works_at_the_charger(env):
    env.battery = 40.0
    _, reward, _, _, info = env.step(RECHARGE)
    assert info["event"] == "invalid"
    assert env.battery <= 40.0

    env.drone_position = env.charger
    _, reward, _, _, info = env.step(RECHARGE)
    assert info["event"] == "recharged"
    assert env.battery > 40.0
    assert reward > 0


def test_wait_costs_little_battery(env):
    before = env.battery
    env.step(WAIT)
    assert 0 < before - env.battery < 1.0


def test_full_mission_collect_deliver_return(env):
    env.step(COLLECT)
    env.drone_position = env.hospital
    _, reward, _, _, info = env.step(COLLECT)
    assert env.delivered
    assert info["event"] == "delivered"
    assert reward > 90

    env.drone_position = (1, 0)
    _, reward, terminated, _, info = env.step(WEST)
    assert terminated
    assert info["event"] == "returned"
    assert env.mission_complete


def test_battery_exhaustion_ends_the_mission(env):
    env.battery = 0.5
    env.drone_position = (1, 0)
    terminated = False
    for action in (EAST, WEST, EAST):
        _, reward, terminated, _, info = env.step(action)
        if terminated:
            break
    assert terminated
    assert info["event"] == "battery_dead"


def test_deadline_truncates_the_episode(env):
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


def test_same_seed_gives_same_weather():
    a, b = BloodDroneEnv(), BloodDroneEnv()
    a.reset(seed=7)
    b.reset(seed=7)
    for _ in range(20):
        a.step(WAIT)
        b.step(WAIT)
    assert a.wind == pytest.approx(b.wind)
    assert a.rain == pytest.approx(b.rain)
