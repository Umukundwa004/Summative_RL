# Blood Delivery Drone: Reinforcement Learning

An autonomous drone, inspired by medical drone delivery in Rwanda, flies blood
from a blood bank to a hospital. It must avoid radio towers, recharge when its
battery is low, wait out storms and keep the blood cold. Four reinforcement
learning algorithms are trained on the same environment and compared:

| Value-based | Policy gradient |
|---|---|
| DQN | REINFORCE, PPO, A2C |

The training code uses **TorchRL**. This is an educational simulation. It is not
a system for controlling real drones or moving real blood products.

## Quick start (uv only)

```bash
uv sync                              # installs everything, no manual setup
uv run main.py                       # menu: train DQN / REINFORCE / PPO / A2C
uv run play.py --model dqn           # watch the best trained agent (pygame window + terminal log)
uv run pytest                        # environment tests
```

Train one algorithm directly: `uv run main.py --model ppo --frames 100000`

## Project structure

```text
├── pyproject.toml        dependencies (managed with uv)
├── uv.lock
├── main.py               train an algorithm (menu)
├── play.py               watch an agent fly (pygame + terminal output)
├── export_episode.py     save a mission as JSON for the 3D viewer
│
├── environment/
│   ├── custom_env.py     the Gymnasium environment (BloodDroneEnv)
│   └── rendering.py      2D pygame drawing used by play.py
│
├── training/             one self-contained file per algorithm (TorchRL)
│   ├── dqn_training.py
│   ├── reinforce_training.py
│   ├── ppo_training.py
│   └── a2c_training.py
│
├── models/               trained model of each algorithm
├── notebooks/            one notebook per algorithm: training + 10 hyperparameter experiments
├── frontend/             3D viewer (Three.js + Vite)
├── assets/               screenshots and figures
└── tests/                environment tests
```

## The environment

A 10 x 10 map. The drone starts at the blood bank, must collect the blood
(action 4), fly to the hospital and deliver it (action 4), then return to the
bank. A mission has at most 200 steps.

**Actions (7)**

| ID | Action | Effect |
|---|---|---|
| 0-3 | Move north / south / east / west | one cell, uses battery |
| 4 | Collect / Deliver | collect at the blood bank, deliver at the hospital |
| 5 | Recharge | +25 battery, only at the charging station |
| 6 | Wait | hold position, uses little battery, lets storms pass |

**Observation (22 numbers, all scaled to 0-1)**

| # | Feature | Source in the simulation |
|---|---|---|
| 1-2 | Drone x, y | current cell / 9 |
| 3 | Battery | battery / 100, drops with every action |
| 4 | Blood collected | 0 or 1 |
| 5-7 | Distance to hospital, bank, charger | Manhattan distance / 18 |
| 8 | Delivery urgency | distance to current target / steps left |
| 9 | Time remaining | (200 - step) / 200 |
| 10-11 | Wind, rain | weather in the drone's own cell (0-10) / 10 |
| 12 | Package temperature | warms while carried, / 10 |
| 13 | Package integrity | 100 minus damage, / 100 |
| 14-17 | Tower to the north / south / east / west | 1 if the next cell is blocked |
| 18 | Mission stage | 0 collect, 0.5 carry, 1 return |
| 19-22 | Rain to the north / south / east / west | rain in the next cell / 10, shows where the storm is |

**Rewards**

| Event | Reward |
|---|---|
| Blood collected | +15 |
| Blood delivered | +100 |
| Back at base after delivery | +15 |
| Moving closer to the current target | +2 per cell |
| Recharging at the station | +5 |
| Every step | -0.1 |
| Invalid action (off the map, wrong place to collect or recharge) | -3 |
| Flying into a tower | -20 (third hit: crash, -100) |
| Missing the deadline | -40 |
| Blood too warm | -60 |
| Battery empty away from the charger | -100 |

**Start state:** drone at the bank, battery 100, blood not collected, calm
weather. **Terminal states:** mission complete, crash (3 tower hits), battery
empty, blood spoiled, or the 200-step deadline.

Weather is local. The map has a calm base level, and one or two storm patches
(about 5 cells wide) appear, drift and fade away. Rain and wind are strongest in
the middle of a storm and are zero or low outside it, so the drone can wait for
a storm to pass or fly around it. Wind and rain are simulation values on a 0-10
scale that I chose; they are not real weather data. Above wind 6 the drone can
be pushed in a random direction, and flying costs more battery in wind and rain.
A move costs 1.5 + 0.25 x wind + 0.3 x rain battery, so a storm makes flying about
three times as expensive. Waiting costs 0.4 battery and a small reward penalty every
step, so the drone does not just hover, but waiting out a storm is still possible.

## Training and hyperparameter experiments

All four algorithms use the same environment, a network with two hidden layers
of 64 units (ReLU), the same budget (100,000 environment steps by default) and
the same evaluation episodes. Each file in `training/` contains everything it
needs.

Each notebook in `notebooks/` trains one algorithm and then runs **10
experiments**: experiment 1 is the baseline and every other experiment changes
one hyperparameter. The notebook shows a results table and curves, picks the
experiment with the highest average reward (ties go to higher delivery success,
then smaller loss) and can write the best values into `DEFAULTS` in the
training file (`APPLY_BEST = True`).

## 3D viewer

```bash
uv run python export_episode.py --seeds 0 1 2 3  # export varied maps/routes for each model
cd frontend && npm install && npm run dev
```

The viewer plays the saved missions in 3D (towers, city, forest, rain) and shows
every action with its reward. Use the episode selector to switch between models
and seeds. The environment requires each delivery to be followed by a recharge
at `(5, 5)` before returning to the bank. Episodes are saved as
`frontend/public/episodes/*.json`, so any other front end (web or mobile) can
use the same JSON.
