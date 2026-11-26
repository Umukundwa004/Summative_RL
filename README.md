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

A 10 x 10 map. The drone starts at the blood bank (0, 0), collects the blood, flies it to a hospital, recharges at the charging station (5, 5) and flies back to the bank. One episode is one full mission, with at most 300 steps. Towers, buildings, the hospital and the weather change in every episode.

**Actions (9)**

| ID | Action | Effect |
|---|---|---|
| 0-3 | Move north / south / east / west | one cell, uses battery |
| 4 | Collect / Deliver | collect at the bank, deliver at the hospital (on the ground only) |
| 5 | Recharge | +25 battery, only at the charging station, on the ground |
| 6 | Wait | hold position, uses a little battery |
| 7 | Climb | go up one height level (0 to 3) |
| 8 | Descend | go down one height level |

Impossible actions are masked, so the agent never chooses them.

**Observation (43 numbers, all scaled to 0-1)**

The full list is in `environment/custom_env.py` (`get_observation`). It includes the drone position, battery, package state, distances, weather, the four neighbouring cells (tower heights and rain), the altitude, whether the drone is ready for a mission, the action mask (9 values) and the position of the current target.

**Rewards**

| Event | Reward |
|---|---|
| Blood collected | +15 |
| Blood delivered | +100 |
| Back at the bank after recharging (mission complete) | +15 |
| Every step | -0.1 |
| Progress toward the current target (potential-based) | about +1 per cell closer, -1 per cell further |
| Invalid action | -3 |
| Waiting | -0.5 |
| Hitting a tower or building | -20 (third hit: crash, -100) |
| Heavy rain outside a shelter (per step) | -2 |
| Severe storm, not landed on a shelter (per step) | -10 |
| Blood too warm | -60 |
| Battery empty away from the charger | -100 |
| Deadline (300 steps) | -40 |

Recharging gives no reward on purpose. Otherwise the agent earned points by recharging again and again.

**Weather** is local. Storms drift, grow and fade. Rain and wind are simulation values from 0 to 10, not real weather data.

## Training and hyperparameter experiments

All four algorithms use the same environment, a network with two hidden layers
of 64 units (ReLU), the same budget (300,000 environment steps by default in the training scripts) and
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
## YOUTUBE LINK : https://youtu.be/GFcS841tubw
