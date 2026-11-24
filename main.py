"""
Blood delivery drone: train an agent with reinforcement learning.

Run with no arguments to pick an algorithm from a menu:
    uv run main.py

Or choose everything on the command line:
    uv run main.py --model ppo --frames 100000

After training, watch the agent with:  uv run play.py --model ppo
"""

import argparse
from pathlib import Path

TOTAL_STEPS = 100_000  # default number of environment steps


def model_path(name):
    return Path("models") / name / f"{name}.pt"


ALGORITHMS = {
    "1": ("DQN", "dqn"),
    "2": ("REINFORCE", "reinforce"),
    "3": ("PPO", "ppo"),
    "4": ("A2C", "a2c"),
}


def choose_model():
    """Display a menu and return the chosen action."""

    print()
    print("=" * 60)
    print("BLOOD DELIVERY DRONE - SELECT MODEL")
    print("=" * 60)

    for number, (name, key) in ALGORITHMS.items():
        status = "trained model found" if Path(model_path(key)).exists() else "not trained yet"
        print(f"  {number}. {name:<10} ({status})")

    while True:
        choice = input("\nEnter 1-4: ").strip()
        if choice in ALGORITHMS:
            name, key = ALGORITHMS[choice]
            print(f"\nSelected: {name}")
            return key
        print("Invalid choice. Please enter 1, 2, 3 or 4.")


def choose_frames(default):
    """Ask how many environment steps to train for."""

    while True:
        text = input(f"Training steps [{default}]: ").strip()
        if not text:
            return default
        if text.replace("_", "").isdigit() and int(text.replace("_", "")) > 0:
            return int(text.replace("_", ""))
        print("Please enter a positive whole number.")


def train(algorithm, frames):
    # Imported here because TorchRL takes a few seconds to load.
    if algorithm == "dqn":
        from training.dqn_training import train_dqn as train_fn
    elif algorithm == "reinforce":
        from training.reinforce_training import train_reinforce as train_fn
    elif algorithm == "ppo":
        from training.ppo_training import train_ppo as train_fn
    else:
        from training.a2c_training import train_a2c as train_fn

    print(f"\nStarting {algorithm.upper()} training for {frames:,} environment steps...\n")
    train_fn(
        {"total_frames": frames},
        run_name=f"main_{algorithm}",
        model_path=model_path(algorithm),
    )
    print(f"\nTraining completed! Model saved to {model_path(algorithm)}")
    print(f"Watch it fly:  uv run play.py --model {algorithm}")


def main():
    parser = argparse.ArgumentParser(description="Train the autonomous blood delivery drone")
    parser.add_argument(
        "--model", choices=[key for _, key in ALGORITHMS.values()], default=None,
        help="algorithm to train (leave out to choose from a menu)",
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help=f"environment steps to train for (default {TOTAL_STEPS:,})",
    )
    args = parser.parse_args()

    choice = args.model
    if choice is None:
        choice = choose_model()
        if args.frames is None:
            args.frames = choose_frames(TOTAL_STEPS)

    train(choice, args.frames or TOTAL_STEPS)


if __name__ == "__main__":
    main()
