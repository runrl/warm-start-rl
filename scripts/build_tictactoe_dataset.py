from __future__ import annotations

import argparse
from pathlib import Path

from c4_scaling.datasets import generate_minimax_dataset, save_imitation_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Tic-Tac-Toe imitation dataset using minimax targets.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the generated dataset (.pt file).")
    parser.add_argument("--num-games", type=int, default=1000, help="Number of self-play games to record.")
    parser.add_argument(
        "--opponent",
        choices=["random", "minimax"],
        default="random",
        help="Opponent strategy used when generating trajectories.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for dataset shuffling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = generate_minimax_dataset(num_games=args.num_games, opponent=args.opponent, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_imitation_dataset(examples, args.output)
    print(f"Saved {len(examples)} state-action pairs to {args.output}")


if __name__ == "__main__":
    main()
