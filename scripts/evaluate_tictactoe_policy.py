from __future__ import annotations

import argparse
from pathlib import Path

import torch

from c4_scaling.policy_gradient import PolicyNetwork, evaluate_policy
from c4_scaling.tictactoe_env import TicTacToeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Tic-Tac-Toe policy against a random opponent.")
    parser.add_argument("--policy", type=Path, required=True, help="Path to a saved policy checkpoint.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes.")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--stochastic", action="store_true", help="Sample evaluation actions instead of greedy argmax.")
    return parser.parse_args()


def _extract_model_hyperparams(checkpoint: dict) -> tuple[int, int, int]:
    source = checkpoint.get("args") or checkpoint.get("config") or {}
    hidden_dim = (
        source.get("hidden_dim")
        or source.get("model_hidden_dim")
        or source.get("policy_hidden_dim")
        or 128
    )
    num_heads = source.get("model_heads") or source.get("num_heads") or 4
    depth = source.get("model_depth") or source.get("depth") or 2
    return int(hidden_dim), int(num_heads), int(depth)


def load_policy(path: Path, device: torch.device) -> PolicyNetwork:
    checkpoint = torch.load(path, map_location=device)
    hidden_dim, num_heads, depth = _extract_model_hyperparams(checkpoint)
    env = TicTacToeEnv(device=device)
    obs_dim = env.reset().numel()
    policy = PolicyNetwork(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        n_head=num_heads,
        depth=depth,
    ).to(device)
    state_key = "policy_state" if "policy_state" in checkpoint else "model_state"
    policy.load_state_dict(checkpoint[state_key])
    return policy


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    policy = load_policy(args.policy, device)
    win_rate, draw_rate = evaluate_policy(
        env_factory=lambda: TicTacToeEnv(device=device),
        policy=policy,
        episodes=args.episodes,
        device=device,
        stochastic=args.stochastic,
    )
    lose_rate = max(0.0, 1.0 - win_rate - draw_rate)
    print(f"Win rate: {win_rate:.3f}, Draw rate: {draw_rate:.3f}, Lose rate: {lose_rate:.3f}")


if __name__ == "__main__":
    main()
