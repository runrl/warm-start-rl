from __future__ import annotations

import argparse
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch

from c4_scaling.policy_gradient import _mask_logits, evaluate_policy
from c4_scaling.ppo_agent import PPOAgent, PPOConfig, Transition
from c4_scaling.tictactoe_env import (
    OpponentStrategy,
    PLAYER_O,
    PLAYER_X,
    TicTacToeEnv,
    enable_minimax_profiling,
    get_minimax_profile_stats,
    minimax_opponent_action,
    reset_minimax_profile,
)

FINAL_MINIMAX_EVAL_EPISODES = 50


@dataclass
class TrainingResult:
    episode_rewards: List[float]
    eval_history: List[Tuple[int, float, float]]
    minimax_history: List[Tuple[int, float, float]]
    final_minimax: Tuple[float, float, float] | None


def moving_average(values: List[float], window: int) -> List[float]:
    window = max(1, window)
    ma: List[float] = []
    running_sum = 0.0
    for idx, value in enumerate(values, start=1):
        running_sum += value
        if idx > window:
            running_sum -= values[idx - window - 1]
        ma.append(running_sum / min(idx, window))
    return ma


def _get_opponent_strategy(name: str):
    if name == "minimax":
        return minimax_opponent_action
    if name == "random":
        return None
    if name == "self":
        return "self"
    raise ValueError(f"Unknown opponent strategy: {name}")


def _encode_self_play_observation(board: torch.Tensor, device: torch.device) -> torch.Tensor:
    board_float = board.to(device=device, dtype=torch.float32) * -1.0
    obs = torch.zeros(2, *board.shape, dtype=torch.float32, device=device)
    obs[0][board_float == float(PLAYER_X)] = 1.0
    obs[1][board_float == float(PLAYER_O)] = 1.0
    current = torch.ones(1, dtype=torch.float32, device=device)
    return torch.cat([obs.view(-1), current])


def _make_self_play_strategy(
    agent: PPOAgent, device: torch.device, deterministic: bool = False
) -> OpponentStrategy:
    def strategy(board: torch.Tensor, legal_actions: List[int]) -> int:
        observation = _encode_self_play_observation(board, device)
        with torch.no_grad():
            if deterministic:
                logits = agent.policy(observation.unsqueeze(0)).squeeze(0)
                masked = _mask_logits(logits, list(legal_actions))
                return int(torch.argmax(masked).item())
            action, _, _ = agent.act(observation, legal_actions)
            return action

    return strategy


def collect_transitions(
    agent: PPOAgent,
    env: TicTacToeEnv,
    num_episodes: int,
    device: torch.device,
) -> Tuple[List[Transition], List[float]]:
    transitions: List[Transition] = []
    episode_rewards: List[float] = []

    for _ in range(num_episodes):
        observation = env.reset().to(device)
        done = False
        total_reward = 0.0

        while not done:
            legal_actions = env.legal_actions()
            action, log_prob, value = agent.act(observation, legal_actions)
            result = env.step(action)

            transitions.append(
                Transition(
                    observation=observation.detach().cpu(),
                    action=action,
                    reward=result.reward,
                    done=result.done,
                    log_prob=log_prob,
                    value=value,
                    legal_actions=list(legal_actions),
                )
            )

            observation = result.observation.to(device)
            done = result.done
            total_reward += result.reward

        episode_rewards.append(total_reward)

    return transitions, episode_rewards


def plot_learning_curve(
    episode_rewards: List[float],
    eval_history: List[Tuple[int, float, float]],
    output_path: Path,
    reward_window: int,
    minimax_history: List[Tuple[int, float, float]] | None = None,
) -> None:
    reward_ma = moving_average(episode_rewards, reward_window)
    if eval_history:
        eval_steps, win_values, draw_values = zip(*eval_history)
    else:
        eval_steps, win_values, draw_values = [], [], []
    if minimax_history:
        mm_steps, mm_win_values, mm_draw_values = zip(*minimax_history)
    else:
        mm_steps, mm_win_values, mm_draw_values = [], [], []

    plt.style.use("ggplot")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(range(1, len(reward_ma) + 1), reward_ma, label=f"Moving avg reward (window={reward_window})", color="tab:blue")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("PPO Training on Tic-Tac-Toe")

    if eval_history or minimax_history:
        ax2 = ax1.twinx()
        if eval_history:
            ax2.plot(eval_steps, win_values, label="Evaluation win rate", color="tab:orange")
            ax2.plot(eval_steps, draw_values, label="Evaluation draw rate", color="tab:green", linestyle="--")
        if minimax_history:
            ax2.plot(mm_steps, mm_win_values, label="Minimax win rate", color="tab:red", linestyle="-.")
            ax2.plot(mm_steps, mm_draw_values, label="Minimax draw rate", color="tab:purple", linestyle=":")
        ax2.set_ylabel("Rate")
        ax2.set_ylim(0, 1)
        ax2.grid(False)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    else:
        ax1.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on Tic-Tac-Toe.")
    parser.add_argument("--episodes", type=int, default=20000, help="Total number of training episodes.")
    parser.add_argument("--batch-size", type=int, default=64, help="Episodes per PPO update.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clipping coefficient.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--update-epochs", type=int, default=4, help="Gradient epochs per PPO update.")
    parser.add_argument("--minibatch-size", type=int, default=64, help="Minibatch size for PPO updates.")
    parser.add_argument("--eval-interval", type=int, default=500, help="Episodes between evaluations.")
    parser.add_argument("--eval-episodes", type=int, default=500, help="Evaluation episodes.")
    parser.add_argument("--eval-minimax-episodes", type=int, default=0, help="Episodes for minimax evaluation (0 to skip).")
    parser.add_argument("--reward-ma-window", type=int, default=500, help="Window for moving average of rewards.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Artifact directory.")
    parser.add_argument("--output-file", type=str, default="tictactoe_ppo_performance.png", help="Learning curve filename.")
    parser.add_argument(
        "--opponent",
        choices=["random", "minimax", "self"],
        default="random",
        help="Opponent strategy during training.",
    )
    parser.add_argument("--model-hidden-dim", type=int, default=128, help="Transformer hidden dimension (d_model).")
    parser.add_argument("--model-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--model-depth", type=int, default=2, help="Number of decoder blocks.")
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true", help="Force CUDA usage when available.")
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false", help="Disable CUDA even if available.")
    parser.add_argument("--use-mps", dest="use_mps", action="store_true", help="Force Apple Metal (MPS) usage when available.")
    parser.add_argument("--no-mps", dest="use_mps", action="store_false", help="Disable Apple Metal (MPS) even if available.")
    parser.set_defaults(use_cuda=None, use_mps=None)
    parser.add_argument("--log-every-episode", action="store_true", help="Print training stats after every episode.")
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Sample evaluation actions instead of greedy argmax for smoother metrics.",
    )
    parser.add_argument(
        "--profile-minimax",
        action="store_true",
        help="Collect timing statistics for minimax calls during evaluations.",
    )
    return parser.parse_args()


def _select_cuda_device() -> torch.device:
    """Pick the CUDA device with the lightest load using utilization and free memory."""
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return torch.device("cuda")

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_env:
        visible = [token.strip() for token in visible_env.split(",") if token.strip()]
        mapping = {
            local_idx: int(token)
            for local_idx, token in enumerate(visible)
            if token.isdigit()
        }
    else:
        mapping = {idx: idx for idx in range(device_count)}

    util_map: Dict[int, float] = {}
    try:
        query = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        result = subprocess.run(query, capture_output=True, text=True, check=True)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        for physical_idx, line in enumerate(lines):
            try:
                util_map[physical_idx] = float(line)
            except ValueError:
                continue
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    candidates: List[Tuple[float, float, float, int]] = []
    for local_idx in range(device_count):
        try:
            with torch.cuda.device(local_idx):
                free_bytes, _ = torch.cuda.mem_get_info()
        except RuntimeError:
            continue
        physical_idx = mapping.get(local_idx, local_idx)
        util = util_map.get(physical_idx, float("inf"))
        jitter = random.random()
        candidates.append((util, -float(free_bytes), jitter, local_idx))

    if not candidates:
        return torch.device("cuda")

    candidates.sort()
    best_local_idx = candidates[0][3]
    return torch.device(f"cuda:{best_local_idx}")


def _select_device(use_cuda: Optional[bool], use_mps: Optional[bool]) -> torch.device:
    cuda_available = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = mps_backend is not None and torch.backends.mps.is_available()

    if use_cuda is None:
        use_cuda = cuda_available
    if use_cuda and cuda_available:
        return _select_cuda_device()

    if use_mps is None:
        use_mps = mps_available
    if use_mps and mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def train(
    args: argparse.Namespace,
    *,
    initial_state: Optional[Dict[str, torch.Tensor]] = None,
    log_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    stop_callback: Optional[Callable[[Dict[str, float]], bool]] = None,
) -> TrainingResult:
    device = _select_device(args.use_cuda, getattr(args, "use_mps", False))
    opponent_strategy = _get_opponent_strategy(args.opponent)
    env_device = torch.device("cpu")
    env = TicTacToeEnv(
        device=env_device,
        opponent_strategy=None if opponent_strategy == "self" else opponent_strategy,
    )
    obs_dim = env.reset().numel()

    config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.model_hidden_dim,
        num_heads=args.model_heads,
        depth=args.model_depth,
    )
    agent = PPOAgent(obs_dim=obs_dim, config=config, device=device)
    if initial_state is not None:
        if "policy_state" in initial_state:
            agent.policy.load_state_dict(initial_state["policy_state"])
        else:
            agent.policy.load_state_dict(initial_state)
        if "value_state" in initial_state:
            agent.value_function.load_state_dict(initial_state["value_state"])

    self_play = args.opponent == "self"
    if self_play:
        env.opponent_strategy = _make_self_play_strategy(agent, device, deterministic=False)

    episode_rewards: List[float] = []
    eval_history: List[Tuple[int, float, float]] = []
    minimax_history: List[Tuple[int, float, float]] = []
    last_minimax: Tuple[int, float, float] | None = None
    next_minimax_eval = args.eval_interval if args.eval_interval > 0 else None
    final_minimax_eval: Tuple[float, float, float] | None = None

    def minimax_env_factory() -> TicTacToeEnv:
        return TicTacToeEnv(device=torch.device("cpu"), opponent_strategy=minimax_opponent_action)

    profile_minimax = bool(getattr(args, "profile_minimax", False))
    enable_minimax_profiling(profile_minimax)

    def _start_minimax_profile() -> None:
        if profile_minimax:
            reset_minimax_profile()

    def _collect_minimax_profile(label: str) -> Dict[str, float]:
        if not profile_minimax:
            return {}
        stats = get_minimax_profile_stats()
        if not stats:
            return {}
        summary_parts = []
        payload: Dict[str, float] = {}
        for key, values in stats.items():
            count = int(values.get("count", 0))
            total = float(values.get("total_seconds", 0.0))
            avg = total / count if count > 0 else 0.0
            summary_parts.append(f"{key}: {total:.3f}s ({count} calls, avg {avg:.4f}s)")
            payload[f"profile_{key}_seconds"] = total
            payload[f"profile_{key}_count"] = float(count)
            payload[f"profile_{key}_avg_seconds"] = avg
        if summary_parts:
            print(f"[minimax] {label}: " + "; ".join(summary_parts))
        reset_minimax_profile()
        return payload

    baseline_log_parts = ["Episode 0: reward=--"]
    baseline_profile = {
        "collect_seconds": 0.0,
        "update_seconds": 0.0,
        "eval_seconds": 0.0,
    }

    def _run_training_body() -> TrainingResult:
        nonlocal next_minimax_eval
        if args.eval_minimax_episodes > 0:
            _start_minimax_profile()
            eval_start = time.perf_counter()
            mm_win, mm_draw = evaluate_policy(
                env_factory=minimax_env_factory,
                policy=agent.policy,
                episodes=args.eval_minimax_episodes,
                device=device,
                stochastic=args.stochastic_eval,
            )
            baseline_profile["eval_seconds"] = time.perf_counter() - eval_start
            minimax_history.append((0, mm_win, mm_draw))
            mm_lose = max(0.0, 1.0 - mm_win - mm_draw)
            last_minimax = (0, mm_win, mm_draw)
            baseline_log_parts.append(
                f"minimax_win={mm_win:.3f}, minimax_draw={mm_draw:.3f}, minimax_lose={mm_lose:.3f}"
            )
            minimax_profile_payload = _collect_minimax_profile("baseline_minimax_eval")
            if log_callback is not None:
                log_callback(
                    {
                        "total_episodes": 0,
                        "minimax_win_rate": mm_win,
                        "minimax_draw_rate": mm_draw,
                        "minimax_lose_rate": mm_lose,
                        "profile_collect_seconds": baseline_profile["collect_seconds"],
                        "profile_update_seconds": baseline_profile["update_seconds"],
                        "profile_eval_seconds": baseline_profile["eval_seconds"],
                        **minimax_profile_payload,
                    }
                )
        else:
            baseline_log_parts.append("minimax_win=--, minimax_draw=--, minimax_lose=--")

        baseline_log_parts.append(
            "profiling="
            + ", ".join(
                [
                    f"collect={baseline_profile['collect_seconds']:.3f}s",
                    f"update={baseline_profile['update_seconds']:.3f}s",
                    f"eval={baseline_profile['eval_seconds']:.3f}s",
                ]
            )
        )
        print(", ".join(baseline_log_parts))

        total_episodes = 0
        stop_training = False
        while total_episodes < args.episodes and not stop_training:
            effective_batch = 1 if args.log_every_episode else args.batch_size
            batch_episodes = min(max(effective_batch, 1), args.episodes - total_episodes)
            collect_start = time.perf_counter()
            transitions, batch_rewards = collect_transitions(agent, env, batch_episodes, device)
            profile_collect_seconds = time.perf_counter() - collect_start

            update_start = time.perf_counter()
            update_metrics = agent.update(transitions)
            profile_update_seconds = time.perf_counter() - update_start
            episode_rewards.extend(batch_rewards)
            total_episodes += batch_episodes

            mm_win = mm_draw = mm_lose = None
            profile_eval_seconds = 0.0
            if args.eval_minimax_episodes > 0:
                if next_minimax_eval is None:
                    eval_step = total_episodes
                    if eval_step > 0:
                        _start_minimax_profile()
                        eval_start = time.perf_counter()
                        mm_win, mm_draw = evaluate_policy(
                            env_factory=minimax_env_factory,
                            policy=agent.policy,
                            episodes=args.eval_minimax_episodes,
                            device=device,
                            stochastic=args.stochastic_eval,
                        )
                        eval_duration = time.perf_counter() - eval_start
                        profile_eval_seconds += eval_duration
                        mm_lose = max(0.0, 1.0 - mm_win - mm_draw)
                        minimax_history.append((eval_step, mm_win, mm_draw))
                        last_minimax = (eval_step, mm_win, mm_draw)
                        if log_callback is not None:
                            minimax_profile_payload = _collect_minimax_profile(f"minimax_eval@{eval_step}")
                            log_callback(
                                {
                                    "total_episodes": eval_step,
                                    "minimax_win_rate": mm_win,
                                    "minimax_draw_rate": mm_draw,
                                    "minimax_lose_rate": mm_lose,
                                    "profile_collect_seconds": 0.0,
                                    "profile_update_seconds": 0.0,
                                    "profile_eval_seconds": eval_duration,
                                    **minimax_profile_payload,
                                }
                            )
                else:
                    while total_episodes >= next_minimax_eval and next_minimax_eval <= args.episodes:
                        eval_step = next_minimax_eval
                        _start_minimax_profile()
                        eval_start = time.perf_counter()
                        mm_win, mm_draw = evaluate_policy(
                            env_factory=minimax_env_factory,
                            policy=agent.policy,
                            episodes=args.eval_minimax_episodes,
                            device=device,
                            stochastic=args.stochastic_eval,
                        )
                        eval_duration = time.perf_counter() - eval_start
                        profile_eval_seconds += eval_duration
                        mm_lose = max(0.0, 1.0 - mm_win - mm_draw)
                        minimax_history.append((eval_step, mm_win, mm_draw))
                        last_minimax = (eval_step, mm_win, mm_draw)
                        if log_callback is not None:
                            minimax_profile_payload = _collect_minimax_profile(f"minimax_eval@{eval_step}")
                            log_callback(
                                {
                                    "total_episodes": eval_step,
                                    "minimax_win_rate": mm_win,
                                    "minimax_draw_rate": mm_draw,
                                    "minimax_lose_rate": mm_lose,
                                    "profile_collect_seconds": 0.0,
                                    "profile_update_seconds": 0.0,
                                    "profile_eval_seconds": eval_duration,
                                    **minimax_profile_payload,
                                }
                            )
                        next_minimax_eval += args.eval_interval

            avg_reward = sum(batch_rewards) / max(len(batch_rewards), 1)
            episode_label = "Episode" if batch_episodes == 1 else "Episodes"
            log_parts = [
                f"{episode_label} {total_episodes}: reward_avg={avg_reward:.2f}",
                f"policy_loss={update_metrics.get('policy_loss', 0.0):.4f}",
                f"value_loss={update_metrics.get('value_loss', 0.0):.4f}",
            ]

            if last_minimax is not None:
                _, win_rate, draw_rate = last_minimax
                lose_rate = max(0.0, 1.0 - win_rate - draw_rate)
                log_parts.append(
                    f"minimax_win={win_rate:.3f}, minimax_draw={draw_rate:.3f}, minimax_lose={lose_rate:.3f}"
                )
            else:
                log_parts.append("minimax_win=--, minimax_draw=--, minimax_lose=--")

            log_parts.append(
                "profiling="
                + ", ".join(
                    [
                        f"collect={profile_collect_seconds:.3f}s",
                        f"update={profile_update_seconds:.3f}s",
                        f"eval={profile_eval_seconds:.3f}s",
                    ]
                )
            )
            print(", ".join(log_parts))
            if log_callback is not None:
                minimax_profile_payload = {}
                if profile_minimax and mm_win is not None:
                    minimax_profile_payload = _collect_minimax_profile(f"minimax_eval@{total_episodes}")
                payload = {
                    "total_episodes": total_episodes,
                    "reward_avg": avg_reward,
                    "policy_loss": update_metrics.get("policy_loss", 0.0),
                    "value_loss": update_metrics.get("value_loss", 0.0),
                    "profile_collect_seconds": profile_collect_seconds,
                    "profile_update_seconds": profile_update_seconds,
                    "profile_eval_seconds": profile_eval_seconds,
                    **minimax_profile_payload,
                }
                if mm_win is not None:
                    payload.update(
                        {
                            "minimax_win_rate": mm_win,
                            "minimax_draw_rate": mm_draw,
                            "minimax_lose_rate": mm_lose,
                        }
                    )
                elif last_minimax is not None:
                    _, win_rate, draw_rate = last_minimax
                    payload.update(
                        {
                            "minimax_win_rate": win_rate,
                            "minimax_draw_rate": draw_rate,
                            "minimax_lose_rate": max(0.0, 1.0 - win_rate - draw_rate),
                        }
                    )
                log_callback(payload)
            else:
                payload = {
                    "total_episodes": total_episodes,
                }
                if mm_win is not None:
                    payload.update(
                        {
                            "minimax_win_rate": mm_win,
                            "minimax_draw_rate": mm_draw,
                            "minimax_lose_rate": mm_lose,
                        }
                    )
                elif last_minimax is not None:
                    _, win_rate, draw_rate = last_minimax
                    payload.update(
                        {
                            "minimax_win_rate": win_rate,
                            "minimax_draw_rate": draw_rate,
                            "minimax_lose_rate": max(0.0, 1.0 - win_rate - draw_rate),
                        }
                    )

            if stop_callback is not None and stop_callback(payload):
                stop_training = True

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        plot_learning_curve(
            episode_rewards=episode_rewards,
            eval_history=eval_history,
            output_path=Path(args.output_dir) / args.output_file,
            reward_window=args.reward_ma_window,
            minimax_history=minimax_history if args.eval_minimax_episodes > 0 else None,
        )

        torch.save(
            {
                "policy_state": agent.policy.state_dict(),
                "value_state": agent.value_function.state_dict(),
                "config": vars(args),
            },
            Path(args.output_dir) / "tictactoe_ppo_policy.pt",
        )
        final_eval_start = time.perf_counter()
        _start_minimax_profile()
        final_minimax_win, final_minimax_draw = evaluate_policy(
            env_factory=minimax_env_factory,
            policy=agent.policy,
            episodes=FINAL_MINIMAX_EVAL_EPISODES,
            device=device,
            stochastic=args.stochastic_eval,
        )
        final_eval_duration = time.perf_counter() - final_eval_start
        final_minimax_lose = max(0.0, 1.0 - final_minimax_win - final_minimax_draw)
        minimax_profile_payload = _collect_minimax_profile("final_minimax_eval")
        print(
            f"Final minimax evaluation ({FINAL_MINIMAX_EVAL_EPISODES} games): "
            f"win_rate={final_minimax_win:.3f}, draw_rate={final_minimax_draw:.3f}, lose_rate={final_minimax_lose:.3f}"
        )
        print(f"Saved PPO artifacts to {args.output_dir}")
        if log_callback is not None:
            log_callback(
                {
                    "total_episodes": args.episodes,
                    "final_minimax_win_rate": final_minimax_win,
                    "final_minimax_draw_rate": final_minimax_draw,
                    "final_minimax_lose_rate": final_minimax_lose,
                    "profile_collect_seconds": 0.0,
                    "profile_update_seconds": 0.0,
                    "profile_eval_seconds": final_eval_duration,
                    **minimax_profile_payload,
                }
            )
        final_minimax_eval = (final_minimax_win, final_minimax_draw, final_minimax_lose)
        return TrainingResult(
            episode_rewards=episode_rewards,
            eval_history=eval_history,
            minimax_history=minimax_history,
            final_minimax=final_minimax_eval,
        )

    try:
        return _run_training_body()
    finally:
        if profile_minimax:
            enable_minimax_profiling(False)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
