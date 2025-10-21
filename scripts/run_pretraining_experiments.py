from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from statistics import mean, stdev
import yaml

try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional
    wandb = None

from c4_scaling.policy_gradient import PolicyNetwork
from c4_scaling.tictactoe_env import TicTacToeEnv

_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAIN_PPO_PATH = _SCRIPT_DIR / "train_tictactoe_ppo.py"
_train_spec = importlib.util.spec_from_file_location("train_tictactoe_ppo", _TRAIN_PPO_PATH)
if _train_spec is None or _train_spec.loader is None:  # pragma: no cover - defensive
    raise ImportError("Unable to load train_tictactoe_ppo module")
_train_module = importlib.util.module_from_spec(_train_spec)
sys.modules["train_tictactoe_ppo"] = _train_module
_train_spec.loader.exec_module(_train_module)
TrainingResult = _train_module.TrainingResult
train = _train_module.train


@dataclass
class ExperimentRun:
    name: str
    pretrain_steps: int
    seeds: List[int] = field(default_factory=list)
    tags: Optional[List[str]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO pretraining scaling experiments from a YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretraining_experiment.yaml"),
        help="Path to the experiment configuration YAML.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging regardless of config settings.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for experiments (1 disables parallelism).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("reports/local_metrics"),
        help="Directory to store per-seed metric CSVs.",
    )
    return parser.parse_args()


def _make_default_args() -> Dict[str, Any]:
    return {
        "episodes": 20000,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "learning_rate": 3e-4,
        "update_epochs": 4,
        "minibatch_size": 64,
        "eval_interval": 500,
        "eval_episodes": 500,
        "eval_minimax_episodes": 0,
        "reward_ma_window": 500,
        "output_dir": "outputs",
        "output_file": "tictactoe_ppo_performance.png",
        "opponent": "random",
        "model_hidden_dim": 128,
        "model_heads": 4,
        "model_depth": 2,
        "use_cuda": None,
        "use_mps": None,
        "log_every_episode": False,
        "stochastic_eval": False,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)


def _init_wandb(run_cfg: Dict[str, Any], disable: bool):
    if disable or wandb is None:
        return None
    mode = run_cfg.get("mode", "online")
    if mode == "disabled":
        return None
    wandb_kwargs = {
        "project": run_cfg.get("project", "ppo-pretraining"),
        "entity": run_cfg.get("entity"),
        "group": run_cfg.get("group"),
        "tags": run_cfg.get("tags"),
        "mode": mode,
    }
    name = run_cfg.get("name")
    if name:
        wandb_kwargs["name"] = name
    return wandb.init(**wandb_kwargs)


def _log_to_wandb(run, metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if run is None:
        return
    clean = {k: v for k, v in metrics.items() if v is not None}
    run.log(clean, step=step)


def _deep_merge(base: Any, override: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override if override is not None else base
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at top-level: {path}")
    return data


def _resolve_config(path: Path) -> Dict[str, Any]:
    raw = _load_yaml_file(path)
    template_ref = raw.pop("template", None)
    model_ref = raw.pop("model_size", None)
    explicit_overrides = raw.pop("overrides", None)

    base: Dict[str, Any] = {}
    if template_ref:
        template_path = (path.parent / template_ref).resolve()
        if not template_path.exists():
            raise FileNotFoundError(f"Template configuration not found: {template_path}")
        base = _resolve_config(template_path)

    merged = _deep_merge(base, raw)

    if model_ref:
        model_path = (path.parent / model_ref).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model-size configuration not found: {model_path}")
        model_cfg = _resolve_config(model_path)
        merged = _deep_merge(merged, model_cfg)

    if explicit_overrides:
        merged = _deep_merge(merged, explicit_overrides)
    return merged


def _architecture_signature(hidden_dim: int, num_heads: int, depth: int) -> tuple[int, int, int]:
    return (int(hidden_dim), int(num_heads), int(depth))


def _checkpoint_architecture(checkpoint: Dict[str, Any]) -> Optional[tuple[int, int, int]]:
    args = checkpoint.get("args")
    if not isinstance(args, dict):
        return None
    try:
        hidden = int(args.get("model_hidden_dim"))
        heads = int(args.get("model_heads"))
        depth = int(args.get("model_depth"))
    except (TypeError, ValueError):
        return None
    return _architecture_signature(hidden, heads, depth)


def _prepare_ppo_args(base: Dict[str, Any], overrides: Dict[str, Any]) -> argparse.Namespace:
    merged = {**_make_default_args(), **base, **overrides}
    return argparse.Namespace(**merged)


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

    if use_mps is None:
        use_mps = mps_available
    if use_mps and mps_available:
        return torch.device("mps")

    if use_cuda is None:
        use_cuda = cuda_available
    if use_cuda and cuda_available:
        return _select_cuda_device()
    return torch.device("cpu")


def _build_policy(obs_dim: int, args: argparse.Namespace, device: torch.device) -> PolicyNetwork:
    policy = PolicyNetwork(
        obs_dim=obs_dim,
        hidden_dim=args.model_hidden_dim,
        n_head=args.model_heads,
        depth=args.model_depth,
    ).to(device)
    return policy


def _episodes_to_target(
    history: Iterable[tuple[int, float, float]],
    target: float,
    consecutive: int = 1,
) -> Optional[int]:
    streak = 0
    for step, _, draw_rate in history:
        if draw_rate >= target:
            streak += 1
            if streak >= consecutive:
                return step
        else:
            streak = 0
    return None


def _execute_experiment(
    run_name: str,
    pretrain_steps: int,
    seed: int,
    checkpoint_dir: str,
    checkpoint_pattern: str,
    ppo_base_cfg: Dict[str, Any],
    wandb_run_cfg: Dict[str, Any],
    base_output_dir: str,
    thresholds: List[float],
    consecutive_required: int,
    tuning_cfg: Dict[str, Any],
    disable_wandb: bool,
    obs_dim: int,
    metrics_dir: Optional[str],
) -> Dict[str, Any]:
    sorted_thresholds = sorted(thresholds) if thresholds else []
    _set_seed(seed)
    metrics_path = Path(metrics_dir).resolve() if metrics_dir else None

    run_display_name = f"{run_name}-seed{seed}"
    wandb_kwargs = {**wandb_run_cfg, "name": run_display_name}
    run = _init_wandb(wandb_kwargs, disable_wandb)
    if run is not None:
        run.config.update(
            {
                "pretrain_steps": pretrain_steps,
                "seed": seed,
            }
        )

    policy_args = _prepare_ppo_args(ppo_base_cfg, {})
    device = _select_device(policy_args.use_cuda, getattr(policy_args, "use_mps", False))
    print(f"[device] {run_display_name}: using {device}")
    policy = _build_policy(obs_dim, policy_args, device)
    desired_arch = _architecture_signature(
        policy_args.model_hidden_dim,
        policy_args.model_heads,
        policy_args.model_depth,
    )
    model_signature = {
        "model_hidden_dim": desired_arch[0],
        "model_heads": desired_arch[1],
        "model_depth": desired_arch[2],
    }

    checkpoint_path = Path(checkpoint_dir) / checkpoint_pattern.format(step=pretrain_steps)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "policy_state" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} missing 'policy_state'")
    checkpoint_arch = _checkpoint_architecture(checkpoint)
    if checkpoint_arch is not None and checkpoint_arch != desired_arch:
        raise RuntimeError(
            (
                f"Checkpoint architecture mismatch for {checkpoint_path}: "
                f"expected hidden={desired_arch[0]}, heads={desired_arch[1]}, depth={desired_arch[2]} but "
                f"found hidden={checkpoint_arch[0]}, heads={checkpoint_arch[1]}, depth={checkpoint_arch[2]}. "
                "Run supervised pretraining for this model size (scripts/pretrain_tictactoe_supervised.py) "
                "and update the config's `pretrained.directory` to point at the matching checkpoints."
            )
        )
    try:
        policy.load_state_dict(checkpoint["policy_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint_path} into policy with architecture "
            f"hidden={desired_arch[0]}, heads={desired_arch[1]}, depth={desired_arch[2]}: {exc}"
        ) from exc
    if "value_state" in checkpoint:
        # load into policy's paired value function representation
        pass
    pretrain_metrics = {
        "policy_loss": checkpoint.get("policy_loss", checkpoint.get("loss")),
        "value_loss": checkpoint.get("value_loss"),
        "accuracy": checkpoint.get("accuracy"),
    }
    if run is not None:
        if pretrain_metrics["policy_loss"] is not None:
            run.summary["pretrain_policy_loss"] = pretrain_metrics["policy_loss"]
        if pretrain_metrics["value_loss"] is not None:
            run.summary["pretrain_value_loss"] = pretrain_metrics["value_loss"]
        if pretrain_metrics["accuracy"] is not None:
            run.summary["pretrain_accuracy"] = pretrain_metrics["accuracy"]
        log_payload = {"pretrain_steps": pretrain_steps}
        if pretrain_metrics["policy_loss"] is not None:
            log_payload["pretrain_policy_loss"] = pretrain_metrics["policy_loss"]
        if pretrain_metrics["value_loss"] is not None:
            log_payload["pretrain_value_loss"] = pretrain_metrics["value_loss"]
        if pretrain_metrics["accuracy"] is not None:
            log_payload["pretrain_accuracy"] = pretrain_metrics["accuracy"]
        run.log(log_payload, step=0)
    pretrain_epoch_offset = 0

    overrides = dict(ppo_base_cfg)
    run_output_dir = Path(base_output_dir) / run_name / f"seed_{seed}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    overrides["output_dir"] = str(run_output_dir)
    overrides.setdefault("output_file", f"{run_name}_seed{seed}.png")
    overrides.setdefault("opponent", "minimax")
    overrides.setdefault("eval_minimax_episodes", 50)
    overrides.setdefault("eval_episodes", 50)
    overrides.setdefault("stochastic_eval", True)


    ppo_args = _prepare_ppo_args(ppo_base_cfg, overrides)

    threshold_tracker = {
        thr: {"streak": 0, "hit": None}
        for thr in sorted_thresholds
    }

    def train_logger(metrics: Dict[str, float]) -> None:
        step_val = metrics.get("total_episodes")
        if step_val is None:
            step_val = ppo_args.episodes
        step = pretrain_epoch_offset + step_val
        _log_to_wandb(run, metrics, step=step)

        if sorted_thresholds and "minimax_draw_rate" in metrics and metrics["minimax_draw_rate"] is not None:
            draw_value = float(metrics["minimax_draw_rate"])
            for thr in sorted_thresholds:
                tracker = threshold_tracker[thr]
                if tracker["hit"] is not None:
                    continue
                if draw_value >= thr:
                    tracker["streak"] += 1
                    if tracker["streak"] >= consecutive_required:
                        tracker["hit"] = step_val
                        if run is not None:
                            key = f"episodes_to_target@{thr:.3f}"
                            run.summary[key] = step_val
                            log_step = pretrain_epoch_offset + step_val
                            payload = {key: step_val, "pretrain_steps": pretrain_steps}
                            if pretrain_metrics is not None:
                                payload["pretrain_policy_loss"] = pretrain_metrics.get("policy_loss")
                                payload["pretrain_value_loss"] = pretrain_metrics.get("value_loss")
                                payload["pretrain_accuracy"] = pretrain_metrics.get("accuracy")
                            run.log(payload, step=log_step)
                else:
                    tracker["streak"] = 0

    def should_stop(metrics: Dict[str, float]) -> bool:
        if not sorted_thresholds:
            return False
        highest = sorted_thresholds[-1]
        tracker = threshold_tracker.get(highest)
        return tracker is not None and tracker["hit"] is not None

    print(f"\n=== Running {run_display_name} ===")
    initial_state = {
        "policy_state": checkpoint["policy_state"],
    }
    if "value_state" in checkpoint:
        initial_state["value_state"] = checkpoint["value_state"]

    result: TrainingResult = train(
        ppo_args,
        initial_state=initial_state,
        log_callback=train_logger,
        stop_callback=should_stop,
    )

    threshold_hits: Dict[float, Optional[int]] = {}
    if sorted_thresholds:
        if result.minimax_history:
            for thr in sorted_thresholds:
                hit = threshold_tracker.get(thr, {}).get("hit")
                if hit is None:
                    hit = _episodes_to_target(result.minimax_history, float(thr), consecutive_required)
                threshold_hits[thr] = hit
        if run is not None and sorted_thresholds:
            primary_threshold = sorted_thresholds[-1]
            run.summary["episodes_to_target"] = threshold_hits.get(primary_threshold)
            table_rows = []
            for thr, episodes in threshold_hits.items():
                if episodes is None:
                    continue
                row = [pretrain_steps, float(thr), episodes]
                row.append(pretrain_metrics.get("policy_loss") if pretrain_metrics else None)
                row.append(pretrain_metrics.get("value_loss") if pretrain_metrics else None)
                row.append(pretrain_metrics.get("accuracy") if pretrain_metrics else None)
                table_rows.append(row)
            if table_rows:
                try:
                    table = wandb.Table(
                        columns=[
                            "pretrain_steps",
                            "threshold",
                            "episodes_to_target",
                            "pretrain_policy_loss",
                            "pretrain_value_loss",
                            "pretrain_accuracy",
                        ],
                        data=table_rows,
                    )
                    log_step = pretrain_epoch_offset + ppo_args.episodes + 1
                    run.log({"episodes_to_target_table": table}, step=log_step)
                except Exception:  # pragma: no cover - wandb optional
                    pass
        print(
            "Target draw rate(s): "
            + ", ".join(
                f">={thr:.3f}: {threshold_hits.get(thr)}" for thr in sorted_thresholds
            )
        )
    else:
        print("No target thresholds configured.")

    final_draw = None
    final_lose = None
    if result.final_minimax is not None:
        _, final_draw, final_lose = result.final_minimax

    if run is not None:
        run.finish()

    if metrics_path is not None:
        run_metrics_dir = metrics_path / run_name
        seed_dir = run_metrics_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        if sorted_thresholds:
            episodes_path = seed_dir / "episodes_to_target.csv"
            with episodes_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        "pretrain_steps",
                        "seed",
                        "threshold",
                        "episodes_to_target",
                        "model_hidden_dim",
                        "model_heads",
                        "model_depth",
                        "pretrain_policy_loss",
                        "pretrain_value_loss",
                        "pretrain_accuracy",
                    ]
                )
                for thr in sorted_thresholds:
                    hit = threshold_hits.get(thr) if threshold_hits else None
                    writer.writerow(
                        [
                            pretrain_steps,
                            seed,
                            thr,
                            "" if hit is None else hit,
                            model_signature["model_hidden_dim"],
                            model_signature["model_heads"],
                            model_signature["model_depth"],
                            pretrain_metrics.get("policy_loss") if pretrain_metrics else None,
                            pretrain_metrics.get("value_loss") if pretrain_metrics else None,
                            pretrain_metrics.get("accuracy") if pretrain_metrics else None,
                        ]
                    )

        if result.minimax_history:
            minimax_path = seed_dir / "minimax_history.csv"
            with minimax_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["episode", "win_rate", "draw_rate"])
                for episode, win_rate, draw_rate in result.minimax_history:
                    writer.writerow([episode, win_rate, draw_rate])

        if result.eval_history:
            eval_path = seed_dir / "evaluation_history.csv"
            with eval_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["episode", "win_rate", "draw_rate"])
                for episode, win_rate, draw_rate in result.eval_history:
                    writer.writerow([episode, win_rate, draw_rate])

        metadata_path = seed_dir / "summary.json"
        metadata = {
            "run_name": run_display_name,
            "pretrain_steps": pretrain_steps,
            "seed": seed,
             "model_hidden_dim": model_signature["model_hidden_dim"],
             "model_heads": model_signature["model_heads"],
             "model_depth": model_signature["model_depth"],
            "threshold_hits": threshold_hits,
            "pretrain_policy_loss": pretrain_metrics.get("policy_loss") if pretrain_metrics else None,
            "pretrain_value_loss": pretrain_metrics.get("value_loss") if pretrain_metrics else None,
            "pretrain_accuracy": pretrain_metrics.get("accuracy") if pretrain_metrics else None,
        }
        if metrics_path is not None:
            metadata["run_id"] = metrics_path.name
            metadata["metrics_root"] = str(metrics_path)
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    return {
        "run_name": run_display_name,
        "seed": seed,
        "pretrain_steps": pretrain_steps,
        "model_hidden_dim": model_signature["model_hidden_dim"],
        "model_heads": model_signature["model_heads"],
        "model_depth": model_signature["model_depth"],
        "threshold_hits": threshold_hits,
        "final_minimax_draw": final_draw,
        "final_minimax_lose": final_lose,
        "pretrain_policy_loss": pretrain_metrics.get("policy_loss") if pretrain_metrics else None,
        "pretrain_value_loss": pretrain_metrics.get("value_loss") if pretrain_metrics else None,
        "pretrain_accuracy": pretrain_metrics.get("accuracy") if pretrain_metrics else None,
        "minimax_history": result.minimax_history,
        "eval_history": result.eval_history,
    }


def _print_summary(results: List[Dict[str, Any]], thresholds: List[float]) -> None:
    if not results:
        return
    thresholds = sorted(thresholds)
    threshold_labels = [f"Episodes@≥{thr:.2f}" for thr in thresholds] if thresholds else ["Episodes"]
    columns = ["Steps", "Hidden", "Heads", "Depth"] + threshold_labels + [
        "Final Draw",
        "Final Lose",
        "Pretrain Policy Loss",
        "Pretrain Value Loss",
        "Pretrain Acc",
    ]
    widths = [10, 8, 7, 7] + [24 for _ in threshold_labels] + [11, 11, 19, 18, 13]
    header = "  ".join(f"{col:>{width}}" for col, width in zip(columns, widths))
    print("\nPretraining sweep summary:")
    print(header)
    print("-" * len(header))

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(row["pretrain_steps"], []).append(row)

    def _format_episode_stats(values: List[Optional[int]]) -> str:
        valid = [v for v in values if v is not None]
        if not valid:
            return "--"
        if len(valid) == 1:
            return f"{valid[0]:.0f}"
        return f"{mean(valid):.0f}±{stdev(valid):.0f}"

    def _format_metric(values: List[Optional[float]]) -> str:
        valid = [v for v in values if v is not None]
        if not valid:
            return "--"
        if len(valid) == 1:
            return f"{valid[0]:.3f}"
        return f"{mean(valid):.3f}±{stdev(valid):.3f}"

    for steps in sorted(grouped.keys()):
        rows = grouped[steps]
        line_parts = [f"{steps:>10}"]
        line_parts.append(f"{rows[0].get('model_hidden_dim', '--'):>8}")
        line_parts.append(f"{rows[0].get('model_heads', '--'):>7}")
        line_parts.append(f"{rows[0].get('model_depth', '--'):>7}")
        if thresholds:
            for thr in thresholds:
                values = [row["threshold_hits"].get(thr) if row.get("threshold_hits") else None for row in rows]
                line_parts.append(f"{_format_episode_stats(values):>24}")
        line_parts.append(f"{_format_metric([row.get('final_minimax_draw') for row in rows]):>11}")
        line_parts.append(f"{_format_metric([row.get('final_minimax_lose') for row in rows]):>11}")
        line_parts.append(f"{_format_metric([row.get('pretrain_policy_loss') for row in rows]):>19}")
        line_parts.append(f"{_format_metric([row.get('pretrain_value_loss') for row in rows]):>18}")
        line_parts.append(f"{_format_metric([row.get('pretrain_accuracy') for row in rows]):>13}")
        print("  ".join(line_parts))


def _write_metrics_summary(results: List[Dict[str, Any]], thresholds: List[float], metrics_dir: Optional[Path]) -> None:
    if not metrics_dir:
        return
    metrics_dir.mkdir(parents=True, exist_ok=True)

    sorted_thresholds = sorted(thresholds) if thresholds else []
    if sorted_thresholds:
        summary_path = metrics_dir / "episodes_to_target_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "run_name",
                    "pretrain_steps",
                    "seed",
                    "model_hidden_dim",
                    "model_heads",
                    "model_depth",
                    "threshold",
                    "episodes_to_target",
                    "pretrain_policy_loss",
                    "pretrain_value_loss",
                    "pretrain_accuracy",
                ]
            )
            for row in results:
                hits = row.get("threshold_hits") or {}
                for thr in sorted_thresholds:
                    hit = hits.get(thr)
                    writer.writerow(
                        [
                            row.get("run_name"),
                            row.get("pretrain_steps"),
                            row.get("seed"),
                            row.get("model_hidden_dim"),
                            row.get("model_heads"),
                            row.get("model_depth"),
                            thr,
                            "" if hit is None else hit,
                            row.get("pretrain_policy_loss"),
                            row.get("pretrain_value_loss"),
                            row.get("pretrain_accuracy"),
                        ]
                    )

    if any(row.get("minimax_history") for row in results):
        minimax_path = metrics_dir / "minimax_history_summary.csv"
        with minimax_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "run_name",
                "pretrain_steps",
                "seed",
                "model_hidden_dim",
                "model_heads",
                "model_depth",
                "episode",
                "win_rate",
                "draw_rate",
            ])
            for row in results:
                for episode, win_rate, draw_rate in row.get("minimax_history", []):
                    writer.writerow(
                        [
                            row.get("run_name"),
                            row.get("pretrain_steps"),
                            row.get("seed"),
                            row.get("model_hidden_dim"),
                            row.get("model_heads"),
                            row.get("model_depth"),
                            episode,
                            win_rate,
                            draw_rate,
                        ]
                    )

    if any(row.get("eval_history") for row in results):
        eval_path = metrics_dir / "evaluation_history_summary.csv"
        with eval_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "run_name",
                "pretrain_steps",
                "seed",
                "model_hidden_dim",
                "model_heads",
                "model_depth",
                "episode",
                "win_rate",
                "draw_rate",
            ])
            for row in results:
                for episode, win_rate, draw_rate in row.get("eval_history", []):
                    writer.writerow(
                        [
                            row.get("run_name"),
                            row.get("pretrain_steps"),
                            row.get("seed"),
                            row.get("model_hidden_dim"),
                            row.get("model_heads"),
                            row.get("model_depth"),
                            episode,
                            win_rate,
                            draw_rate,
                        ]
                    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = _resolve_config(config_path)
    run_timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    run_token = f"run_{run_timestamp}"
    metrics_dir_cfg = config.get("metrics_output_dir")
    candidate_metrics_dir = metrics_dir_cfg if metrics_dir_cfg is not None else args.metrics_dir
    metrics_dir: Optional[Path]
    if candidate_metrics_dir:
        metrics_base_dir = Path(candidate_metrics_dir).expanduser().resolve()
        metrics_base_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = metrics_base_dir / run_token
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = metrics_dir.resolve()
    else:
        metrics_dir = None

    experiments_cfg = config.get("experiments", {})
    default_seeds: Optional[List[int]]
    seeds_cfg = experiments_cfg.get("seeds")
    seed_count_cfg = experiments_cfg.get("seed_count")
    if seeds_cfg is None:
        count = int(seed_count_cfg) if seed_count_cfg is not None else 5
        default_seeds = list(range(count))
    elif isinstance(seeds_cfg, int):
        default_seeds = [int(seeds_cfg)]
    else:
        default_seeds = [int(s) for s in seeds_cfg]
    name_prefix = experiments_cfg.get("run_name_prefix", "")
    name_suffix = experiments_cfg.get("run_name_suffix", "")
    runs: List[ExperimentRun] = []
    for run_cfg in experiments_cfg.get("runs", []):
        run_name = f"{name_prefix}{run_cfg['name']}{name_suffix}"
        run_seeds_cfg = run_cfg.get("seeds")
        run_seed_count_cfg = run_cfg.get("seed_count")
        if run_seeds_cfg is None:
            if run_seed_count_cfg is not None:
                run_seeds = list(range(int(run_seed_count_cfg)))
            else:
                run_seeds = list(default_seeds)
        elif isinstance(run_seeds_cfg, int):
            run_seeds = [int(run_seeds_cfg)]
        else:
            run_seeds = [int(s) for s in run_seeds_cfg]
        run_tags = run_cfg.get("tags")
        if isinstance(run_tags, str):
            run_tags = [run_tags]
        runs.append(
            ExperimentRun(
                name=run_name,
                pretrain_steps=int(run_cfg["pretrain_steps"]),
                seeds=run_seeds,
                tags=run_tags,
            )
        )
    if not runs:
        raise ValueError("No experiment runs specified in configuration")

    wandb_run_cfg = config.get("wandb", {}).copy()
    if wandb_run_cfg and wandb_run_cfg.get("group") is None:
        wandb_run_cfg["group"] = f"pretrain-{run_timestamp}"
    run_tag_prefix = wandb_run_cfg.get("run_tag_prefix")
    if run_tag_prefix is not None:
        wandb_run_cfg.pop("run_tag_prefix", None)
    global_tags = wandb_run_cfg.get("tags") or []
    if isinstance(global_tags, str):
        global_tags = [global_tags]
    timestamp_tag = wandb_run_cfg["group"]
    if timestamp_tag not in global_tags:
        global_tags.append(timestamp_tag)
    wandb_run_cfg["tags"] = global_tags
    base_output_dir = Path(config.get("output_root", "outputs/experiments")).expanduser()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    base_output_dir = (base_output_dir / run_token)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    base_output_dir = base_output_dir.resolve()
    print(f"[paths] PPO outputs root: {base_output_dir}")
    if metrics_dir is not None:
        print(f"[paths] Metrics root: {metrics_dir}")

    env = TicTacToeEnv()
    obs_dim = env.reset().numel()

    target_metrics = config.get("target_metrics", {})
    thresholds_cfg = target_metrics.get("thresholds")
    if thresholds_cfg is None:
        single = target_metrics.get("minimax_draw_rate")
        thresholds = [float(single)] if single is not None else []
    else:
        thresholds = [float(x) for x in thresholds_cfg]
    consecutive_required = int(target_metrics.get("consecutive", 1))

    ppo_base_cfg = config.get("ppo", {})
    tuning_cfg = config.get("ppo_tuning", {})
    pretrained_cfg = config.get("pretrained", {})
    checkpoint_dir = Path(pretrained_cfg.get("directory", "checkpoints"))
    checkpoint_pattern = pretrained_cfg.get("file_pattern", "checkpoint_step_{step:04d}.pt")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Pretrained checkpoint directory not found: {checkpoint_dir}")

    tasks: List[tuple] = []
    for run_cfg in runs:
        for seed in run_cfg.seeds:
            run_specific_tags = list(global_tags)
            if run_tag_prefix:
                run_specific_tags.append(f"{run_tag_prefix}-{run_cfg.name}")
            if run_cfg.tags:
                run_specific_tags.extend(run_cfg.tags)
            if run_specific_tags:
                # preserve order while removing duplicates
                seen = set()
                deduped_tags = []
                for tag in run_specific_tags:
                    if tag not in seen:
                        deduped_tags.append(tag)
                        seen.add(tag)
                run_specific_tags = deduped_tags
            tasks.append(
                (
                    run_cfg.name,
                    run_cfg.pretrain_steps,
                    seed,
                    str(checkpoint_dir),
                    checkpoint_pattern,
                    ppo_base_cfg,
                    {**wandb_run_cfg, "tags": run_specific_tags},
                    str(base_output_dir),
                    thresholds,
                    consecutive_required,
                    tuning_cfg,
                    args.disable_wandb,
                    obs_dim,
                    str(metrics_dir) if metrics_dir is not None else None,
                )
            )

    if not tasks:
        print("No experiment tasks to run.")
        return

    results: List[Dict[str, Any]] = []
    max_workers = max(1, args.max_workers)
    if max_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(_execute_experiment, *task): task for task in tasks}
            for future in as_completed(future_to_task):
                results.append(future.result())
    else:
        for task in tasks:
            results.append(_execute_experiment(*task))

    results.sort(key=lambda r: (r["pretrain_steps"], r.get("seed", 0)))
    _print_summary(results, thresholds)
    _write_metrics_summary(results, thresholds, metrics_dir)


if __name__ == "__main__":
    main()
