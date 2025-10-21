from __future__ import annotations

import argparse
import itertools
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:  # pragma: no cover - optional dependency handling
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except ImportError:  # pragma: no cover - fallback when tqdm is missing
    _tqdm = None  # type: ignore


class _TqdmFallback:
    def __init__(self, iterable=None, **kwargs):
        if iterable is None:
            total = kwargs.get("total")
            iterable = range(total or 0)
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _make_progress(iterable=None, **kwargs):
    if _tqdm is not None:
        return _tqdm(iterable, **kwargs)
    return _TqdmFallback(iterable=iterable, **kwargs)


def _progress_write(message: str) -> None:
    if _tqdm is not None:
        _tqdm.write(message)
    else:
        print(message)

from c4_scaling.datasets import load_imitation_dataset
from c4_scaling.policy_gradient import PolicyNetwork, ValueNetwork
from c4_scaling.tictactoe_env import TicTacToeEnv


def _log(message: str) -> None:
    """Log while playing nicely with progress bars."""
    _progress_write(message)


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

    candidates = []
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


def _evaluate(
    policy: PolicyNetwork,
    value_function: ValueNetwork,
    dataloader: DataLoader,
    device: torch.device,
    *,
    show_progress: bool = False,
) -> Tuple[float, float, float]:
    policy.eval()
    value_function.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_correct = 0
    total_examples = 0
    iterator = dataloader
    progress_bar: Optional[object] = None
    if show_progress:
        progress_bar = _make_progress(dataloader, desc="Evaluating", unit="batch", leave=False)
        iterator = progress_bar
    with torch.no_grad():
        for obs, actions, values in iterator:
            obs = obs.to(device)
            actions = actions.to(device)
            values = values.to(device)
            logits = policy(obs)
            preds = torch.argmax(logits, dim=-1)
            policy_loss = F.cross_entropy(logits, actions, reduction="sum")
            value_preds = value_function(obs)
            value_loss = F.mse_loss(value_preds, values, reduction="sum")
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_correct += int((preds == actions).sum().item())
            total_examples += actions.size(0)
    if progress_bar is not None:
        progress_bar.close()
    avg_policy_loss = total_policy_loss / max(total_examples, 1)
    avg_value_loss = total_value_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return avg_policy_loss, avg_value_loss, accuracy


def _save_checkpoint(
    policy: PolicyNetwork,
    value_function: ValueNetwork,
    device: torch.device,
    eval_loader: DataLoader,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    *,
    show_progress: bool = False,
) -> Dict[str, float]:
    policy_loss, value_loss, acc = _evaluate(
        policy,
        value_function,
        eval_loader,
        device,
        show_progress=show_progress,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_step_{step:04d}.pt"
    args_dict: Dict[str, object] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    payload = {
        "step": step,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "accuracy": acc,
        "policy_state": policy.state_dict(),
        "value_state": value_function.state_dict(),
        "args": args_dict,
    }
    torch.save(payload, path)
    return {"policy_loss": policy_loss, "value_loss": value_loss, "accuracy": acc}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised pretraining for Tic-Tac-Toe policy.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to imitation dataset (.pt).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store checkpoints.")
    parser.add_argument("--total-steps", type=int, default=200, help="Total gradient steps to run.")
    parser.add_argument("--save-interval", type=int, default=20, help="Interval (in steps) between checkpoints.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for supervised updates.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset during training.")
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true", help="Force CUDA usage when available.")
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false", help="Disable CUDA even if available.")
    parser.add_argument("--use-mps", dest="use_mps", action="store_true", help="Force Apple MPS usage when available.")
    parser.add_argument("--no-mps", dest="use_mps", action="store_false", help="Disable Apple MPS even if available.")
    parser.set_defaults(use_cuda=None, use_mps=None)
    parser.add_argument("--model-hidden-dim", type=int, default=128, help="Transformer hidden dimension.")
    parser.add_argument("--model-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--model-depth", type=int, default=2, help="Number of decoder blocks.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = _select_device(args.use_cuda, args.use_mps)
    _log(f"Using device: {device}")

    dataset = load_imitation_dataset(args.dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=False,
    )
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    env = TicTacToeEnv()
    obs_dim = env.reset().numel()
    policy = PolicyNetwork(
        obs_dim=obs_dim,
        hidden_dim=args.model_hidden_dim,
        n_head=args.model_heads,
        depth=args.model_depth,
    ).to(device)

    value_function = ValueNetwork(
        obs_dim=obs_dim,
        hidden_dim=args.model_hidden_dim,
        n_head=args.model_heads,
        depth=args.model_depth,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_function.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    metrics = _save_checkpoint(
        policy,
        value_function,
        device,
        eval_loader,
        args.output_dir,
        step=0,
        args=args,
        show_progress=True,
    )
    _log(
        f"Saved checkpoint at step 0 | "
        f"policy_loss={metrics['policy_loss']:.4f}, value_loss={metrics['value_loss']:.4f}, acc={metrics['accuracy']:.4f}"
    )

    loader_iter = itertools.cycle(train_loader)
    policy.train()
    value_function.train()
    progress_bar = _make_progress(
        range(1, args.total_steps + 1),
        desc="Training",
        unit="step",
        total=args.total_steps,
    )
    for step in progress_bar:
        obs, actions, values = next(loader_iter)
        obs = obs.to(device)
        actions = actions.to(device)
        values = values.to(device)

        logits = policy(obs)
        policy_loss = F.cross_entropy(logits, actions)
        value_preds = value_function(obs)
        value_loss = F.mse_loss(value_preds, values)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(
            {
                "policy_loss": f"{policy_loss.item():.4f}",
                "value_loss": f"{value_loss.item():.4f}",
            },
            refresh=False,
        )

        if step % args.save_interval == 0 or step == args.total_steps:
            metrics = _save_checkpoint(
                policy,
                value_function,
                device,
                eval_loader,
                args.output_dir,
                step=step,
                args=args,
                show_progress=True,
            )
            _log(
                f"Saved checkpoint at step {step} | "
                f"policy_loss={metrics['policy_loss']:.4f}, value_loss={metrics['value_loss']:.4f}, "
                f"acc={metrics['accuracy']:.4f}"
            )

    if hasattr(progress_bar, "close"):
        progress_bar.close()


if __name__ == "__main__":
    main()
