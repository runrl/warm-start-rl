#!/usr/bin/env python3
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"

SIZE_CONFIGS = [
    {
        "csv": ROOT / "half_episodes_to_target.csv",
        "label": "Small (451k)",
        "color": "#1f77b4",
        "marker": "o",
    },
    {
        "csv": ROOT / "1x_episodes_to_target.csv",
        "label": "Medium (798k)",
        "color": "#ff7f0e",
        "marker": "s",
    },
    {
        "csv": ROOT / "2x_episodes_to_target.csv",
        "label": "Big (1.79M)",
        "color": "#9467bd",
        "marker": "D",
    },
]


def load_median_series(csv_path: Path, threshold: float) -> list[tuple[int, float]]:
    by_steps: dict[int, list[float]] = {}
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("episodes_to_target"):
                continue
            try:
                if float(row["threshold"]) != threshold:
                    continue
                episodes = float(row["episodes_to_target"])
                pretrain = int(row["pretrain_steps"])
            except (TypeError, ValueError):
                continue
            by_steps.setdefault(pretrain, []).append(episodes)
    series = []
    for pretrain, values in by_steps.items():
        if values:
            series.append((pretrain, statistics.median(values)))
    return sorted(series)


def plot_threshold(threshold: float, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    step_values: set[int] = set()
    for cfg in SIZE_CONFIGS:
        series = load_median_series(cfg["csv"], threshold)
        if not series:
            continue
        steps, medians = zip(*series)
        step_values.update(steps)
        ax.plot(
            steps,
            medians,
            label=cfg["label"],
            color=cfg["color"],
            marker=cfg["marker"],
            linewidth=2.4,
            markersize=7,
        )
    if step_values:
        ax.set_xticks(sorted(step_values))
    ax.set_xlabel("Pretraining steps")
    ax.set_ylabel(f"Median PPO episodes to reach draw rate >= {threshold:.2f}")
    ax.set_title(f"PPO episodes vs. pretraining for draw rate >= {threshold:.2f}")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main() -> None:
    plot_threshold(0.97, "warm-start-episodes-0_97.png")
    plot_threshold(0.99, "warm-start-episodes-0_99.png")


if __name__ == "__main__":
    main()
