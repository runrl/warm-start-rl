#!/usr/bin/env python3
"""
Compute log-quadratic fits between pretraining policy loss and PPO episodes.

Reads the per-run CSV summaries produced in reports/local_metrics/* and writes a
compact table of the fitted coefficients and R^2 scores to
reports/log_quadratic_fits.csv. The fit has the form

    log(E) = a * (log L)^2 + b * log L + c

where E is the number of PPO episodes required to hit the target draw rate and
L is the pretraining policy loss.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPORTS_DIR = Path(__file__).resolve().parent
LOCAL_METRICS_DIR = REPORTS_DIR / "local_metrics"
OUTPUT_PATH = REPORTS_DIR / "log_quadratic_fits.csv"
THRESHOLD = 0.97

# Directory name -> human-readable label.
SIZES = {
    "half": "0.5×",
    "1x": "1×",
    "double": "2×",
}


@dataclass
class FitResult:
    label: str
    a: float
    b: float
    c: float
    r_squared: float
    count: int


def iter_summary_paths(size: str) -> list[Path]:
    """Return all summary CSV paths for a given model size."""
    base = LOCAL_METRICS_DIR / size
    if not base.exists():
        return []
    return sorted(base.glob("*/episodes_to_target_summary.csv"))


def load_points(paths: list[Path]) -> list[tuple[float, float]]:
    """Load (loss, episodes) pairs at the desired threshold from the CSV logs."""
    points: list[tuple[float, float]] = []
    for path in paths:
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("episodes_to_target"):
                    continue
                try:
                    threshold = float(row["threshold"])
                except (TypeError, ValueError):
                    continue
                if threshold != THRESHOLD:
                    continue
                try:
                    loss = float(row["pretrain_policy_loss"])
                    episodes = float(row["episodes_to_target"])
                except (TypeError, ValueError):
                    continue
                if loss <= 0 or episodes <= 0:
                    continue
                points.append((loss, episodes))
    return points


def fit_log_quadratic(points: list[tuple[float, float]], label: str) -> FitResult | None:
    """Perform the least-squares log-quadratic regression."""
    if not points:
        return None
    log_losses = []
    log_episodes = []
    for loss, episodes in points:
        log_losses.append(math.log(loss))
        log_episodes.append(math.log(episodes))
    X = np.column_stack(
        [
            np.square(log_losses),
            log_losses,
            np.ones(len(log_losses)),
        ]
    )
    y = np.asarray(log_episodes)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    ss_res = float(np.sum(np.square(y - y_pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return FitResult(label, coeffs[0], coeffs[1], coeffs[2], r_squared, len(points))


def write_results(results: list[FitResult]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "coefficient_a",
                "coefficient_b",
                "coefficient_c",
                "r_squared",
                "count",
                "threshold",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.label,
                    f"{result.a:.9f}",
                    f"{result.b:.9f}",
                    f"{result.c:.9f}",
                    f"{result.r_squared:.6f}",
                    result.count,
                    THRESHOLD,
                ]
            )


def main() -> None:
    per_size_results: list[FitResult] = []
    aggregate_points: list[tuple[float, float]] = []

    for size_dir, label in SIZES.items():
        paths = iter_summary_paths(size_dir)
        points = load_points(paths)
        if points:
            aggregate_points.extend(points)
            result = fit_log_quadratic(points, label)
            if result:
                per_size_results.append(result)

    aggregate_result = fit_log_quadratic(aggregate_points, "All sizes")
    results: list[FitResult] = []
    if aggregate_result:
        results.append(aggregate_result)
    results.extend(per_size_results)
    write_results(results)


if __name__ == "__main__":
    main()
