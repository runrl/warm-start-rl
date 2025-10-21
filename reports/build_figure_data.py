#!/usr/bin/env python3
"""
Utility script to extract figure-ready datasets from experiment CSV logs.

Writes compact CSV files into reports/data/ that can be consumed directly by
PGFPlots in the LaTeX paper.
"""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT
DATA_DIR = REPORTS_DIR / "data"

SIZES = ["half", "1x", "2x"]
DISPLAY_LABELS = {"half": "small", "1x": "medium", "2x": "big"}
PARAM_COUNTS = {"half": 451402, "1x": 798474, "2x": 1787530}
THRESHOLDS = [0.97, 0.99]


def load_episode_data(size: str) -> list[dict[str, str]]:
    path = REPORTS_DIR / f"{size}_episodes_to_target.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_draw_rates(size: str) -> dict[int, float]:
    """Return final recorded draw rate per pretraining schedule."""
    path = REPORTS_DIR / f"{size}_draw_rate_raw.csv"
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]

    result: dict[int, float] = {}
    for idx, col in enumerate(header[1:], start=1):
        if "__" in col:
            continue
        # Column format: pretrain_{steps}_{size}-seedX - minimax_draw_rate
        pieces = col.split("_")
        try:
            pretrain_steps = int(pieces[1])
        except (IndexError, ValueError):
            continue

        value = None
        for row in reversed(rows):
            if idx < len(row) and row[idx]:
                value = float(row[idx])
                break
        if value is not None:
            result[pretrain_steps] = value
    return dict(sorted(result.items()))


def write_threshold_series(size: str, threshold: float, records: list[dict[str, str]]) -> None:
    output = DATA_DIR / f"{size}_episodes_threshold_{int(threshold * 100):02d}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["pretrain_steps", "episodes_to_target"])
        seen = []
        for row in records:
            if float(row["threshold"]) != threshold:
                continue
            if not row["episodes_to_target"]:
                continue
            pretrain = int(row["pretrain_steps"])
            episodes = int(row["episodes_to_target"])
            seen.append((pretrain, episodes))
        for pretrain, episodes in sorted(seen):
            writer.writerow([pretrain, episodes])


def write_draw_rate_series(size: str, draw_rates: dict[int, float]) -> None:
    output = DATA_DIR / f"{size}_final_draw_rate.csv"
    with output.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["pretrain_steps", "draw_rate"])
        for pretrain, value in draw_rates.items():
            writer.writerow([pretrain, value])


def write_baseline_table(episodes: dict[str, list[dict[str, str]]]) -> None:
    """Capture sample efficiency with and without pretraining across sizes."""
    output = DATA_DIR / "baseline_summary.csv"
    with output.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "params", "baseline_ep_0.97", "best_ep_0.97", "best_pretrain_0.97"])
        for size, rows in episodes.items():
            baseline = None
            best = None
            best_pre = None
            for row in rows:
                if float(row["threshold"]) != 0.97:
                    continue
                if not row["episodes_to_target"]:
                    continue
                pretrain = int(row["pretrain_steps"])
                episodes_needed = int(row["episodes_to_target"])
                if pretrain == 0:
                    baseline = episodes_needed
                if best is None or episodes_needed < best:
                    best = episodes_needed
                    best_pre = pretrain
            writer.writerow([DISPLAY_LABELS.get(size, size), PARAM_COUNTS.get(size, ""), baseline, best, best_pre])


def main() -> None:
    episodes_by_size: dict[str, list[dict[str, str]]] = {}
    for size in SIZES:
        records = load_episode_data(size)
        episodes_by_size[size] = records
        for threshold in THRESHOLDS:
            write_threshold_series(size, threshold, records)
        draw_rates = load_draw_rates(size)
        write_draw_rate_series(size, draw_rates)
    write_baseline_table(episodes_by_size)


if __name__ == "__main__":
    main()
