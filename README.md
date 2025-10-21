# Warm start RL

[Blog post](https://runrl.com/blog/warm-start-rl)

```bash
uv sync

uv run python scripts/build_tictactoe_dataset.py \
  --output data/minimax_dataset.pt \
  --num-games 20000 \
  --seed 42

# Supervised warm starts for each model size (checkpoints feed the PPO sweeps below).
# These create distinct folders in checkpoints/pretraining_run_{half,1x,double}.

PYTHONPATH=src uv run python scripts/pretrain_tictactoe_supervised.py \
  --dataset data/minimax_dataset.pt \
  --output-dir checkpoints/pretraining_run_half \
  --model-hidden-dim 96 \
  --model-heads 3 \
  --model-depth 2 \
  --total-steps 100 \
  --save-interval 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --shuffle

PYTHONPATH=src uv run python scripts/pretrain_tictactoe_supervised.py \
  --dataset data/minimax_dataset.pt \
  --output-dir checkpoints/pretraining_run_1x \
  --model-hidden-dim 128 \
  --model-heads 4 \
  --model-depth 2 \
  --total-steps 100 \
  --save-interval 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --shuffle

PYTHONPATH=src uv run python scripts/pretrain_tictactoe_supervised.py \
  --dataset data/minimax_dataset.pt \
  --output-dir checkpoints/pretraining_run_double \
  --model-hidden-dim 192 \
  --model-heads 6 \
  --model-depth 2 \
  --total-steps 100 \
  --save-interval 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --shuffle

# Run PPO warm-start sweeps (5 seeds each) for every model size.
# Ensure src/ is on PYTHONPATH for package imports. Output directories are isolated per model size,
# so you can run these three sweeps in any order without files colliding.

PYTHONPATH=src uv run python scripts/run_pretraining_experiments.py \
  --config configs/pretraining_experiment_half.yaml \
  --max-workers 24

PYTHONPATH=src uv run python scripts/run_pretraining_experiments.py \
  --config configs/pretraining_experiment.yaml \
  --max-workers 24

PYTHONPATH=src uv run python scripts/run_pretraining_experiments.py \
  --config configs/pretraining_experiment_double.yaml \
  --max-workers 24
```

The default template sweeps pretraining checkpoints at steps 0, 20, 40, 60, 80, and 100 for each of the five seeds listed in the configs.

Each invocation writes to a fresh timestamped folder (for example `run_20240512-153045-123456`) under both `outputs/pretraining_quick/<model>/` and `reports/local_metrics/<model>/`, so repeat sweeps never collide.

Per-seed metrics land in `reports/local_metrics/<model>/run_<timestamp>/seed_<n>/`, and aggregate CSVs sit beside them for downstream plotting. PPO artifacts are written under `outputs/pretraining_quick/<model>/run_<timestamp>/<run_name>/seed_<n>/`.

Both the supervised and PPO scripts automatically select the best available device (preferring Apple MPS when present, otherwise CUDA, otherwise CPU), so no extra flags are required.

Episode logs (and the corresponding metrics payloads) now include profiling durations for data collection, PPO updates, and evaluation, making it easier to pinpoint slow phases.

Pass `--profile-minimax` (or set `ppo.profile_minimax: true` in a config) to add detailed minimax timing stats, including cache hits and recursive search totals, to the console and metric streams.
