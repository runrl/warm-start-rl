#!/bin/bash
uv run python scripts/build_tictactoe_dataset.py --output data/minimax_dataset.pt --num-games 20000 --seed 42
uv run python scripts/pretrain_tictactoe_supervised.py \
  --dataset data/minimax_dataset.pt \
  --output-dir checkpoints/pretraining_run \
  --total-steps 50 \
  --save-interval 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --shuffle
uv run python scripts/run_pretraining_experiments.py --config configs/pretraining_experiment.yaml --max-workers 1 "$@"
