#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_N=${TRAIN_N:-1000}
EVAL_N=${EVAL_N:-250}

python src/load_bbh.py \
  --config configs/tasks.yaml \
  --train-n "$TRAIN_N" \
  --eval-n "$EVAL_N" \
  --out data/processed
