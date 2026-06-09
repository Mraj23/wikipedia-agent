#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Defaults come from configs/tasks.yaml. Override per-task sizes via env, e.g.
#   TRAIN_N=150 EVAL_N=100 scripts/00_prepare_data.sh
# Restrict tasks via TASKS, e.g. TASKS="tracking_shuffled_objects gsm8k".
ARGS=(--config configs/tasks.yaml --out data/processed)
[[ -n "${TRAIN_N:-}" ]] && ARGS+=(--train-n "$TRAIN_N")
[[ -n "${EVAL_N:-}" ]] && ARGS+=(--eval-n "$EVAL_N")
[[ -n "${SEED:-}" ]] && ARGS+=(--seed "$SEED")
[[ -n "${TASKS:-}" ]] && ARGS+=(--tasks $TASKS)

python src/load_bbh.py "${ARGS[@]}"
