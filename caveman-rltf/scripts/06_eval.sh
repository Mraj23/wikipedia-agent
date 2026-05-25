#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -z "${TINKER_API_KEY:-}" ]] && { echo "TINKER_API_KEY not set" >&2; exit 1; }

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
RENDERER=${RENDERER:-qwen2_5_instruct}
RUN_NAME=${RUN_NAME:-rltf_sd}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
CONCURRENCY=${CONCURRENCY:-16}

# Base eval (no checkpoint)
python src/eval.py \
  --model "$MODEL" --renderer "$RENDERER" \
  --eval-data data/processed/eval.jsonl \
  --output outputs/evals/base.jsonl \
  --condition base \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --concurrency "$CONCURRENCY"

# Trained checkpoint eval
python src/eval.py \
  --model "$MODEL" --renderer "$RENDERER" \
  --manifest outputs/checkpoints/"$RUN_NAME"/manifest.json \
  --eval-data data/processed/eval.jsonl \
  --output outputs/evals/"$RUN_NAME".jsonl \
  --condition "$RUN_NAME" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --concurrency "$CONCURRENCY"

python src/analyze.py \
  --input outputs/evals \
  --output outputs/evals

python src/plot.py \
  --input outputs/evals/summary.csv \
  --output-dir outputs/plots
