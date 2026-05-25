#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -z "${TINKER_API_KEY:-}" ]] && { echo "TINKER_API_KEY not set" >&2; exit 1; }

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
RENDERER=${RENDERER:-qwen2_5_instruct}
TEMPERATURE=${TEMPERATURE:-0.7}
N_SAMPLES=${N_SAMPLES:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
CONCURRENCY=${CONCURRENCY:-16}

python src/generate_y0.py \
  --model "$MODEL" \
  --renderer "$RENDERER" \
  --input data/processed/train.jsonl \
  --output data/rollouts/y0.jsonl \
  --temperature "$TEMPERATURE" \
  --n-samples "$N_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --concurrency "$CONCURRENCY"

python src/grade.py \
  --input data/rollouts/y0.jsonl \
  --output data/rollouts/y0_graded.jsonl
