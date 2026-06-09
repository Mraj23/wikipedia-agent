#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -z "${TINKER_API_KEY:-}" ]] && { echo "TINKER_API_KEY not set" >&2; exit 1; }

MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
RENDERER=${RENDERER:-qwen3}
TEMPERATURE=${TEMPERATURE:-0.7}
N_SAMPLES=${N_SAMPLES:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
CONCURRENCY=${CONCURRENCY:-16}

python src/generate_y1.py \
  --model "$MODEL" \
  --renderer "$RENDERER" \
  --input data/feedback/x1.jsonl \
  --output data/revised/y1.jsonl \
  --temperature "$TEMPERATURE" \
  --n-samples "$N_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --concurrency "$CONCURRENCY"

python src/grade.py \
  --input data/revised/y1.jsonl \
  --output data/revised/y1_graded.jsonl
