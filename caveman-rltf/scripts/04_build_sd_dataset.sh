#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TOKENIZER=${TOKENIZER:-Qwen/Qwen2.5-7B-Instruct}
LENGTH_RATIO=${LENGTH_RATIO:-0.8}

python src/build_rltf_sd_dataset.py \
  --input data/revised/y1_graded.jsonl \
  --y0-graded data/rollouts/y0_graded.jsonl \
  --output data/train/rltf_sd.jsonl \
  --tokenizer "$TOKENIZER" \
  --length-ratio "$LENGTH_RATIO"
