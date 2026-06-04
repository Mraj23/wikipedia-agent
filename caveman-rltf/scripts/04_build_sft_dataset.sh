#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TOKENIZER=${TOKENIZER:-Qwen/Qwen2.5-7B-Instruct}
LENGTH_RATIO=${LENGTH_RATIO:-0.8}

# Main arm: correct + shorter y1, trained as x0 -> y1.
python src/build_sft_dataset.py \
  --mode rltf_sft \
  --y1-graded data/revised/y1_graded.jsonl \
  --y0-graded data/rollouts/y0_graded.jsonl \
  --output data/train/rltf_sft.jsonl \
  --tokenizer "$TOKENIZER" \
  --length-ratio "$LENGTH_RATIO"
