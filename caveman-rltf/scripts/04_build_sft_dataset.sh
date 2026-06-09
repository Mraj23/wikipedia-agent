#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LENGTH_RATIO=${LENGTH_RATIO:-0.8}

# Main arm: correct + shorter y1, trained as x0 -> y1. Length uses the exact
# generated token count recorded at sampling time (no tokenizer needed).
python src/build_sft_dataset.py \
  --mode rltf_sft \
  --y1-graded data/revised/y1_graded.jsonl \
  --y0-graded data/rollouts/y0_graded.jsonl \
  --output data/train/rltf_sft.jsonl \
  --length-ratio "$LENGTH_RATIO"
