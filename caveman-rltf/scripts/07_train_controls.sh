#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Control / ablation arms. Run AFTER 01-03 have produced y0_graded and
# y1_graded. Each trained checkpoint can then be added to RUN_NAMES in 06.
[[ -z "${TINKER_API_KEY:-}" ]] && { echo "TINKER_API_KEY not set" >&2; exit 1; }

MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
RENDERER=${RENDERER:-qwen3}

# --- sft_caveman: model's own shortest correct first attempt ---
python src/build_sft_dataset.py --mode sft_caveman \
  --y0-graded data/rollouts/y0_graded.jsonl \
  --output data/train/sft_caveman.jsonl
TRAIN_FILE=data/train/sft_caveman.jsonl RUN_NAME=sft_caveman scripts/05_train_rltf_sft.sh

# --- sft_y1: correct y1, no length gate (isolates the length filter) ---
python src/build_sft_dataset.py --mode sft_y1 \
  --y1-graded data/revised/y1_graded.jsonl \
  --output data/train/sft_y1.jsonl
TRAIN_FILE=data/train/sft_y1.jsonl RUN_NAME=sft_y1 scripts/05_train_rltf_sft.sh

# --- grpo_length: RL control that directly rewards shortness-if-correct ---
ALPHA=${ALPHA:-0.1}
TOKEN_BUDGET=${TOKEN_BUDGET:-256}
python src/train_grpo_length.py \
  --model "$MODEL" --renderer "$RENDERER" \
  --train data/processed/train.jsonl \
  --output outputs/checkpoints/grpo_length \
  --run-name grpo_length \
  --alpha "$ALPHA" --token-budget "$TOKEN_BUDGET"

echo "controls trained. Add to eval with:"
echo "  RUN_NAMES='rltf_sft sft_caveman sft_y1 grpo_length' scripts/06_eval.sh"
