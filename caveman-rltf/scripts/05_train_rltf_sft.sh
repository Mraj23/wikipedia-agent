#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -z "${TINKER_API_KEY:-}" ]] && { echo "TINKER_API_KEY not set" >&2; exit 1; }

MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
RENDERER=${RENDERER:-qwen3}
LORA_RANK=${LORA_RANK:-16}
EPOCHS=${EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-5}
RUN_NAME=${RUN_NAME:-rltf_sft}
TRAIN_FILE=${TRAIN_FILE:-data/train/rltf_sft.jsonl}

python src/train_sft.py \
  --model "$MODEL" \
  --renderer "$RENDERER" \
  --train "$TRAIN_FILE" \
  --output outputs/checkpoints/"$RUN_NAME" \
  --run-name "$RUN_NAME" \
  --lora-rank "$LORA_RANK" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LR"
