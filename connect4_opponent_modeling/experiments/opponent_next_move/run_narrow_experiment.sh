#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [ -d "$ROOT_DIR/.venv" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

if [ -f "$ROOT_DIR/scripts/gpu_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/gpu_env.sh"
fi

EXP_DIR="experiments/opponent_next_move"
DATA_PATH="$EXP_DIR/data/connect4_eval_banks.jsonl"
RESULTS_DIR="$EXP_DIR/results"
LOG_DIR="$EXP_DIR/logs"
CKPT_DIR="$EXP_DIR/checkpoints"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
SEEDS="${SEEDS:-42 43 44}"
GAME_STEPS="${GAME_STEPS:-500}"
GROUP_SIZE="${GROUP_SIZE:-64}"
EVAL_EVERY="${EVAL_EVERY:-250}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_EVAL_PER_SPLIT="${MAX_EVAL_PER_SPLIT:-}"
USE_VLLM="${USE_VLLM:-0}"
WANDB="${WANDB:-0}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$CKPT_DIR"

if [ ! -f "$DATA_PATH" ]; then
  python "$EXP_DIR/make_position_banks.py" --output "$DATA_PATH"
else
  echo "Using locked eval banks at $DATA_PATH"
fi

EVAL_LIMIT_ARGS=()
if [ -n "$MAX_EVAL_PER_SPLIT" ]; then
  EVAL_LIMIT_ARGS=(--max_positions_per_split "$MAX_EVAL_PER_SPLIT")
fi

echo "=== BaseScaffold eval: $MODEL ==="
python "$EXP_DIR/eval_move_quality.py" \
  --model "$MODEL" \
  --condition BaseScaffold \
  --model_label BaseScaffold \
  --banks "$DATA_PATH" \
  --output "$RESULTS_DIR/base_scaffold.json" \
  --max_new_tokens "$MAX_TOKENS" \
  "${EVAL_LIMIT_ARGS[@]}"

for CONDITION in Value OpponentNextMove; do
  for SEED in $SEEDS; do
    RUN_NAME="${CONDITION}_seed${SEED}"
    RUN_LOG_DIR="$LOG_DIR/$RUN_NAME"
    RUN_CKPT_DIR="$CKPT_DIR/$RUN_NAME"

    TRAIN_ARGS=(
      -m spiral.train
      --condition "$CONDITION"
      --model "$MODEL"
      --log_dir "$RUN_LOG_DIR"
      --checkpoint_dir "$RUN_CKPT_DIR"
      --game_steps "$GAME_STEPS"
      --group_size "$GROUP_SIZE"
      --eval_every "$EVAL_EVERY"
      --max_tokens "$MAX_TOKENS"
      --seed "$SEED"
      --wandb_run_name "$RUN_NAME"
    )
    if [ "$USE_VLLM" = "1" ]; then
      TRAIN_ARGS+=(--use_vllm)
    fi
    if [ "$WANDB" = "1" ]; then
      TRAIN_ARGS+=(--wandb)
    fi

    echo "=== Training $RUN_NAME ==="
    python "${TRAIN_ARGS[@]}"

    echo "=== Evaluating $RUN_NAME ==="
    python "$EXP_DIR/eval_move_quality.py" \
      --model "$RUN_CKPT_DIR/final" \
      --condition "$CONDITION" \
      --model_label "$RUN_NAME" \
      --banks "$DATA_PATH" \
      --output "$RESULTS_DIR/${RUN_NAME}.json" \
      --max_new_tokens "$MAX_TOKENS" \
      "${EVAL_LIMIT_ARGS[@]}"
  done
done

python "$EXP_DIR/summarize_results.py" \
  --results_dir "$RESULTS_DIR" \
  --output "$RESULTS_DIR/summary.json"

echo "=== Narrow experiment complete ==="
echo "Results: $RESULTS_DIR"
