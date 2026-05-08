#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [ -d "$ROOT_DIR/.venv" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

export C4_REQUIRE_PONS_SOLVER="${C4_REQUIRE_PONS_SOLVER:-1}"

EXP_DIR="experiments/opponent_next_move"
DATA_PATH="${DATA_PATH:-$EXP_DIR/data/connect4_eval_banks.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_DIR/tinker_results}"
LOG_DIR="${LOG_DIR:-$EXP_DIR/tinker_logs}"

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
TINKER_RENDERER="${TINKER_RENDERER:-qwen3}"
TINKER_PROJECT_ID="${TINKER_PROJECT_ID:-}"
POSITION_BUFFER="${POSITION_BUFFER:-data/position_buffer.json}"
SEEDS="${SEEDS:-42}"
RL_STEPS="${RL_STEPS:-100}"
SFT_STEPS="${SFT_STEPS:-100}"
POSITIONS_PER_STEP="${POSITIONS_PER_STEP:-1}"
GROUP_SIZE="${GROUP_SIZE:-32}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
LORA_RANK="${LORA_RANK:-32}"
LOSS_FN="${LOSS_FN:-importance_sampling}"
MAX_EVAL_PER_SPLIT="${MAX_EVAL_PER_SPLIT:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
RUN_SFT="${RUN_SFT:-0}"
BASE_CONDITIONS="${BASE_CONDITIONS-BaseSimple BaseScaffold}"
TRAIN_CONDITIONS="${TRAIN_CONDITIONS-Value OpponentNextMove}"
WANDB="${WANDB:-0}"
VERIFY_SETUP="${VERIFY_SETUP:-1}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

if [ "$VERIFY_SETUP" = "1" ]; then
  python - <<'PY'
import os
import sys

if not os.environ.get("TINKER_API_KEY"):
    raise SystemExit("TINKER_API_KEY is not set.")

missing = []
for pkg in ("tinker", "tinker_cookbook"):
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    raise SystemExit(
        "Missing optional Tinker dependencies: "
        + ", ".join(missing)
        + "\nInstall with: pip install tinker tinker-cookbook"
    )
PY
fi

if [ ! -f "$DATA_PATH" ]; then
  python "$EXP_DIR/make_position_banks.py" --output "$DATA_PATH"
else
  echo "Using locked eval banks at $DATA_PATH"
fi

EVAL_LIMIT_ARGS=()
if [ -n "$MAX_EVAL_PER_SPLIT" ]; then
  EVAL_LIMIT_ARGS=(--max_positions_per_split "$MAX_EVAL_PER_SPLIT")
fi
PROJECT_ARGS=()
if [ -n "$TINKER_PROJECT_ID" ]; then
  PROJECT_ARGS=(--project_id "$TINKER_PROJECT_ID")
fi

echo "=== Tinker base evals: $MODEL ==="
for BASE_CONDITION in $BASE_CONDITIONS; do
  python "$EXP_DIR/tinker_eval_move_quality.py" \
    --model "$MODEL" \
    --condition "$BASE_CONDITION" \
    --model_label "Tinker${BASE_CONDITION}" \
    --banks "$DATA_PATH" \
    --output "$RESULTS_DIR/base_${BASE_CONDITION}.json" \
    --renderer "$TINKER_RENDERER" \
    --max_new_tokens "$EVAL_MAX_TOKENS" \
    --temperature 0.0 \
    --batch_size "$EVAL_BATCH_SIZE" \
    ${PROJECT_ARGS[@]-} \
    ${EVAL_LIMIT_ARGS[@]-}
done

if [ "$RUN_SFT" = "1" ]; then
  for SEED in $SEEDS; do
    RUN_NAME="SFTBestMove_seed${SEED}"
    RUN_LOG_DIR="$LOG_DIR/$RUN_NAME"

    echo "=== Tinker SFT training $RUN_NAME ==="
    python "$EXP_DIR/tinker_train.py" \
      --mode sft \
      --base_model "$MODEL" \
      --renderer "$TINKER_RENDERER" \
      ${PROJECT_ARGS[@]-} \
      --output_dir "$RUN_LOG_DIR" \
      --position_buffer "$POSITION_BUFFER" \
      --run_name_prefix "$RUN_NAME" \
      --steps "$SFT_STEPS" \
      --positions_per_step "$POSITIONS_PER_STEP" \
      --learning_rate "$LEARNING_RATE" \
      --lora_rank "$LORA_RANK" \
      --seed "$SEED"

    SAMPLER_PATH="$(
      python - "$RUN_LOG_DIR/checkpoint_paths.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["final_sampler_path"])
PY
    )"

    echo "=== Tinker SFT eval $RUN_NAME ==="
    python "$EXP_DIR/tinker_eval_move_quality.py" \
      --model "$SAMPLER_PATH" \
      --condition BaseSimple \
      --model_label "$RUN_NAME" \
      --banks "$DATA_PATH" \
      --output "$RESULTS_DIR/${RUN_NAME}.json" \
      --renderer "$TINKER_RENDERER" \
      --max_new_tokens "$EVAL_MAX_TOKENS" \
      --temperature 0.0 \
      --batch_size "$EVAL_BATCH_SIZE" \
      ${PROJECT_ARGS[@]-} \
      ${EVAL_LIMIT_ARGS[@]-}
  done
fi

for CONDITION in $TRAIN_CONDITIONS; do
  for SEED in $SEEDS; do
    RUN_NAME="${CONDITION}_seed${SEED}"
    RUN_LOG_DIR="$LOG_DIR/$RUN_NAME"

    TRAIN_ARGS=(
      "$EXP_DIR/tinker_train.py"
      --mode rl
      --condition "$CONDITION"
      --base_model "$MODEL"
      --renderer "$TINKER_RENDERER"
      ${PROJECT_ARGS[@]-}
      --output_dir "$RUN_LOG_DIR"
      --position_buffer "$POSITION_BUFFER"
      --run_name_prefix "$RUN_NAME"
      --steps "$RL_STEPS"
      --positions_per_step "$POSITIONS_PER_STEP"
      --group_size "$GROUP_SIZE"
      --max_tokens "$MAX_TOKENS"
      --temperature "$TEMPERATURE"
      --learning_rate "$LEARNING_RATE"
      --lora_rank "$LORA_RANK"
      --loss_fn "$LOSS_FN"
      --seed "$SEED"
    )
    if [ "$WANDB" = "1" ]; then
      TRAIN_ARGS+=(--wandb --wandb_run_name "tinker_${RUN_NAME}")
    fi

    echo "=== Tinker RL training $RUN_NAME ==="
    python "${TRAIN_ARGS[@]}"

    SAMPLER_PATH="$(
      python - "$RUN_LOG_DIR/checkpoint_paths.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["final_sampler_path"])
PY
    )"

    echo "=== Tinker RL eval $RUN_NAME ==="
    python "$EXP_DIR/tinker_eval_move_quality.py" \
      --model "$SAMPLER_PATH" \
      --condition "$CONDITION" \
      --model_label "$RUN_NAME" \
      --banks "$DATA_PATH" \
      --output "$RESULTS_DIR/${RUN_NAME}.json" \
      --renderer "$TINKER_RENDERER" \
      --max_new_tokens "$EVAL_MAX_TOKENS" \
      --temperature 0.0 \
      --batch_size "$EVAL_BATCH_SIZE" \
      ${PROJECT_ARGS[@]-} \
      ${EVAL_LIMIT_ARGS[@]-}
  done
done

python "$EXP_DIR/summarize_results.py" \
  --results_dir "$RESULTS_DIR" \
  --output "$RESULTS_DIR/summary.json"

echo "=== Tinker experiment complete ==="
echo "Results: $RESULTS_DIR"
