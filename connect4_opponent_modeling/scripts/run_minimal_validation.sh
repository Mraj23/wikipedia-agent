#!/usr/bin/env bash
# Smallest experiment that can plausibly validate the early D vs E signal.
#
# What it does:
#   1. Optional baseline evals for A and F
#   2. Short RL training runs for D and E only
#   3. Canonical post-training evals for D and E on a reduced transfer set
#
# Keeps the core recipe intact (same model family, same reward definitions,
# same eval harness) while cutting runtime enough for a first-pass read.
#
# Important:
#   - Mid-training evals are disabled by default here because even "light"
#     callbacks can dominate wall-clock time on 40GB GPUs.
#   - This script auto-tunes group size + vLLM memory reservation based on
#     detected GPU VRAM, unless you explicitly override them.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -d "$ROOT_DIR/.venv" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/gpu_env.sh"

MODEL=${MODEL:-Qwen/Qwen3-4B}
STEPS=${STEPS:-150}
TRAIN_SEED=${TRAIN_SEED:-42}
EVAL_SEED=${EVAL_SEED:-42}
NUM_GAMES=${NUM_GAMES:-20}
BASELINE_NUM_GAMES=${BASELINE_NUM_GAMES:-10}
PROJECT=${PROJECT:-connect4-opponent-modeling}
USE_VLLM=${USE_VLLM:-1}
RUN_BASELINES=${RUN_BASELINES:-1}
GAMES=${GAMES:-connect_four breakthrough}
BASELINE_GAMES=${BASELINE_GAMES:-breakthrough}
BASELINE_SKIP=${BASELINE_SKIP:-pons_benchmark math}
CONDITIONS=${CONDITIONS:-E D}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GPU_TOTAL_MIB=""

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
fi

if [ -z "${GROUP+x}" ]; then
  if [ -n "$GPU_TOTAL_MIB" ] && [ "$GPU_TOTAL_MIB" -lt 60000 ]; then
    GROUP=32
  else
    GROUP=64
  fi
else
  GROUP=${GROUP}
fi

if [ -z "${EVAL_EVERY+x}" ]; then
  EVAL_EVERY=999999
else
  EVAL_EVERY=${EVAL_EVERY}
fi

if [ -z "${C4_VLLM_GPU_MEMORY_UTILIZATION+x}" ]; then
  if [ -n "$GPU_TOTAL_MIB" ] && [ "$GPU_TOTAL_MIB" -lt 60000 ]; then
    export C4_VLLM_GPU_MEMORY_UTILIZATION=0.50
  fi
fi

TRAIN_ARGS=()
if [ "$USE_VLLM" = "1" ]; then
  TRAIN_ARGS+=(--use_vllm)
fi

python "$ROOT_DIR/scripts/verify_setup.py" --expect-gpu

echo "============================================"
echo "Minimal Validation Run"
echo "  Model: $MODEL"
echo "  Steps: $STEPS per condition"
if [ -n "$GPU_TOTAL_MIB" ]; then
  echo "  GPU memory: ${GPU_TOTAL_MIB} MiB"
fi
echo "  Group size: $GROUP"
echo "  Eval every: $EVAL_EVERY"
echo "  vLLM gpu util: ${C4_VLLM_GPU_MEMORY_UTILIZATION:-default}"
echo "  Conditions: $CONDITIONS"
echo "  Transfer games: $GAMES"
echo "  Ladder games/opponent: $NUM_GAMES"
echo "  Baseline games: $BASELINE_GAMES"
echo "  Baseline ladder games/opponent: $BASELINE_NUM_GAMES"
echo "  W&B project: $PROJECT"
echo "============================================"
echo ""

if [ "$RUN_BASELINES" = "1" ]; then
  echo "--- Baseline eval: A ---"
  python -m eval.baseline_eval \
    --model "$MODEL" \
    --condition A \
    --output results/minimal_validation \
    --games $BASELINE_GAMES \
    --num_games "$BASELINE_NUM_GAMES" \
    --seed "$EVAL_SEED" \
    --skip $BASELINE_SKIP

  echo ""
  echo "--- Baseline eval: F ---"
  python -m eval.baseline_eval \
    --model "$MODEL" \
    --condition F \
    --prompt_style opponent_aware \
    --output results/minimal_validation \
    --games $BASELINE_GAMES \
    --num_games "$BASELINE_NUM_GAMES" \
    --seed "$EVAL_SEED" \
    --skip $BASELINE_SKIP
  echo ""
fi

for COND in $CONDITIONS; do
  COND_LOWER=$(echo "$COND" | tr '[:upper:]' '[:lower:]')
  LOG_DIR="logs/minimal_${COND_LOWER}_${TIMESTAMP}"
  RUN_NAME="minimal_${COND}_${STEPS}steps_g${GROUP}_${TIMESTAMP}"

  echo "=== Training Condition $COND ==="
  echo "  Log dir: $LOG_DIR"
  echo "  W&B run: $RUN_NAME"
  echo ""

  python -m spiral.train \
    --condition "$COND" \
    --model "$MODEL" \
    --log_dir "$LOG_DIR" \
    --game_steps "$STEPS" \
    --group_size "$GROUP" \
    --eval_every "$EVAL_EVERY" \
    --seed "$TRAIN_SEED" \
    --wandb \
    --wandb_project "$PROJECT" \
    --wandb_run_name "$RUN_NAME" \
    "${TRAIN_ARGS[@]}"

  echo ""
  echo "--- Post-training eval: $COND ---"
  python -m eval.baseline_eval \
    --model "checkpoints/condition_${COND_LOWER}/final" \
    --condition "$COND" \
    --output results/minimal_validation \
    --games $GAMES \
    --num_games "$NUM_GAMES" \
    --seed "$EVAL_SEED" \
    --skip math
  echo ""
done

echo "============================================"
echo "Minimal validation run complete."
echo "Results: results/minimal_validation/"
echo "Check W&B for:"
echo "  train/valid_pct"
echo "  train/zero_reward_pct"
echo "  eval/pons_pct_optimal"
echo "  eval/probe_accuracy"
echo "============================================"
