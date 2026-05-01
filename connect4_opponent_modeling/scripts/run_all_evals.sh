#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -d "$ROOT_DIR/.venv" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/gpu_env.sh"

echo "=== Running Canonical Experiment Evaluations ==="

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
NUM_GAMES=${NUM_GAMES:-50}
EVAL_SEED=${EVAL_SEED:-42}

echo ""
echo "--- Evaluating Condition A (instruct baseline) from $BASE_MODEL ---"
python -m eval.baseline_eval \
  --model "$BASE_MODEL" \
  --condition A \
  --output results/ \
  --num_games "$NUM_GAMES" \
  --seed "$EVAL_SEED"

echo ""
echo "--- Evaluating Condition F (prompt-only opponent-aware baseline) from $BASE_MODEL ---"
python -m eval.baseline_eval \
  --model "$BASE_MODEL" \
  --condition F \
  --prompt_style opponent_aware \
  --output results/ \
  --num_games "$NUM_GAMES" \
  --seed "$EVAL_SEED"

for CONDITION in B C D E; do
  CKPT="checkpoints/condition_${CONDITION,,}"
  if [ -d "$CKPT" ]; then
    echo ""
    echo "--- Evaluating Condition $CONDITION from $CKPT ---"
    python -m eval.baseline_eval \
      --model "$CKPT" \
      --condition "$CONDITION" \
      --output results/ \
      --num_games "$NUM_GAMES" \
      --seed "$EVAL_SEED"
  else
    echo "Skipping Condition $CONDITION: checkpoint not found at $CKPT"
  fi
done

echo ""
echo "=== All evaluations complete ==="
echo "Results saved to results/"

# Generate plots
echo ""
echo "Generating plots..."
python -m analysis.plot_curves --results results/ --output results/

# Print correlation analysis
echo ""
python -m analysis.correlation --results results/
