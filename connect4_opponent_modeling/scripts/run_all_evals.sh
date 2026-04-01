#!/bin/bash
set -e

echo "=== Running All Condition Evaluations ==="

for CONDITION in A B C D E F; do
  CKPT="checkpoints/condition_${CONDITION,,}"
  if [ -d "$CKPT" ]; then
    echo ""
    echo "--- Evaluating Condition $CONDITION from $CKPT ---"
    python -m eval.baseline_eval \
      --model "$CKPT" \
      --condition "$CONDITION" \
      --output results/ \
      --skip gamebench
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
