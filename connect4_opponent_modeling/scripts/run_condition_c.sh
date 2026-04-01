#!/bin/bash
set -e

CONDITION="C"
SFT_CKPT="checkpoints/condition_a"
LOG_DIR="logs/condition_${CONDITION,,}"

# 1. Check SFT checkpoint exists
if [ ! -d "$SFT_CKPT" ]; then
  echo "ERROR: SFT warmup checkpoint not found at $SFT_CKPT. Run sft_train.py first."
  exit 1
fi

# 2. Lock probe positions if not already locked
python -c "
from eval.probe import lock_probe_positions
import os
if not os.path.exists('data/probe_positions_locked.jsonl'):
    lock_probe_positions('data/pons_benchmark')
    print('Probe positions locked.')
else:
    print('Probe positions already locked.')
"

# 3. Launch SPIRAL training (placeholder — fill in your SPIRAL command)
mkdir -p $LOG_DIR
echo "Starting Condition $CONDITION RL training..."
# python -m spiral.train --config training/grpo_config.py --condition $CONDITION \
#   --model $SFT_CKPT --log_dir $LOG_DIR

# 4. Eval every 1000 steps (called by training loop via callback)
echo "Training complete. Running final eval..."
python -m eval.baseline_eval \
  --model "checkpoints/condition_${CONDITION,,}" \
  --condition $CONDITION \
  --output results/
