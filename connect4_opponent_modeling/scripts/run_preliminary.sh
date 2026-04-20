#!/bin/bash
# Run preliminary RL training for conditions C, E (then optionally B).
# Matches SPIRAL paper approach: base model → RL directly, no SFT.
#
# Config: group_size=64, lr=1e-6, 1000 steps per condition.
#
# Prerequisites:
#   - W&B authenticated (wandb login or WANDB_API_KEY set)
#   - GPU available
#   - Pons solver binary + 7x6.book in project dir
#
# Usage:
#   tmux new-session -d -s training 'bash scripts/run_preliminary.sh'
#   tmux attach -t training    # to monitor
#   # Ctrl+b, d to detach (training continues)

set -e

# Activate environment
source ~/c4env/bin/activate 2>/dev/null || true
cd "$(dirname "$0")/.."

# No SFT checkpoint needed — go directly from base model (SPIRAL approach)
MODEL=Qwen/Qwen3-4B-Base
STEPS=1000
GROUP=64
SEED=42
LR=1e-6
PROJECT=connect4-opponent-modeling

# Verify GPU
python -c "import torch; assert torch.cuda.is_available(), 'No GPU found!'"

echo "============================================"
echo "Preliminary RL Training (SPIRAL approach)"
echo "  Model: $MODEL (base, no SFT)"
echo "  Steps: $STEPS per condition"
echo "  Group size: $GROUP"
echo "  Learning rate: $LR"
echo "  Conditions: C, E (dense reward first), then B (sparse)"
echo "  W&B project: $PROJECT"
echo "============================================"
echo ""

START_TIME=$(date +%s)

# Run C and E first (dense rewards bootstrap format learning)
# Then B (sparse — only if C/E succeed)
for COND in C E; do
    COND_LOWER=$(echo $COND | tr '[:upper:]' '[:lower:]')
    LOG_DIR=logs/prelim_${COND_LOWER}
    RUN_NAME=prelim_${COND}_${STEPS}steps_g${GROUP}

    echo "=== Condition $COND ==="
    echo "  Log dir: $LOG_DIR"
    echo "  W&B run: $RUN_NAME"
    echo "  Started: $(date)"
    echo ""

    python -m spiral.train \
        --condition $COND \
        --model $MODEL \
        --log_dir $LOG_DIR \
        --game_steps $STEPS \
        --group_size $GROUP \
        --seed $SEED \
        --wandb \
        --wandb_project $PROJECT \
        --wandb_run_name $RUN_NAME

    echo ""
    echo "  Condition $COND complete: $(date)"

    # Quick check: did rewards collapse?
    python -c "
import json
log = json.load(open('$LOG_DIR/train_log.json'))
last_10 = log[-10:] if len(log) >= 10 else log
avg_r = sum(e['mean_reward'] for e in last_10) / len(last_10)
print(f'  Last 10 steps avg reward: {avg_r:.4f}')
if avg_r == 0.0:
    print('  WARNING: Reward collapsed to zero!')
else:
    print('  Rewards look healthy.')
" 2>/dev/null || true
    echo ""
done

# Only run B if C and E succeeded
echo "=== Condition B (sparse reward) ==="
echo "  Started: $(date)"
python -m spiral.train \
    --condition B \
    --model $MODEL \
    --log_dir logs/prelim_b \
    --game_steps $STEPS \
    --group_size $GROUP \
    --seed $SEED \
    --wandb \
    --wandb_project $PROJECT \
    --wandb_run_name prelim_B_${STEPS}steps_g${GROUP}

echo "  Condition B complete: $(date)"
echo ""

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo "============================================"
echo "All preliminary conditions complete!"
echo "Total time: ${ELAPSED} minutes"
echo ""
echo "Next steps:"
echo "  1. Check W&B dashboard for training curves"
echo "  2. Compare final rewards:"
echo "     python -c \"import json; [print(f'{c}: {json.load(open(f\\\"logs/prelim_{c}/train_log.json\\\"))[-1][\\\"mean_reward\\\"]:.4f}') for c in ['c','e','b']]\""
echo "============================================"
