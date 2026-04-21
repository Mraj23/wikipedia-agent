#!/bin/bash
# Run preliminary RL training for conditions C, D, E.
# Tests the causal ladder: value-only → future-state → opponent modeling.
#
# Uses Qwen3-4B (instruct) directly — no SFT warmup needed.
# Pre-generated position buffer loaded from data/position_buffer.json.
#
# Prerequisites:
#   - W&B authenticated (wandb login or WANDB_API_KEY set)
#   - GPU available (80GB+ recommended)
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

MODEL=Qwen/Qwen3-4B
STEPS=500
GROUP=64
SEED=42
PROJECT=connect4-opponent-modeling

# Verify GPU
python -c "import torch; assert torch.cuda.is_available(), 'No GPU found!'"

# Verify position buffer exists
if [ ! -f data/position_buffer.json ]; then
    echo "ERROR: Position buffer not found at data/position_buffer.json"
    echo "Generate it locally first (CPU): python -c 'from spiral.position_buffer import PositionBuffer; b = PositionBuffer(1000, 2); b.save(\"data/position_buffer.json\")'"
    exit 1
fi

echo "============================================"
echo "Preliminary RL Training (Causal Ladder)"
echo "  Model: $MODEL (instruct, no SFT)"
echo "  Steps: $STEPS per condition"
echo "  Group size: $GROUP"
echo "  Conditions: C (value), D (future-state), E (opponent modeling)"
echo "  W&B project: $PROJECT"
echo "============================================"
echo ""

START_TIME=$(date +%s)

for COND in C D E; do
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

    # Quick reward check
    python -c "
import json
log = json.load(open('$LOG_DIR/train_log.json'))
last_10 = log[-10:] if len(log) >= 10 else log
avg_r = sum(e['mean_reward'] for e in last_10) / len(last_10)
first_10 = log[:10]
avg_r0 = sum(e['mean_reward'] for e in first_10) / len(first_10)
print(f'  First 10 avg reward: {avg_r0:.4f}')
print(f'  Last 10 avg reward:  {avg_r:.4f}')
if avg_r > avg_r0 + 0.01:
    print('  Reward is INCREASING — learning signal detected.')
elif avg_r == 0.0:
    print('  WARNING: Reward collapsed to zero!')
else:
    print('  Reward is FLAT — no clear learning yet.')
" 2>/dev/null || true
    echo ""
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo "============================================"
echo "All preliminary conditions complete!"
echo "Total time: ${ELAPSED} minutes"
echo ""
echo "Key comparison: D vs E on transfer evals"
echo "  python -m eval.baseline_eval --model checkpoints/condition_d/best --condition D"
echo "  python -m eval.baseline_eval --model checkpoints/condition_e/best --condition E"
echo "============================================"
