#!/bin/bash
# Run preliminary RL training for conditions B, C, E.
# Each condition runs 1000 steps with group_size=8 and W&B logging.
#
# Prerequisites:
#   - SFT checkpoint at checkpoints/condition_a/best/
#   - W&B authenticated (wandb login)
#   - GPU available
#
# Usage:
#   tmux new-session -d -s training 'bash scripts/run_preliminary.sh'
#   tmux attach -t training    # to monitor
#   # Ctrl+b, d to detach (training continues)

set -e

# Activate environment
source ~/c4env/bin/activate 2>/dev/null || true
cd "$(dirname "$0")/.."

MODEL=checkpoints/condition_a/best
STEPS=1000
GROUP=8
SEED=42
PROJECT=connect4-opponent-modeling

# Verify SFT checkpoint exists
if [ ! -d "$MODEL" ]; then
    echo "ERROR: SFT checkpoint not found at $MODEL"
    echo "Run SFT warmup first:"
    echo "  python -m training.sft_train --model Qwen/Qwen3-4B-Base --data data/sft_warmup.jsonl --output_dir checkpoints/condition_a --epochs 3"
    exit 1
fi

# Verify GPU
python -c "import torch; assert torch.cuda.is_available(), 'No GPU found!'"

echo "============================================"
echo "Preliminary RL Training"
echo "  Model: $MODEL"
echo "  Steps: $STEPS per condition"
echo "  Group size: $GROUP"
echo "  Conditions: B, C, E"
echo "  W&B project: $PROJECT"
echo "============================================"
echo ""

START_TIME=$(date +%s)

for COND in B C E; do
    COND_LOWER=$(echo $COND | tr '[:upper:]' '[:lower:]')
    LOG_DIR=logs/prelim_${COND_LOWER}
    RUN_NAME=prelim_${COND}_${STEPS}steps

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
    echo ""
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo "============================================"
echo "All preliminary conditions complete!"
echo "Total time: ${ELAPSED} minutes"
echo ""
echo "Next steps:"
echo "  1. Check W&B dashboard for training curves"
echo "  2. Compare checkpoints:"
echo "     python -m eval.baseline_eval --model checkpoints/condition_c/best --condition C --output results/prelim_c.json"
echo "     python -m eval.baseline_eval --model checkpoints/condition_e/best --condition E --output results/prelim_e.json"
echo "  3. Compare results:"
echo "     python -c \"import json; c=json.load(open('logs/prelim_c/train_log.json')); e=json.load(open('logs/prelim_e/train_log.json')); print(f'C final reward: {c[-1][\\\"mean_reward\\\"]:.3f}'); print(f'E final reward: {e[-1][\\\"mean_reward\\\"]:.3f}')\""
echo "============================================"
