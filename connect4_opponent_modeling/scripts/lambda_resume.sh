#!/bin/bash
# Resume experiment on a fresh Lambda Labs instance.
#
# This script re-creates the environment and picks up training
# from wherever it left off. Run after SSH-ing into a new instance.
#
# Usage:
#   # First time: clone and run
#   git clone https://github.com/Mraj23/wikipedia-agent.git ~/wikipedia-agent
#   bash ~/wikipedia-agent/connect4_opponent_modeling/scripts/lambda_resume.sh
#
#   # Or if repo already exists:
#   bash ~/wikipedia-agent/connect4_opponent_modeling/scripts/lambda_resume.sh

set -e

PROJECT=~/wikipedia-agent/connect4_opponent_modeling
WANDB_KEY="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"

echo "=== Lambda Labs Resume Script ==="
echo ""

# --- Step 1: Clone or update repo ---
echo "[1/5] Repository..."
if [ ! -d ~/wikipedia-agent ]; then
    echo "  Cloning..."
    git clone https://github.com/Mraj23/wikipedia-agent.git ~/wikipedia-agent
else
    echo "  Updating..."
    cd ~/wikipedia-agent && git pull || echo "  Warning: git pull failed"
fi
echo ""

# --- Step 2: Create venv and install deps ---
echo "[2/5] Python environment..."
if [ ! -d ~/c4env ]; then
    echo "  Creating venv..."
    python3 -m venv ~/c4env
    source ~/c4env/bin/activate
    pip install --quiet --upgrade pip
    echo "  Installing PyTorch..."
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
    echo "  Installing dependencies..."
    pip install --quiet \
        open_spiel transformers datasets accelerate peft \
        tqdm matplotlib pandas scipy wandb bitsandbytes
else
    echo "  Venv exists, activating..."
    source ~/c4env/bin/activate
fi
echo "  Python: $(python --version)"
echo ""

# --- Step 3: Build Pons solver ---
echo "[3/5] Pons solver..."
cd $PROJECT
if [ -x ./connect4_solver ] && file ./connect4_solver | grep -q "ELF"; then
    echo "  Linux binary exists."
else
    echo "  Building from source..."
    sudo apt-get update -qq && sudo apt-get install -y -qq build-essential > /dev/null 2>&1
    rm -rf /tmp/pons_solver
    git clone --quiet https://github.com/PascalPons/connect4 /tmp/pons_solver
    cd /tmp/pons_solver && make -j$(nproc) 2>&1 | tail -1
    cp c4solver $PROJECT/connect4_solver 2>/dev/null \
        || cp connect4_solver $PROJECT/connect4_solver 2>/dev/null \
        || echo "  WARNING: Build failed, using minimax fallback"
    chmod +x $PROJECT/connect4_solver 2>/dev/null || true
    cd $PROJECT
fi
echo ""

# --- Step 4: Configure W&B ---
echo "[4/5] Weights & Biases..."
export WANDB_API_KEY=$WANDB_KEY
echo "export WANDB_API_KEY=$WANDB_KEY" >> ~/.bashrc 2>/dev/null
echo "export PATH=\$HOME/.local/bin:\$PATH" >> ~/.bashrc 2>/dev/null
python -c "import wandb; wandb.login(); print('  Authenticated as:', wandb.api.default_entity)" 2>/dev/null || echo "  W&B login failed (non-fatal)"
echo ""

# --- Step 5: Check experiment state ---
echo "[5/5] Experiment state..."
cd $PROJECT

# Check SFT checkpoint
if [ -d checkpoints/condition_a/best ]; then
    echo "  SFT checkpoint: READY at checkpoints/condition_a/best/"
    SFT_DONE=true
else
    echo "  SFT checkpoint: NOT FOUND — need to run SFT warmup"
    SFT_DONE=false
fi

# Check RL conditions
for COND in b c e; do
    COND_UPPER=$(echo $COND | tr '[:lower:]' '[:upper:]')
    LOG=logs/prelim_${COND}/train_log.json
    CKPT=checkpoints/condition_${COND}/final
    if [ -d "$CKPT" ]; then
        STEPS=$(python -c "import json; l=json.load(open('$LOG')); print(len(l))" 2>/dev/null || echo "?")
        echo "  Condition $COND_UPPER: COMPLETE ($STEPS steps)"
    elif [ -f "$LOG" ]; then
        STEPS=$(python -c "import json; l=json.load(open('$LOG')); print(len(l))" 2>/dev/null || echo "?")
        echo "  Condition $COND_UPPER: PARTIAL ($STEPS steps, no final checkpoint)"
    else
        echo "  Condition $COND_UPPER: NOT STARTED"
    fi
done

# Check data files
echo ""
if [ -f data/sft_warmup.jsonl ]; then
    echo "  SFT data: present"
else
    echo "  SFT data: MISSING — need to upload or regenerate"
fi
if [ -f data/probe_positions_locked.jsonl ]; then
    echo "  Probe positions: present (locked)"
else
    echo "  Probe positions: MISSING — need to upload"
fi

# GPU info
echo ""
python -c "import torch; print('  GPU:', torch.cuda.get_device_name(0), '|', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')" 2>/dev/null || echo "  GPU: not detected"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""

if [ "$SFT_DONE" = false ]; then
    echo "NEXT: Run SFT warmup:"
    echo "  source ~/c4env/bin/activate && cd $PROJECT"
    echo "  python -m training.sft_train --model Qwen/Qwen3-4B-Base \\"
    echo "    --data data/sft_warmup.jsonl --output_dir checkpoints/condition_a \\"
    echo "    --epochs 3 --batch_size 8 --lr 2e-5"
else
    echo "NEXT: Run preliminary RL training:"
    echo "  source ~/c4env/bin/activate && cd $PROJECT"
    echo "  bash scripts/run_preliminary.sh"
fi
echo "============================================"
