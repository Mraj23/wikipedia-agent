#!/bin/bash
# Lambda Labs instance setup for Connect Four Opponent Modeling experiment.
# Run this after SSH-ing into a fresh Lambda Labs instance.
#
# Prerequisites:
#   - Persistent filesystem attached at /lambda/nfs/persistent-storage/
#   - SSH access to the instance
#
# Usage:
#   bash lambda_setup.sh
#
# After running, activate the environment with:
#   source ~/c4env/bin/activate
#   cd /lambda/nfs/persistent-storage/wikipedia-agent/connect4_opponent_modeling

set -e

PERSISTENT=/lambda/nfs/persistent-storage
PROJECT=$PERSISTENT/wikipedia-agent/connect4_opponent_modeling

echo "=== Lambda Labs Setup for Connect Four Experiment ==="
echo ""

# --- Step 1: Clone or update repo ---
echo "[1/6] Setting up repository..."
if [ -d "$PERSISTENT/wikipedia-agent" ]; then
    echo "  Repo exists, pulling latest..."
    cd $PERSISTENT/wikipedia-agent
    git pull || echo "  Warning: git pull failed, using existing code"
else
    echo "  Cloning repository..."
    cd $PERSISTENT
    git clone https://github.com/Mraj23/wikipedia-agent.git
fi
cd $PROJECT
echo "  Done. Working directory: $(pwd)"
echo ""

# --- Step 2: Python environment ---
echo "[2/6] Setting up Python environment..."
if [ -d ~/c4env ]; then
    echo "  Existing venv found, reusing..."
else
    echo "  Creating venv with system site-packages (keeps Lambda's PyTorch)..."
    python3 -m venv --system-site-packages ~/c4env
fi
source ~/c4env/bin/activate
echo "  Python: $(python --version)"
echo "  Pip: $(pip --version)"
echo ""

# --- Step 3: Install dependencies ---
echo "[3/6] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    open_spiel \
    transformers>=4.51.0 \
    datasets \
    accelerate \
    peft \
    tqdm \
    matplotlib \
    pandas \
    scipy \
    tensorboard \
    wandb \
    bitsandbytes \
    pytest
echo "  Dependencies installed."
echo ""

# --- Step 4: Build Pons solver for Linux ---
echo "[4/6] Building Pons Connect Four solver..."
if [ -x "$PROJECT/connect4_solver" ] && file "$PROJECT/connect4_solver" | grep -q "ELF"; then
    echo "  Linux solver binary already exists, skipping build."
else
    echo "  Installing build tools..."
    sudo apt-get update -qq && sudo apt-get install -y -qq build-essential > /dev/null 2>&1

    echo "  Cloning Pons solver repo..."
    rm -rf /tmp/pons_solver
    git clone --quiet https://github.com/PascalPons/connect4 /tmp/pons_solver

    echo "  Building solver..."
    cd /tmp/pons_solver
    make -j$(nproc) 2>&1 | tail -3

    # The Pons repo may produce different binary names depending on version
    if [ -f c4solver ]; then
        cp c4solver $PROJECT/connect4_solver
    elif [ -f connect4_solver ]; then
        cp connect4_solver $PROJECT/connect4_solver
    else
        # Find any executable that was built
        BUILT=$(find . -maxdepth 1 -type f -executable ! -name Makefile | head -1)
        if [ -n "$BUILT" ]; then
            echo "  Found built binary: $BUILT"
            cp "$BUILT" $PROJECT/connect4_solver
        else
            echo "  WARNING: Solver build produced no binary. Will use minimax fallback."
        fi
    fi

    cd $PROJECT
    chmod +x connect4_solver 2>/dev/null || true
    echo "  Solver build complete."
fi
echo ""

# --- Step 5: Verify installation ---
echo "[5/6] Running verification checks..."
echo -n "  CUDA: "
python -c "import torch; print(f'available={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo -n "  OpenSpiel: "
python -c "import pyspiel; print('OK')"

echo -n "  Pons solver: "
python -c "from env.pons_wrapper import PonsSolver; s = PonsSolver(); print(f'available={s.is_available()}')"

echo -n "  Transformers: "
python -c "import transformers; print(f'v{transformers.__version__}')"

echo -n "  W&B: "
python -c "import wandb; print(f'v{wandb.__version__}')"

echo ""
echo "  Running unit tests..."
cd $PROJECT
python -m pytest tests/ -v --tb=short 2>&1 | tail -5
echo ""

# --- Step 6: W&B login prompt ---
echo "[6/6] Weights & Biases authentication..."
if [ -f ~/.netrc ] && grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    echo "  W&B already authenticated."
else
    echo "  Please log in to W&B (get API key from https://wandb.ai/authorize):"
    wandb login
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps (SPIRAL approach — no SFT, direct RL from base model):"
echo "  1. Activate env:   source ~/c4env/bin/activate"
echo "  2. Go to project:  cd $PROJECT"
echo "  3. Run preliminary: bash scripts/run_preliminary.sh"
echo "     (Downloads Qwen3-4B-Base from HF, trains C and E with group_size=64)"
echo "============================================"
