# Does Explicit Opponent Modeling During RL Training Develop Transferable Adversarial Reasoning?

## A Connect Four Study

This repository contains the experimental infrastructure for studying whether training an LLM to explicitly model its opponent during reinforcement learning develops internal representations that transfer to novel adversarial domains.

### The Six Conditions

| Condition | Description | Reward Signal | Key Addition |
|-----------|-------------|---------------|--------------|
| **A** | SFT Only | Imitation of solver-optimal moves | Baseline |
| **B** | Self-Play RL | Win/loss/draw | Adversarial pressure |
| **C** | Solver-RL (value) | 0.6 move + 0.3 terminal + 0.1 format | Position evaluation |
| **D** | Solver-RL + future-state | 0.5 move + 0.2 future + 0.2 terminal + 0.1 format | Forward projection |
| **E** | Solver-RL + opponent modeling | 0.5 move + 0.2 pred + 0.2 terminal + 0.1 format | Adversarial projection |
| **F** | Prompt-only baseline | No RL (inference-time reasoning) | ReTA comparison |

**Key comparisons:** C→D (value to future-state), D→E (future-state to opponent modeling), E vs F (training vs prompting).

## Installation

```bash
# Clone and install
git clone <repo-url>
cd connect4_opponent_modeling
pip install -r requirements.txt

# Verify OpenSpiel works
python -c "import pyspiel; print(pyspiel.load_game('connect_four'))"

# Optional: Install GTBench for transfer evaluation
git clone https://github.com/jinhaoduan/GTBench
cd GTBench && pip install -e . && cd ..

# Optional: Install GameBench
# See https://github.com/Costarelli/GameBench
```

## Quick Start — Smoke Test

```bash
# Run the Connect Four environment demo
python -m env.connect_four_env

# Run minimax solver demo
python -m training.minimax

# Run all tests
pytest tests/ -v
```

## Experiment Phases

### Phase 0: Setup & Eval Pipeline
Verify infrastructure works on CPU without models.

```bash
pytest tests/ -v
python -m env.connect_four_env
python -m training.prompts
```

### Phase 1: Base Model Baseline
Record baseline eval for the untuned Qwen3-4B-Base model.

```bash
python -m eval.baseline_eval --model Qwen/Qwen3-4B-Base --condition base --output results/
```

### Phase 2: SFT Warmup (Condition A)
Generate training data and run supervised fine-tuning.

```bash
# Generate 50k positions (takes ~30 min on CPU)
python -m training.sft_data_gen --n 50000 --output data/sft_warmup.jsonl

# Train SFT warmup (requires GPU)
python -m training.sft_train \
  --model Qwen/Qwen3-4B-Base \
  --data data/sft_warmup.jsonl \
  --output_dir checkpoints/condition_a \
  --epochs 3
```

**Gate check:** Val accuracy must be 35-65%. Below 35% = insufficient warmup. Above 65% = potential overfitting.

### Phase 3: Lock Probe Positions
**This step runs ONCE and must never be repeated.**

```bash
python -c "from eval.probe import lock_probe_positions; lock_probe_positions('data/pons_benchmark')"
```

### Phase 4: RL Training (Conditions B-E) + Condition F
```bash
bash scripts/run_condition_b.sh
bash scripts/run_condition_c.sh
bash scripts/run_condition_d.sh
bash scripts/run_condition_e.sh
```

Condition F requires no training — it's evaluated at inference time with the SFT checkpoint and opponent-modeling prompts.

### Phase 5: Analysis
```bash
bash scripts/run_all_evals.sh
python -m analysis.correlation --results results/
python -m analysis.plot_curves --results results/ --output results/
```

## Benchmark Data

### Pons Connect Four Benchmark
Download from [blog.gamesolver.org](https://blog.gamesolver.org/solving-connect-four/):

1. Download the test position files (Test_L3_R1 through Test_L1_R3)
2. Place them in `data/pons_benchmark/`

The benchmark provides positions at various game phases with known optimal moves, scored by the [Pons solver](https://github.com/PascalPons/connect4).

### Optional: Pons Solver Binary
For perfect play analysis (instead of minimax fallback):

```bash
git clone https://github.com/PascalPons/connect4
cd connect4
make
cp connect4_solver ../connect4_opponent_modeling/
```

## Project Structure

```
connect4_opponent_modeling/
├── env/                    # Game environment
│   ├── connect_four_env.py # OpenSpiel wrapper
│   └── pons_wrapper.py     # Pons solver + minimax fallback
├── training/               # Training infrastructure
│   ├── prompts.py          # 6 condition prompt templates
│   ├── reward.py           # Reward functions (conditions B-E)
│   ├── minimax.py          # Alpha-beta minimax solver
│   ├── sft_data_gen.py     # SFT warmup data generation
│   ├── sft_train.py        # SFT training script
│   └── grpo_config.py      # GRPO hyperparameters
├── eval/                   # Evaluation suite
│   ├── baseline_eval.py    # Master eval runner
│   ├── pons_benchmark.py   # In-domain Connect Four eval
│   ├── probe.py            # Mechanistic probes
│   ├── gtbench_eval.py     # GTBench transfer eval
│   ├── gamebench_eval.py   # GameBench stub
│   └── math_eval.py        # GSM8K + MATH-500
├── analysis/               # Results analysis
│   ├── correlation.py      # Probe-transfer correlation
│   └── plot_curves.py      # Learning curve plots
├── scripts/                # Shell scripts
├── tests/                  # Unit tests
├── data/                   # Data directory
├── CLAUDE.md               # Claude Code instructions
└── README.md               # This file
```

## Base Model

All conditions use **Qwen3-4B-Base**. Never compare results across different base models.

## Citation

```bibtex
@article{connect4_opponent_modeling_2026,
  title={Does Explicit Opponent Modeling During RL Training Develop Transferable Adversarial Reasoning? A Connect Four Study},
  author={TBD},
  year={2026}
}
```
