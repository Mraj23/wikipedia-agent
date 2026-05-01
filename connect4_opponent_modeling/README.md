# Connect Four Opponent Modeling

This repository contains the training and evaluation infrastructure for a study of whether explicitly training an LLM to predict an opponent's next move develops adversarial reasoning that transfers beyond Connect Four.

The current research question, causal ladder, and experimental invariants live in `CLAUDE.md`. This README is the practical entry point for the codebase and reflects the repo's current status as of April 30, 2026.

## Current Status

- The baseline evaluation pipeline is working for Connect Four, Breakthrough, Nim, and Tic-Tac-Toe.
- Untrained Qwen3-4B instruct baselines have been measured and summarized in `results/EVALUATION_BASELINES.md`.
- Prompt-only opponent-modeling did not help in the baseline prompt comparison.
- The mechanistic probe is implemented in `eval/probe.py`, and probe positions are locked at `data/probe_positions_locked.jsonl`.
- The main scientific comparison (`C` vs `D` vs `E`) is not complete yet.
- Earlier RL attempts showed collapse to degenerate play, so current work is focused on stable preliminary training runs and careful bookkeeping.

## Current Experiment Design

All conditions are described in more detail in `CLAUDE.md`, but the current intended ladder is:

| Label | Name | What it adds |
|---|---|---|
| A | Instruct baseline | No RL, evaluate pre-existing instruct capabilities |
| B | Sparse RL | Win/loss optimization pressure |
| C | Solver-RL (value) | Position evaluation |
| D | Solver-RL + future-state | Forward projection |
| E | Solver-RL + opponent modeling | Adversarial projection |
| F | Prompt-only baseline | Inference-time opponent reasoning without RL |

Current reward weights for RL conditions are:

| Condition | Reward weights |
|---|---|
| B | sparse terminal only |
| C | `move=0.67`, `terminal=0.33` |
| D | `move=0.56`, `future=0.22`, `terminal=0.22` |
| E | `move=0.56`, `pred=0.22`, `terminal=0.22` |

Format compliance is currently treated as a binary gate rather than a weighted reward component. See `training/grpo_config.py`.

## What To Trust

The repository currently contains two partially overlapping execution paths:

- The current experiment narrative is the instruct-model design in `CLAUDE.md` plus the sequential preliminary runner in `scripts/run_preliminary.sh`.
- Some legacy modules still assume an SFT checkpoint in `checkpoints/condition_a` and a base-model-first workflow.

For now, treat these as the main sources of truth:

- `CLAUDE.md`: experiment design and invariants
- `results/EVALUATION_BASELINES.md`: reportable baseline numbers
- `results/PRELIMINARY_RESULTS.md`: project-status summary
- `docs/BOOKKEEPING.md`: artifact hygiene and reportability rules

## Quick Start

Run the fast local checks first:

```bash
pytest tests/ -v
python -m env.connect_four_env
python -m training.minimax
python -m training.prompts
```

## Baseline Evaluation

The strongest completed evidence in the repo today is the untrained instruct-model baseline calibration.

Helpful references:

- `results/EVALUATION_BASELINES.md`
- `results/calibration_v4/`
- `scripts/calibrate_transfer.py`

Example command:

```bash
python scripts/calibrate_transfer.py \
  --model Qwen/Qwen3-4B \
  --games connect_four breakthrough nim tic_tac_toe \
  --num_games 10
```

## Mechanistic Probe

The neutral-prompt opponent-prediction probe is implemented in `eval/probe.py`.

Important invariant:

- `data/probe_positions_locked.jsonl` must not be regenerated casually.

Lock positions once:

```bash
python -c "from eval.probe import lock_probe_positions; lock_probe_positions('data/pons_benchmark')"
```

Run the master evaluation suite:

```bash
python -m eval.baseline_eval \
  --model checkpoints/condition_e/best \
  --condition E \
  --output results/
```

## Preliminary RL Workflow

The current lightweight preliminary runner uses the instruct checkpoint directly:

```bash
bash scripts/run_preliminary.sh
```

That script currently:

- trains `C`, `D`, and `E` sequentially
- uses `Qwen/Qwen3-4B`
- assumes a GPU is available
- expects `data/position_buffer.json` to exist

## Legacy SFT Path

Some repo scripts still point to an older SFT-first workflow. Those files are still useful if you want to inspect that path, but they are not the clearest representation of the current experiment description.

Files in that category include:

- `training/sft_data_gen.py`
- `training/sft_train.py`

Older per-condition shell wrappers and personal Lambda helper scripts have been removed from the repo. If they are ever needed again, they should be replaced with small, documented, credential-free wrappers around the current supported entry points.

## Project Layout

```text
connect4_opponent_modeling/
├── analysis/          # Plotting and correlation analysis
├── data/              # Benchmarks, locked probes, position buffers
├── docs/              # Bookkeeping and documentation notes
├── env/               # Connect Four env and solver wrapper
├── eval/              # Benchmark and probe runners
├── results/           # Saved summaries and evaluation artifacts
├── scripts/           # Training/eval convenience scripts
├── spiral/            # Main GRPO/SPIRAL training path
├── tests/             # Unit tests
└── training/          # Prompts, reward definitions, legacy SFT path
```

## Notes For Future Cleanup

- Align the remaining SFT-first scripts with the current instruct-first study, or explicitly archive them.
- Prefer `python -m spiral.train` and `scripts/run_preliminary.sh` over ad hoc shell wrappers.
- Keep smoke-test artifacts separate from reportable experiment results.
- Avoid interpreting placeholder checkpoints or 3-step smoke runs as scientific evidence.
- Keep deployment and backup helpers out of the research repo unless they are generic, credential-free, and part of the supported workflow.
