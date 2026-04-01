# CLAUDE.md — Connect Four Opponent Modeling Study

## Project Purpose

Causal study of whether explicit opponent modeling during RL training develops transferable adversarial reasoning. Six experimental conditions (A-F) form a causal ladder, each adding one capability atop the previous.

**Research question:** Does training an LLM to model its opponent during Connect Four RL produce internal representations that transfer to novel adversarial games and non-game reasoning?

## The Causal Ladder

- **A: SFT only** — imitation baseline. Supervised fine-tuning on solver-optimal moves.
- **B: Self-play RL** — adds adversarial pressure via win/loss rewards.
- **C: Solver-RL (value)** — adds position evaluation via Pons solver rewards.
- **D: Solver-RL + future-state** — adds forward projection after own move.
- **E: Solver-RL + opponent modeling** — adds adversarial projection after opponent response.
- **F: Prompt-only baseline** — inference-time opponent reasoning with no RL training (ReTA comparison).

### Key Comparisons

- **C vs D** = value → future-state (does predicting consequences help?)
- **D vs E** = future-state → opponent modeling (does modeling the adversary help?)
- **E vs F** = training vs prompting (is RL-trained reasoning better than prompt-elicited?)

## Critical Invariants

**These must be enforced by any code written in this repository:**

1. **`data/probe_positions_locked.jsonl`** is written once in Phase 3 and NEVER regenerated. `lock_probe_positions()` raises `FileExistsError` if the file already exists — never bypass this.

2. **All RL conditions (B-E)** must start from the exact same SFT checkpoint at `checkpoints/condition_a/`.

3. **Board convention:** row 0 = top, row 5 = floor. OpenSpiel Connect Four uses this convention natively.

4. **The game engine is OpenSpiel** (`pyspiel.load_game("connect_four")`). Do not reimplement game logic.

5. **Never compare results across different base models.** All conditions use Qwen3-4B-Base.

## Phase Sequence

- **Phase 0:** Setup + eval pipeline (this repo)
- **Phase 1:** Base model baseline evals
- **Phase 2:** SFT warmup (Condition A)
- **Phase 3:** Lock probe positions (ONCE, never regenerate)
- **Phase 4:** RL conditions B-E + Condition F eval
- **Phase 5:** Mechanistic analysis (probes, correlation)
- **Phase 6:** Paper writing

## Do Not

- Skip the SFT warmup gate check (val accuracy must be 35-65%)
- Regenerate probe positions after Phase 3
- Run evals before Phase 1 baseline is recorded
- Change reward weights mid-experiment
- Use any game engine other than OpenSpiel
- Compare results across different base models

## Where Things Live

| Artifact | Location |
|---|---|
| Model checkpoints | `checkpoints/` |
| Evaluation results | `results/` |
| Training logs | `logs/` |
| Locked probe positions | `data/probe_positions_locked.jsonl` |
| SFT warmup data | `data/sft_warmup.jsonl` |
| SFT metadata | `data/sft_warmup_meta.json` |
| Pons benchmark files | `data/pons_benchmark/` |
| Generated figures | `results/figures/` |

## Reward Weights by Condition

| Condition | move_quality | terminal | format | future_state | prediction |
|---|---|---|---|---|---|
| B | — | sparse (win=1, loss=0, draw=0.5) | — | — | — |
| C | 0.6 | 0.3 | 0.1 | — | — |
| D | 0.5 | 0.2 | 0.1 | 0.2 | — |
| E | 0.5 | 0.2 | 0.1 | — | 0.2 |

## Running Tests

```bash
cd connect4_opponent_modeling
pip install open_spiel pytest
pytest tests/ -v
```

## Running Evaluations

```bash
# Single condition
python -m eval.baseline_eval --model checkpoints/condition_e --condition E --output results/

# All conditions
bash scripts/run_all_evals.sh
```

## Key Dependencies

- `open_spiel` — game engine (required)
- `torch`, `transformers` — model training/inference
- `datasets` — GSM8K and MATH benchmark loading
- `tqdm` — progress bars
- `matplotlib`, `pandas`, `scipy` — analysis and plotting
- GTBench (optional) — `git clone https://github.com/jinhaoduan/GTBench`
- GameBench (optional) — `https://github.com/Costarelli/GameBench`
