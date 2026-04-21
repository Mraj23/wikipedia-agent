# CLAUDE.md — Connect Four Opponent Modeling Study

## Research Question

Does explicitly training an LLM to predict its opponent's next move — as a required reasoning step with gradient signal — develop adversarial reasoning that transfers to unseen adversarial tasks, and does this happen through genuine internal representation change rather than output formatting?

The hypothesis is not that opponent modeling training improves all reasoning. It's that it improves *adversarial* reasoning more than general reasoning, and more than either value-only RL or inference-time prompting alone produce. That specificity is the contribution.

## Training Domain

**Connect Four.** Chosen because it's strongly solved (Pons solver = perfect oracle, no neural approximation), hard enough that LLMs currently fail against MCTS (GTBench baseline), fast enough for large-scale RL (20-25 moves/game), and untouched by prior LLM training work.

**Base model:** Qwen3-4B (instruct). Instruct model used because the field universally uses instruct models for game evaluation (GTBench, Chess-R1, LLM Chess benchmark). All conditions start from the same instruct checkpoint.

## The Causal Ladder

Six conditions where each step adds exactly one cognitive operation. All conditions use the same static solver opponent (Pons/minimax) to keep the opponent constant across conditions.

| Label | Name | What it adds | Opponent |
|---|---|---|---|
| A | Instruct baseline | No RL — pre-existing instruct capabilities | N/A |
| B | Sparse RL | Optimization pressure from win/loss signal | Static solver |
| C | Solver-RL (value) | Position evaluation — "what is good" | Static solver |
| D | Solver-RL + future-state | Forward projection — "what happens after I act" | Static solver |
| E | Solver-RL + opponent modeling | Adversarial projection — "what happens after they respond" | Static solver |
| F | Prompt-only (ReTA baseline) | Inference-time opponent reasoning with no RL training | Static solver |

### Key Comparisons

- **C vs D** = does future-state reasoning add to value evaluation?
- **D vs E** = does opponent modeling add to future-state reasoning? ← **the central claim**
- **E vs F** = does training add anything beyond sophisticated prompting? ← the anti-ReTA test

### Why Each Condition Matters

- **D is the critical control.** Without it, a reviewer says "E is better because it has richer output format." D has equally rich output (predicts future state) but doesn't model the opponent. If E > D on transfer, it's the opponent modeling, not the format.
- **F eliminates the ReTA objection.** If E > F, training produces something prompting alone cannot.
- **The mechanistic probe** eliminates the "output formatting" objection. After training, all conditions are evaluated with a neutral prompt (no `<opponent_prediction>` tag). If E predicts opponents better than D on this neutral prompt, representations genuinely changed.

## Reward Structure

Format compliance is a binary gate (invalid format → reward 0), not a weighted component. This prevents zero-variance RAE explosions with instruct models.

| Condition | move_quality | terminal | future_state | prediction |
|---|---|---|---|---|
| B | — | sparse (win=1, loss=0, draw=0.5) | — | — |
| C | 0.67 | 0.33 | — | — |
| D | 0.56 | 0.22 | 0.22 | — |
| E | 0.56 | 0.22 | — | 0.22 |

Prediction accuracy returns 0.0 on terminal positions — the model only gets credit for actually modeling non-terminal opponent responses.

## Evaluation Suite

**Primary transfer (adversarial):** GTBench Breakthrough and Nim vs MCTS — complete deterministic games the model was never trained on. If E > D here, opponent modeling specifically drives adversarial transfer.

**Secondary transfer:** GameBench (Santorini, Hive, Sea Battle) — different adversarial structure, tests whether transfer generalizes.

**Non-adversarial control:** GSM8K and MATH-500 — if E ≈ D here while E > D on GTBench, the transfer is adversarially specific, not generic RL benefit.

**In-domain sanity check:** Pons benchmark accuracy and win rate vs minimax — confirms training worked.

## The Mechanistic Probe

After training all conditions, every model is asked to predict opponent responses using a neutral prompt that omits the `<opponent_prediction>` tag. If Condition E shows higher accuracy than D on this probe, training changed internal representations. If they're the same, it was just output formatting.

Probe positions are locked before any RL training begins and never regenerated.

A Pearson correlation across conditions and training checkpoints between probe accuracy and GTBench win rate tests: does internal opponent modeling predict adversarial transfer?

## Critical Invariants

1. **`data/probe_positions_locked.jsonl`** is written once and NEVER regenerated. `lock_probe_positions()` raises `FileExistsError` if the file already exists.
2. **All RL conditions** must start from the exact same instruct checkpoint.
3. **Board convention:** row 0 = top, row 5 = floor. OpenSpiel native.
4. **The game engine is OpenSpiel** (`pyspiel.load_game("connect_four")`). Do not reimplement game logic.
5. **Never compare results across different base models.**
6. **Never change reward weights mid-experiment.**

## Phase Sequence

- **Phase 1:** Instruct model baseline evals (Condition A)
- **Phase 2:** Lock probe positions (ONCE, never regenerate)
- **Phase 3:** RL conditions B-E
- **Phase 4:** Condition F eval (inference-time prompting)
- **Phase 5:** Mechanistic analysis (probes, correlation, transfer evals)
- **Phase 6:** Paper writing

## Where Things Live

| Artifact | Location |
|---|---|
| Model checkpoints | `checkpoints/` |
| Evaluation results | `results/` |
| Training logs | `logs/` |
| Locked probe positions | `data/probe_positions_locked.jsonl` |
| Position buffer (pre-generated) | `data/position_buffer.json` |
| Pons benchmark files | `data/pons_benchmark/` |
| Generated figures | `results/figures/` |

## Running Training

```bash
cd connect4_opponent_modeling

# Single condition (downloads Qwen3-4B from HF on first run)
python -m spiral.train --condition E --game_steps 500 --group_size 64 --wandb

# Preliminary run (C, D, E sequentially)
bash scripts/run_preliminary.sh
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Dependencies

- `open_spiel` — game engine (required)
- `torch`, `transformers` — model training/inference
- `wandb` — experiment monitoring
- `bitsandbytes` — 8-bit optimizer for GPU memory
- `datasets` — benchmark loading
- `tqdm`, `matplotlib`, `pandas`, `scipy` — utilities and analysis
- GTBench (optional) — `git clone https://github.com/jinhaoduan/GTBench`
- GameBench (optional) — `https://github.com/Costarelli/GameBench`

## Relationship to Prior Work

This experiment is motivated by SPIRAL (arxiv 2506.24119) which showed game self-play improves general reasoning. However, SPIRAL's transfer claim has been critiqued (Eugene's RAE ablation analysis). Our mechanistic probe directly addresses this weakness — it tests whether training actually changes internal representations, rather than relying on downstream benchmark correlations that may be noise.
