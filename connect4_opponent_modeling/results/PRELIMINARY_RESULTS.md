# Preliminary Results: Connect Four Opponent Modeling

**Last Updated:** April 30, 2026
**Model:** Qwen3-4B (instruct)

---

## Project Status

### What's Done
- [x] Evaluation pipeline: difficulty ladder across 4 games (Connect Four, Breakthrough, Nim, TicTacToe)
- [x] Baseline calibration: untrained model performance measured at all difficulty levels
- [x] Prompt comparison: confirmed no effect from opponent-modeling prompts alone
- [x] Prompt engineering: `/no_think` + `<reasoning>/<answer>` format producing concise strategic reasoning
- [x] Clean board representation for Connect Four
- [x] Lenient move parsing for all games
- [x] GTBench integration with local HF model adapter
- [x] Training infrastructure: GRPO with entropy bonus, balanced position buffer, collapse detection, vLLM
- [x] W&B monitoring integration

### What's Not Done
- [ ] Successful RL training run (previous attempts collapsed to "always play center")
- [ ] Conditions D and E (the actual experiment)
- [ ] Transfer evaluation after training
- [ ] Mechanistic probe (neutral prompt opponent prediction test)
- [ ] GSM8K/MATH-500 non-adversarial control

---

## Key Results

### 1. Baseline Game Performance (Untrained Model)

| Opponent | Connect Four | Breakthrough | Nim | TicTacToe |
|---|---|---|---|---|
| Random | 70% | **90%** | 40% | 40% |
| Minimax-1 | 10% | **60%** | 20% | 20% |
| Minimax-2 | 0% | **50%** | 10% | 0% |
| Minimax-4 | 0% | **40%** | 0% | 0% |
| MCTS-100 | 0% | **10%** | 0% | 0% |

Breakthrough has the smoothest gradient and most room for improvement.

### 2. Prompt-Only Opponent Modeling Has No Effect

Base prompt: 20% win rate vs Minimax-1 (20 games)
Opponent-modeling prompt: 15% win rate vs Minimax-1 (20 games)

Prompting the model to consider opponent responses doesn't help when thinking is suppressed. This validates condition F as a real baseline.

### 3. Training Collapsed to Mode Exploitation

Condition C (value-only RL) trained for 500 steps. Reward climbed from 0.2 → 0.67, but the model converged to "always play column 3." Pons benchmark: 0%.

Root causes: no entropy regularization, biased position buffer (35% column-3-optimal), no collapse monitoring.

Fixes implemented (untested on GPU): entropy bonus, column-balanced buffer (14% per column), dynamic temperature, diversity monitoring in W&B.

---

## Training Infrastructure Status

| Component | Status | Notes |
|---|---|---|
| GRPO trainer | Ready | Entropy bonus, 8-bit AdamW, gradient checkpointing |
| vLLM generation | Ready | CPU offloading, ~55s/step with group_size=64 |
| Position buffer | Ready | Column-balanced, 1000 positions, cached to JSON |
| W&B monitoring | Ready | Loss, KL, rewards, unique_moves, most_common_pct, temperature |
| Collapse detection | Ready | Warns when >80% play same column, auto-increases temperature |
| Prompt format | Ready | `/no_think` + `<reasoning>/<answer>` tags |
| Evaluation | Ready | Difficulty ladder, 4 games, detailed JSON logs |
| Mechanistic probe | Not built | Needs neutral prompt opponent prediction evaluation |

---

## Costs to Date

| Run | Hardware | Time | Cost |
|---|---|---|---|
| GH200 training attempt | 1x GH200 96GB | ~3 hrs | ~$7 |
| H100 training (collapsed) | 1x H100 80GB | ~9 hrs | ~$39 |
| A100 calibration | 1x A100 40GB | ~4 hrs | ~$5 |
| **Total** | | | **~$51** |

---

## Next Steps

1. **Run training with all fixes** on 80GB GPU (entropy, balanced buffer, new prompt format)
2. **Verify no collapse** within first 50 steps via W&B monitoring
3. **Run conditions C, D, E** sequentially (500 steps each)
4. **Re-evaluate on difficulty ladder** — compare C vs D vs E on Breakthrough and Connect Four
5. **Build mechanistic probe** — neutral prompt opponent prediction test
6. **Run non-adversarial controls** — GSM8K, MATH-500

---

## Files Reference

| File | Purpose |
|---|---|
| `scripts/calibrate_transfer.py` | Difficulty ladder evaluation |
| `scripts/run_preliminary.sh` | Chain C → D → E training |
| `scripts/lambda_setup.sh` | Instance setup |
| `training/prompts.py` | `/no_think` + `<reasoning>/<answer>` prompt templates |
| `training/grpo_config.py` | Hyperparameters (entropy_coef=0.01, group_size=64, lr=1e-6) |
| `spiral/grpo_trainer.py` | GRPO training loop with vLLM, monitoring |
| `data/position_buffer.json` | Column-balanced 1000 positions |
| `results/EVALUATION_BASELINES.md` | Full baseline data and methodology |
| `results/calibration_v4/` | Detailed per-move JSON logs |
| `CLAUDE.md` | Experiment design and critical invariants |
