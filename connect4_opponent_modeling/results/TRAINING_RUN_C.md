# Training Run: Condition C (Value-Only RL)

**Date:** April 30 - May 1, 2026
**Model:** Qwen3-4B (instruct)
**Hardware:** Lambda Labs GH200 480GB
**Run ID:** exp_C_nothink_v2
**W&B:** https://wandb.ai/raj_deniz_josh_simrit/connect4-opponent-modeling

---

## Configuration

| Parameter | Value |
|---|---|
| Condition | C (value-only: move_quality + terminal) |
| Base model | Qwen/Qwen3-4B (instruct, no SFT) |
| Prompt format | `/no_think` + `<reasoning>/<answer>` tags |
| max_tokens | 512 |
| group_size | 64 |
| lr | 1e-6 |
| entropy_coef | 0.01 |
| clip_ratio | 0.2 |
| kl_coef | 0.001 |
| Reward weights | move=0.67, terminal=0.33 |
| Position buffer | 1000 positions, column-balanced |
| Generation | vLLM CPU offloading (~57s/step) |
| Board format | OpenSpiel native (lowercase, pieces at bottom, spaced with column labels) |
| Chat template | Applied via tokenizer.apply_chat_template() |

---

## Training Progress

| Step | Reward | Valid | Think Tokens | Unique Moves | KL | Loss | Temp | Collapses |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.325 | 35/64 (54%) | 286 | 5 | 0.000 | 0.0139 | 0.70 | 0 |
| 100 | 0.187 | 41/64 (64%) | 248 | 6 | 0.082 | 0.0056 | 0.70 | 0 |
| 200 | 0.021 | 45/64 (70%) | 213 | 7 | 0.124 | -0.0008 | 0.70 | 0 |
| 300 | 0.031 | 50/64 (78%) | 188 | 7 | 0.171 | -0.0030 | 0.70 | 0 |
| 400 | 0.254 | 50/64 (78%) | 191 | 7 | 0.208 | 0.0012 | 0.70 | 0 |

---

## Key Observations

### Model learns concise thinking
- Average thinking tokens dropped 33%: 286 → 191
- Valid completion rate rose from 54% → 78%
- RL is shaping the model to produce shorter, more focused reasoning that fits within the token budget
- This matches SPIRAL's finding that thinking length is shaped by reward signal

### Zero mode collapse
- 0 collapse warnings across 420+ steps
- Entropy bonus (0.01) + balanced position buffer + dynamic temperature are working
- Move diversity stable at 7 unique columns (out of 7 possible)
- Previous runs collapsed by step 12-200 — this is the first stable run

### Reward follows U-curve
- Starts at 0.325, dips to 0.021 (step 200), recovers to 0.254 (step 400)
- The dip is common in RL: the model reorganizes its policy before improving
- The recovery is a positive signal that learning is happening
- Need to see if reward continues climbing post-400

### KL divergence is controlled
- KL increases steadily: 0.000 → 0.208 over 400 steps
- Well below the 1.0 warning threshold
- Reference model is still providing meaningful regularization

---

## What's Different From Previous Failed Runs

| Issue | Previous Runs | This Run |
|---|---|---|
| Mode collapse | Collapsed by step 12-200 | **0 collapses in 420 steps** |
| Valid completions | 6/64 (9%) with native thinking | **50/64 (78%) with /no_think** |
| Entropy bonus | None | **0.01** |
| Position buffer | 35% column-3-optimal | **14% per column (balanced)** |
| Board format | Uppercase, pieces at top (wrong) | **Lowercase, pieces at bottom (correct)** |
| Chat template | Not applied (/no_think ignored) | **Applied via apply_chat_template()** |
| Thinking budget | 256 (too short) or 2048 (too long) | **512 (right-sized)** |
| Monitoring | Only reward logged | **valid_pct, think_tokens, unique_moves, temperature** |

---

## What's Still Running

- **Condition C:** ~80 steps remaining, finishes in ~1.5 hours
- **Condition E:** Auto-starts after C (chained via /tmp/run_e.sh, polls every 30 min)
- **Estimated total:** C done ~4:30 UTC, E done ~14:00 UTC May 1

---

## Next Steps After Training

1. Evaluate both C and E on difficulty ladder (Connect Four + Breakthrough)
2. Compare: does E (opponent modeling) improve more than C (value-only)?
3. Check W&B for final reward, valid_pct, thinking token trends
4. Run mechanistic probe if E shows improvement
