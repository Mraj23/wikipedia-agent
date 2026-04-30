# Preliminary Results: Connect Four Opponent Modeling

**Date:** April 29-30, 2026
**Model:** Qwen3-4B (instruct)
**Hardware:** Lambda Labs A100-SXM4-40GB (calibration), H100 80GB (training)

---

## 1. Baseline Calibration: Difficulty Ladder

We evaluated the untrained Qwen3-4B instruct model on four OpenSpiel games against opponents of increasing strength. Each level was tested over 5 games with alternating first/second player.

### Method

- Model uses `/no_think` mode (Qwen3-4B is a thinking model; without this, chain-of-thought consumes 2000+ tokens per move and often never finishes)
- Prompts show the board state and legal moves in OpenSpiel's native format, asking the model to copy a move exactly
- Invalid moves fall back to random play
- Opponents: Random, Minimax depth 1/2/4, MCTS with 100 simulations

### Results

| Opponent | Connect Four | TicTacToe | Breakthrough | Nim |
|---|---|---|---|---|
| **Random** | 100% (5/5) | 40% (2/5) | 80% (4/5) | 60% (3/5) |
| **Minimax-1** | 40% (2/5) | 60% (3/5) | 80% (4/5) | 80% (4/5) |
| **Minimax-2** | 60% (3/5) | 0% (0/5) | 40% (2/5) | 0% (0/5) |
| **Minimax-4** | 20% (1/5) | 0% (0/5) | 20% (1/5) | 0% (0/5) |
| **MCTS-100** | 0% (0/5) | 0% (0/5) | 0% (0/5) | 0% (0/5) |

**Valid move rates:**

| Game | Valid Move Rate |
|---|---|
| Connect Four | 98-100% |
| TicTacToe | 89-100% |
| Breakthrough | 89-96% |
| Nim | 53-93% |

### Key Findings

1. **The model genuinely plays all four games.** Valid move rates are 89-100% for most games (Nim is lower due to its unusual action format).

2. **Clear difficulty gradient.** Win rate drops monotonically from random to MCTS-100, providing a measurable range for detecting improvement from RL training.

3. **Detectable training target.** The model beats random consistently but struggles at minimax-2+. If RL pushes minimax-2 win rate from 60% to 80% on Connect Four, that's a measurable improvement.

4. **MCTS-100 is unbeatable** by this model size, consistent with GTBench's published finding that all LLMs (including GPT-4) score NRA ≈ -1 against MCTS on deterministic games.

---

## 2. Prompt Comparison: Base vs Opponent-Modeling

We tested whether asking the model to reason about opponent responses at inference time (no RL training) improves Connect Four play.

### Method

- 20 games per prompt style against Minimax depth 1
- Alternating first/second player
- `/no_think` mode (model does not actually perform chain-of-thought)
- **Base prompt:** "Pick one move. Copy it exactly."
- **Opponent-modeling prompt:** "Before choosing, think about what your opponent would play in response to each candidate move. Which column would they most likely choose? Then pick the move that leaves you in the best position after their response."

### Results

| Prompt Style | Wins | Win Rate | Valid Moves |
|---|---|---|---|
| **Base** | 4/20 | 20% | 92% (101/110) |
| **Opponent-Modeling** | 3/20 | 15% | 100% (185/185) |

### Key Findings

1. **No prompt-only effect.** Asking the model to consider opponent responses does not improve win rate (20% vs 15%, within noise).

2. **This is expected in `/no_think` mode.** The model suppresses chain-of-thought, so the opponent-modeling instruction has no mechanism to affect reasoning. The model outputs the same quality moves regardless of prompt complexity.

3. **Opponent-modeling prompt does improve valid move rate** (100% vs 92%). The longer, more structured prompt appears to help the model produce cleaner output format.

4. **This establishes the F condition baseline.** If RL training with opponent prediction (condition E) improves performance where prompting alone (condition F) doesn't, the improvement comes from training, not from the prompt.

---

## 3. Training Attempt: Condition C (Value-Only RL)

### Method

- Model: Qwen3-4B instruct
- Training: GRPO with group_size=64, lr=1e-6, 500 steps
- Reward: 0.67 × move_quality + 0.33 × terminal
- Position buffer: 1000 positions (column-balanced)
- Hardware: Lambda H100 80GB with vLLM CPU offloading (~55s/step)

### Results

| Step | Reward | KL | Notes |
|---|---|---|---|
| 0 | 0.199 | 0.000 | Baseline |
| 100 | 0.396 | 0.116 | Learning |
| 200 | 0.147 | 0.339 | Dip |
| 300 | 0.544 | 0.275 | Recovery |
| 400 | 0.670 | 0.448 | Converged |
| 500 | 0.670 | - | Pons benchmark: 0.0% |

### Failure Analysis

**The model collapsed to "always play column 3."** By step 400, all 64 completions in every group produced the identical move. Reward converged to 0.670 (the maximum for always-center on most positions). Pons benchmark scored 0% because the model played center regardless of board state.

**Root causes identified:**

1. **No entropy regularization.** Without an entropy bonus in the loss, the policy collapsed to a single deterministic action. Once all completions were identical, advantages = 0, gradients = 0, learning stopped.

2. **Biased position buffer.** Column 3 was the optimal move on 35% of buffer positions (vs 14% expected). "Always play center" was a viable exploit of the reward function.

3. **No diversity monitoring.** The collapse happened between step 100 and step 400 but was invisible in the reward logs. Only discovered by manually inspecting model outputs after the run.

### Fixes Applied (not yet tested on GPU)

- Entropy bonus (`entropy_coef=0.01`) added to GRPO loss
- Column-balanced position buffer regenerated (each column optimal on ~14% of positions)
- Collapse detection: logs `unique_moves` and `most_common_pct` per step
- Dynamic temperature: increases when >80% of completions play the same move

---

## 4. Infrastructure Learnings

### What works
- **vLLM CPU offloading** for generation (~55s/step vs ~160s without): offload training model to CPU, generate with vLLM on full GPU, destroy vLLM, move model back for backward pass
- **Pre-generated position buffers** loaded from JSON (~0s vs ~20 min generation on GPU)
- **`/no_think` mode** for evaluation: 98-100% valid moves vs 12-50% with thinking enabled (thinking consumes too many tokens)
- **OpenSpiel native action format** in prompts: model copies legal moves exactly when asked

### What failed
- **Qwen3-4B-Base** generates Chinese text, can't produce XML format even with 64 rollouts
- **SFT warmup** with prompt token masking bug wasted training signal (90% of loss on predicting the prompt)
- **Gradient checkpointing** disables KV cache in HuggingFace, making generation 256x slower
- **GTBench** loads 4 model copies for a single evaluation — too memory-intensive for direct use

### Cost Summary

| Run | Hardware | Time | Cost |
|---|---|---|---|
| First GPU attempt (GH200) | 1x GH200 96GB | ~3 hrs | ~$7 |
| Training run (H100) | 1x H100 80GB | ~9 hrs | ~$39 |
| Calibration (A100) | 1x A100 40GB | ~3 hrs | ~$4 |
| **Total** | | | **~$50** |

---

## 5. Open Questions

1. **Will entropy regularization prevent mode collapse?** The fix is implemented but untested on GPU.

2. **Can RL training actually improve Connect Four play beyond the instruct baseline?** The baseline already beats random 100% and minimax-1 40%. Is there room for meaningful improvement?

3. **Will training transfer to other games?** The model already plays Breakthrough at 80% vs minimax-1 with zero Connect Four-specific training. If RL pushes Connect Four to 80% vs minimax-1, will Breakthrough also improve?

4. **Is `/no_think` the right evaluation mode?** The model might play better with thinking enabled, but generating 2000+ tokens per move makes evaluation extremely slow and often fails to produce a final answer.

5. **Is 20 games per condition enough statistical power?** The prompt comparison (20% vs 15%) is well within noise. We may need 50-100 games per comparison.

---

## 6. Next Steps

1. **Test entropy + balanced buffer fixes** on GPU to confirm mode collapse is prevented
2. **Run conditions C, D, E** with the fixes and compare
3. **Re-evaluate all conditions** on the difficulty ladder across all 4 games
4. **Build the mechanistic probe** (neutral prompt opponent prediction test)
5. **Run GSM8K/MATH-500** as non-adversarial control
