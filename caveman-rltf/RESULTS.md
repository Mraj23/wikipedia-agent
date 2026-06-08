# Results

## Pilot: scheduled length-penalty GRPO compresses thinking (Qwen3.5-4B, GSM8K)

Run `traj_4b_gsm8k_long` (80 steps, plain prompt, α-ramp 0→0.8 on total output
tokens, completed-within-budget reward). Held-out eval (n=16, GSM8K), greedy.

| step | α | completed-acc | total tokens | closed `</think>` |
|---|---|---|---|---|
| 0 | 0.00 | 69% | 2125 | 81% |
| 24 | 0.00 | 88% | 1941 | 88% |
| 40 | 0.23 | 94% | 1471 | 94% |
| 64 | 0.58 | 94% | 1504 | 94% |
| 80 | 0.80 | **94%** | **1408** | **94%** |

**Net: +25 pts completed-accuracy, −34% output tokens, closed-`</think>` stays
high (no reward hack).** Plot: `outputs/plots/traj_4b_gsm8k_long.png`. Cost ~$3.

Two effects, stated separately:
- Warmup (α=0, steps 0–24): correctness RL raises accuracy 69→88% (raw acc rose
  too — elicitation), tokens drift 2125→1941.
- Compression (α ramps, steps 32–80): tokens fall 1941→1408 while accuracy
  holds ~94%.

### Caveats (before any blog claim)
- n=16 eval is noisy (±~12%); endpoint needs validation on the full 100 held-out.
- Single seed.
- Easiest setup (4B / GSM8K). The harder "complete within budget" story is on
  date_understanding / logical_deduction with 35B-A3B (where base truncates ~35%).

## Lessons logged along the way
- **Reward hack:** penalizing tokens-before-`</think>` → model never closes the
  tag (counter reads 0) and rambles. Caught via completion-rate collapse
  (100%→15%). Fix: penalize TOTAL tokens + require closed-`</think>`+`Answer`.
- **Reward floor:** `max(0, 1-α·n/budget)` floored long-correct to ≈wrong → dead
  gradient (24-step null). Fix: `1-α·min(n,norm)/norm` (correct always > wrong).
- **Scale:** 20–24 steps showed nothing; the effect emerged at 80 steps.
- **Infra:** Tinker JWT expires mid-run → save_state + in-process auto-resume.
- **Model/task fit:** 4B is floored on date/logical (~25% acc) but usable on
  GSM8K (~69%→94%); 35B-A3B is usable on date/logical.
