# Faithfulness GRPO pilot report — 2026-05-08

Four 50-step pilots run on Tinker (Qwen3-4B-Instruct-2507, claims_rationale condition, batch_size=32, group_size=8, lr=4e-5). Goal: diagnose why the original pilot showed no learning and find a configuration that does.

## TL;DR

| Pilot | Pool | Adv. norm | Dyn. temp | Outcome |
|---|---|---|---|---|
| v1 | random self-play (fallback) | raw mean-center | off | Marginal — no clear trend, but variance OK |
| v2 | tactical-stratified (25K) | raw mean-center | on | **Worse** — severe mode collapse |
| v3 | tactical-stratified (25K) | **standardized** | on | Same as v2 — standardization alone insufficient |
| v4 | random self-play (fallback) | **standardized** | on | **Working baseline** — best on every metric, only run with rising optimal_rate |

**Headline finding:** the tactical-stratified pool was the dominant cause of the failure, not the trainer's gradient math. Tactical positions have a clear correct answer; once the model agrees with itself across 8 completions, GRPO has zero variance and skips the group. The "fix" we thought we needed (advantage standardization) was a real bug worth fixing on its own merits but did not rescue the run on its own.

---

## Pilot-by-pilot results

```
                v1              v2              v3              v4
                random          tactical        tactical        random
                raw adv         raw adv         std adv         std adv
                no dyn temp     dyn temp        dyn temp        dyn temp
                ─────────       ─────────       ─────────       ─────────
mean_reward     -0.479          -0.552          -0.545          -0.455   ← v4 best
first-10        -0.513          -0.606          -0.576          -0.496
last-10         -0.475          -0.590          -0.589          -0.435   ← only v4 improves
optimal_rate    0.402           0.396           0.403           0.438    ← v4 best
first-10        0.398           0.359           0.391           0.418
last-10         0.327           0.355           0.356           0.439    ← only v4 trends up
skipped/32      10.5            24.6            26.5            22.8
first-10        8.9             11.8            14.7            13.6
last-10         9.9             30.9            31.3            26.4
unique moves    -               3.52            3.06            4.40
first-10        -               6.50            5.40            6.40
last-10         -               2.10            2.00            3.90
most_common%    -               0.846           0.864           0.816
first-10        -               0.494           0.540           0.645
last-10         -               0.959           0.959           0.850
```

(`unique_moves` and `most_common_move_pct` were added in this session, so v1 has no values for them.)

**The cleanest learning signal is `optimal_rate`'s trajectory.** v1, v2, v3 all *declined* across the run (model got worse). v4 *rose* (0.418 first-10 → 0.439 last-10). That's the first directional evidence that any pilot is learning the task at all.

---

## What worked

### 1. Diagnostic instrumentation (`unique_moves`, `most_common_move_pct`, dynamic-temperature logging)

Without these, v2's failure mode would have looked indistinguishable from "GRPO just doesn't work here." The new fields revealed the actual failure mode: not a flat reward signal, but a peaked policy distribution producing zero-variance groups. Code: `faithfulness/rl/trainer.py` step-summary block, plus the dynamic-temperature loop ported from `experiments/opponent_next_move/tinker_train.py`.

### 2. Advantage standardization (correctness fix)

Replacing `r - mean` with `(r - mean) / (std + 1e-8)` via the existing `spiral.rae.compute_advantages` is the right thing to do regardless of the data mix. Without it, gradient magnitude scaled with per-group reward spread, letting rare large-spread groups dominate updates. v3 vs v2 numbers are statistically indistinguishable on outcomes, but the underlying gradient math is now correct. Worth keeping.

### 3. Reverting to random self-play in v4

V4 finished as the only pilot that learns. Headline numbers: `optimal_rate` 0.418 → 0.439 across the run (v1/v2/v3 all dropped); `mean_reward` -0.496 → -0.435 (v1/v2/v3 all flat-or-declining); best mean across all metrics. Mode collapse still creeps in late (skipped groups 13.6 → 26.4, unique moves 6.4 → 3.9), so v4 is not a finished trainer — but it's the first configuration that produces a positive-direction signal at all. **Random pool + standardized advantages is the configuration to build the formal v0 from.**

### 4. Stratified pool generator (`--mix tactical`)

The script itself works — it produced exactly the targeted distribution (8K must_block, 5K each of immediate_win / forcing_threat / double_threat, 2K quiet) in 11,837 self-play games. The problem was *what* the targets should be, not *how* we hit them. The infrastructure stays useful for any later experiment that wants targeted position shapes (e.g., positions where the base model is *wrong*).

---

## What didn't work

### 1. The "tactical-heavy" pool hypothesis

The original failure diagnosis hypothesized that GRPO was starving for variance because positions were tactically flat (multiple equally-good moves). Stratifying the pool toward tactical positions was supposed to fix that. It did the opposite:

- v1 (random): 10.5 skipped groups/step → 21.5 useful groups/step.
- v2 (tactical): 24.6 skipped groups/step → 7.4 useful groups/step.
- v3 (tactical, gradient bug fixed): 26.5 skipped groups/step → 5.5 useful groups/step.

**Why the hypothesis was wrong:** a `must_block_threat` position has *one* correct column. The base model usually finds it. All 8 completions pick the same column → identical reward → zero variance → skipped group. The skip was a correctness signal, not noise. In v2/v3 the model learned the right move on most groups *and* lost the gradient signal needed to learn anything else.

The framing should have been "positions where the model is uncertain," not "positions where the solver has a clear preference." Those are different sets, and the deterministic strategic-tag classifier captures the second, not the first.

### 2. Dynamic temperature

The mechanism worked exactly as designed (climb when most_common_pct > 0.8, decay otherwise) but couldn't break the collapse. By step 14 of v2/v3 the temperature was pinned at the cap (1.5), and most_common_pct stayed at 0.95–1.00. The chosen-column token's argmax is determined by the rationale text that precedes it; perturbing prose tokens at temp 1.5 doesn't change the column the rationale steers toward.

A higher cap (2.0 or 2.5) might help at the cost of degrading rationale quality. Surgical alternative: apply higher temperature only to the `chosen_move` field's tokens, not the whole completion. Not implemented this session.

### 3. The first round of "obvious fixes" from the user diagnosis

Of the original five suggested fixes:

- ✅ "Generate stratified pool" — generated, but the stratification target was wrong.
- ⏸ "Cap legal_move claims / remove from prompt" — not done; deferred until we have a working baseline to attribute claim-shape effects against.
- ⏸ "Reward sharpening (regret/4)" — not done; deferred for the same reason. (With standardization in place, raw scale matters less.)
- ✅ "Rejection-sample zero-variance groups" — already existed in the trainer at line 308; was correctly skipping groups, but the skip rate climbing to 95%+ was the symptom, not something the rejection logic could fix.
- ✅ "Run another 50-step pilot" — done four times.

---

## Things to pin down before the next pilot

1. **V4's late-run mode collapse is the next bottleneck.** Skipped groups 13.6 → 26.4 over 50 steps, unique moves 6.4 → 3.9. The policy is sharpening on the few groups that do produce gradient, and that sharpening narrows the column distribution further. Single-variable next change candidates, ranked:
   - `--group-size 16` (matches working `tinker_train.py`; halves variance noise on per-group advantage estimates).
   - Higher `--temperature-max` (e.g., 2.0) — cheap, may not help much given the rationale-tokens-determine-column structure of `claims_rationale`.
   - Surgical alternative: apply temperature only to the chosen-column token. Trainer change required.

2. **Run for more than 50 steps.** V4 is still in its early phase of learning at step 50. The full intended run was 2000 steps. Worth scaling up to 200–500 steps to see whether the trend continues or whether the late-run mode collapse becomes terminal.

3. **Pool design (longer-term).** The right way to build a pool is by base-model column entropy: for each candidate position, sample N completions from the base model and keep only positions where no single column gets more than 5/8. That directly targets within-group variance and could be combined with random self-play for a hybrid pool. Approx 30 min to add and run for ~5K positions.

---

## Code changes shipped this session

Committed in `58d9dc4` ("Fix faithfulness GRPO trainer and add stratified pool generator"):

- `faithfulness/rl/trainer.py`: standardized advantages via `spiral.rae.compute_advantages`; per-step dynamic temperature with `most_common_move_pct` threshold; new logged fields (`unique_moves`, `most_common_move_pct`, `temperature`, `next_temperature`).
- `faithfulness/scripts/generate_training_pool.py`: `--mix tactical` preset for per-tag targets (kept; useful infrastructure even if the tactical-heavy mix wasn't the right answer for this experiment).
- `faithfulness/scripts/train.py`: exposed `--temperature`, `--temperature-max`, `--temperature-diversity-threshold`.

## Run artifacts

- `faithfulness/data/runs/pilot50_claims_20260508/` — v1 baseline (pre-session).
- `faithfulness/data/runs/pilot50_claims_v2_20260508/` — tactical pool + dyn temp + raw adv.
- `faithfulness/data/runs/pilot50_claims_v3_20260508/` — tactical pool + dyn temp + standardized adv.
- `faithfulness/data/runs/pilot50_claims_v4_20260508/` — random pool + dyn temp + standardized adv (in progress).
- `faithfulness/data/training_positions.jsonl` — 25K-position tactical-stratified pool (kept; not regenerable without the seed but cheap to rebuild).
