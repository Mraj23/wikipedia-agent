# faithfulness_v2 — clean restart

## Rule

This directory does not contain claims, schemas, KL-shaped rewards, dynamic temperature, or any of the regularizers from `faithfulness/`. It will not contain any of those things until move-only RL beats the base model by a meaningful margin on a balanced held-out eval.

"Meaningful margin" = mean clipped regret on the balanced eval drops by ≥0.10 vs the pinned base baseline, and held-out optimal_rate exceeds base on every checkpoint after the first ~30 steps. If we don't see that, we don't proceed.

## Files

- `generate_pool.py` — generate a Connect 4 training pool. Default `--mode random`: stratified random self-play (early/mid/late game) filtered by Pons solver-spread, no Tinker calls. Optional `--mode entropy` adds a base-model column-entropy filter on top (Tinker required, diagnostic).
- `train_move_only.py` — GRPO move-only RL trainer. One condition. One reward (clipped solver-regret, [0, 1]). No claims. No KL shaping. Group acceptance reads raw reward variance only.
- `eval_move_quality.py` — deterministic balanced eval, base vs checkpoint. Greedy mode (one sample, temperature=0) for the headline number; multi-sample mode for distribution-aware scoring.

## Future files (do not create until move-only beats base)

- `train_tactical_reporter.py` — reporter warmup with truth-only reward or SFT. Goal: get out of the all-empty-fields regime so action RL has a non-trivial trace to preserve.
- `eval_tactical_truth.py` — per-field truth/precision/recall/F1, plus causal-intervention metrics. Separate from action learning.

## Workflow

```bash
cd connect4_opponent_modeling

# 1. Pin the base baseline on the balanced held-out eval.
TINKER_API_KEY=... python -m faithfulness_v2.eval_move_quality \
    --eval-set faithfulness/data/eval_boards.jsonl \
    --run base:BASE \
    --mode greedy \
    --output faithfulness_v2/runs/base_greedy.json

# 2. Generate a stratified random training pool (no Tinker calls in default mode).
python -m faithfulness_v2.generate_pool \
    --output faithfulness_v2/data/pool_v2.jsonl \
    --target 5000

# 2b. (Optional diagnostic) regenerate with the base-model entropy filter on top.
# TINKER_API_KEY=... python -m faithfulness_v2.generate_pool \
#     --mode entropy \
#     --output faithfulness_v2/data/pool_v2_entropy.jsonl \
#     --target 5000 --candidates-per-target 4 --max-column-share 0.5

# 3. Train a move-only pilot.
TINKER_API_KEY=... python -m faithfulness_v2.train_move_only \
    --pool faithfulness_v2/data/pool_v2.jsonl \
    --eval-set faithfulness/data/eval_boards.jsonl \
    --steps 100 \
    --group-size 16 \
    --learning-rate 1e-5 \
    --eval-every 25 \
    --save-every 10 \
    --max-runtime-seconds 1200 \
    --output faithfulness_v2/runs/pilot1

# 3b. Resume if the chunk stops before Tinker JWT/session expiry.
# Use `state_path` from `checkpoint_paths.jsonl`, not `sampler_path`.
TINKER_API_KEY=... python -m faithfulness_v2.train_move_only \
    --pool faithfulness_v2/data/pool_v2.jsonl \
    --eval-set faithfulness/data/eval_boards.jsonl \
    --steps 50 \
    --start-step 51 \
    --resume-state 'tinker://.../weights/...' \
    --group-size 16 \
    --learning-rate 1e-5 \
    --eval-every 25 \
    --save-every 10 \
    --max-runtime-seconds 1200 \
    --output faithfulness_v2/runs/pilot1_resume

# 4. Evaluate the trained checkpoint vs base on the same eval set.
TINKER_API_KEY=... python -m faithfulness_v2.eval_move_quality \
    --eval-set faithfulness/data/eval_boards.jsonl \
    --run base:BASE \
    --run pilot1:tinker://path/from/pilot1/checkpoint_paths.json \
    --mode greedy \
    --output faithfulness_v2/runs/pilot1_compare.json
```

## Configuration constraints (non-negotiable)

1. **Group acceptance uses raw reward variance only.** No shaped reward gates anywhere. The variance check reads the unmodified solver-regret reward.
2. **KL is monitoring only.** When `--log-kl` is enabled, KL is computed and logged but never subtracted from reward.
3. **One reward.** `1.0 - clipped_regret/2.0` ∈ [0, 1]. Optimal=1.0, worst legal=0.0, illegal=0.0, invalid=0.0.
4. **GRPO batch shape: 1 position × N rollouts.** `positions_per_step=1`, `group_size=16` or `32`. No multi-position batches. No retry loops.
5. **Position pool comes from `generate_pool.py` only.** Not from `faithfulness/data/training_positions.jsonl` (that's the failed tactical-stratified pool). Pool generation is reproducible by seed.
6. **Eval is balanced and pinned.** `eval_move_quality.py` reads a fixed eval set; the base baseline is scored once and pinned for every checkpoint comparison.
7. **No reasoning prefix encoded into the prompt.** Move-only. Prompt asks for a single integer.

## JWT-safe chunking / resume

Tinker JWT/session failures are not a Connect Four failure mode. They happen when a long-running process outlives a short-lived auth token. The trainer avoids that by default with `--max-runtime-seconds 1200`: after roughly 20 minutes, it saves a resumable training-state checkpoint and exits cleanly before the token expires. This is intentionally conservative because individual Tinker futures can stall for many minutes.

The important distinction:

- `state_path`: written by Tinker `save_state`; use this with `--resume-state` to continue training with optimizer state.
- `sampler_path`: written by `save_weights_for_sampler`; use this for eval/inference only.

Every checkpoint record in `checkpoint_paths.jsonl` includes both paths when available. Scheduled checkpoints save `state_path` even when that step had zero reward variance, so a skipped checkpoint step is still resumable. For training continuation, copy the latest `state_path` from the most recent `"reason": "scheduled"`, `"time_limit"`, or `"final"` record and pass it to `--resume-state`. Keep `--start-step` at the next logical step so logs, checkpoint names, and the deterministic pool-choice stream continue correctly.

## Code-level guarantees

These five behaviours are enforced by the code, not by convention:

1. **Post-update sampler.** Held-out eval and checkpoint paths in `train_move_only.py` reflect the model state *after* the optim step on that interval, not before. If a step is skipped (raw reward variance ≈ 0), the rollout sampler is current — it's reused. The `eval_log.jsonl` and `checkpoint_paths.jsonl` records carry a `post_update` flag.
2. **Eval = full balanced set.** All 100 boards (5 strata × 20: immediate_win_available, opponent_immediate_threat, blunder_state, no_immediate_tactic, hard_midgame) are scored every eval, with no random subsampling. Smaller eval sets must be pre-stratified offline and passed via `--eval-set`.
3. **Default pool is stratified random + solver-spread.** `generate_pool.py --mode random` (default) produces a ply-balanced pool with no Tinker calls. `--mode entropy` adds base-model entropy filtering as a diagnostic on top.
4. **Strict Pons by default.** All three scripts fail loud if the Pons binary is unavailable. Dev-only opt-out: `--no-strict-solver`.
5. **Prompt parity is enforced.** `tests/test_faithfulness_v2_prompt_consistency.py` asserts `SYSTEM_PROMPT`, `_render_board`, `make_messages`, `parse_column`, and `_COLUMN_RE` are byte-identical across the three v2 scripts. Run before any pilot:
   ```
   pytest tests/test_faithfulness_v2_prompt_consistency.py -v
   ```

## What was wrong in v1

See `/Users/rajmehta/.claude/plans/parallel-wondering-turtle.md` for the failure inventory and the staged recovery plan v2 implements.
