# Bookkeeping Notes

This file exists to keep the repository's experiment status legible while the codebase is still evolving.

## Source Of Truth

Use these files in this order when deciding what the current experiment is:

1. `CLAUDE.md`
2. `results/PRELIMINARY_RESULTS.md`
3. `results/EVALUATION_BASELINES.md`
4. `README.md`

If another script or comment disagrees with those files, assume the script or comment is stale until verified.

## Reportable vs Non-Reportable Artifacts

Treat these as reportable today:

- Baseline calibration summaries in `results/EVALUATION_BASELINES.md`
- Baseline calibration logs in `results/calibration_v4/`
- Locked probe positions in `data/probe_positions_locked.jsonl`

Treat these as non-reportable pipeline artifacts unless they are explicitly rerun and documented:

- `logs/smoke_*`
- `logs/local_smoke*`
- `logs/rl_smoke`
- placeholder-model checkpoints
- 3-step or similarly tiny sanity-check runs

## Current Repo Reality

The repo currently mixes two workflows:

- A current instruct-first preliminary workflow centered on `scripts/run_preliminary.sh`
- An older SFT-first workflow still visible in modules such as `training/sft_train.py`

Both are useful engineering artifacts, but only the first one matches the current design narrative in `CLAUDE.md`.

The old personal Lambda provisioning and backup helpers have been removed. The stale per-condition shell wrappers have also been removed in favor of `python -m spiral.train` and `scripts/run_preliminary.sh`. These should not be reintroduced unless they are rewritten as thin, documented, credential-free wrappers around the supported flow.

## Naming Guidance

When saving new results, prefer names that make the run type obvious:

- `baseline_*` for reportable untuned evaluations
- `prelim_*` for real preliminary RL runs
- `smoke_*` for debugging-only runs
- `scratch_*` for one-off local experiments

## Before Adding New Results

Check these questions first:

- Is this a real experiment run or just a pipeline sanity check?
- Does the result depend on a placeholder model or tiny step count?
- Is the base model the same one used by the current study?
- Should this live under `results/` or only under `logs/`?

If the answer is ambiguous, add a short note near the artifact rather than letting the filename imply more confidence than the run deserves.
