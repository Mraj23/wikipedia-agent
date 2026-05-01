# Bookkeeping Notes

This file exists to keep the repository's experiment status legible while the codebase is still evolving.

## Source Of Truth

Use these files in this order when deciding what the current experiment is:

1. `CLAUDE.md`
2. `docs/ACTIVE_PROTOCOL.md`
3. `README.md`
4. canonical JSON outputs under `results/`

If another script or comment disagrees with those files, assume the script or comment is stale until verified.

## Reportable vs Non-Reportable Artifacts

Treat these as reportable today:

- Canonical evaluation JSONs emitted after the May 1, 2026 cleanup
- Locked probe positions in `data/probe_positions_locked.jsonl`

Treat these as non-reportable pipeline artifacts unless they are explicitly rerun and documented:

- `logs/smoke_*`
- `logs/local_smoke*`
- `logs/rl_smoke`
- placeholder-model checkpoints
- 3-step or similarly tiny sanity-check runs

## Current Repo Reality

The repo now supports one active workflow and one archived history:

- Active: instruct-first RL centered on `scripts/bootstrap_gpu.sh`, `scripts/run_preliminary.sh`, and the canonical eval suite
- Archived: older mixed-protocol work preserved under `archive/invalidated_2026_05_01/`

Only the active path matches the current design narrative in `CLAUDE.md`.

The repo should remain self-bootstrapping for a fresh GPU machine. If a step is required to train or evaluate and it is not encoded in `scripts/bootstrap_gpu.sh`, `scripts/gpu_env.sh`, or the canonical scripts, treat that as a repo bug.

## External Artifact Policy

- `7x6.book` is tracked in-repo for convenience, and the bootstrap script should restore it if absent.
- model weights should not be pushed to GitHub
- solver source should be rebuilt by script when absent
- reportable outputs should be small JSON artifacts, plots, or markdown generated from those JSONs

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
