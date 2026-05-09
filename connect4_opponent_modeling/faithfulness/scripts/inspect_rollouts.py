"""Pretty-print rollouts from a faithfulness training run.

Two sources:
    train_log.jsonl  — one `sample` field per step (always written).
    rollouts.jsonl   — every rollout's full completion + reward + parsed
                       (only present in runs started after the rollout-dump
                       feature shipped).

Usage:
    python -m faithfulness.scripts.inspect_rollouts --run-dir <path>
    python -m faithfulness.scripts.inspect_rollouts --run-dir <path> --steps 0,5,10
    python -m faithfulness.scripts.inspect_rollouts --run-dir <path> \
        --rollouts --filter populated  # only show rollouts with non-empty claims
    python -m faithfulness.scripts.inspect_rollouts --run-dir <path> \
        --rollouts --filter optimal    # only show rollouts where the move was optimal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _is_populated_tactical(completion: str) -> bool:
    """Heuristic: tactical_claims completion has any non-empty array field."""
    if "tactical_claims" not in completion:
        return False
    # Any [non-empty] array under one of the 5 keys is enough.
    for key in (
        "self_immediate_win_columns",
        "opponent_immediate_win_columns",
        "unsafe_moves",
        "self_double_threat_moves",
        "self_single_threat_moves",
    ):
        marker_empty = f'"{key}": []'
        if marker_empty not in completion and f'"{key}":' in completion:
            return True
    return False


def _step_filter(allowed):
    if allowed is None:
        return lambda s: True
    return lambda s: s in allowed


def _print_train_log(run_dir: Path, step_filter):
    path = run_dir / "train_log.jsonl"
    if not path.exists():
        print(f"no train_log.jsonl in {run_dir}", file=sys.stderr)
        return
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        if not step_filter(r["step"]):
            continue
        print(
            f"=== step {r['step']:>3}  reward={r['mean_reward']:+.3f}"
            f"  optimal={r['optimal_rate']:.2f}"
            f"  unique_moves={r['unique_moves']}"
            f"  most_common={r['most_common_move_pct']:.2f}"
            f"  zero_var={r['zero_variance_groups']}"
            f"  ==="
        )
        print(r.get("sample", "<no sample>"))
        print()


def _print_rollouts(run_dir: Path, step_filter, mode):
    path = run_dir / "rollouts.jsonl"
    if not path.exists():
        print(f"no rollouts.jsonl in {run_dir} (run was started before per-rollout dump shipped)", file=sys.stderr)
        return

    n = 0
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if not step_filter(r["step"]):
                continue
            keep = True
            if mode == "populated":
                keep = _is_populated_tactical(r["completion"])
            elif mode == "optimal":
                keep = bool(r.get("optimal"))
            elif mode == "wrong":
                keep = r.get("legal") and not r.get("optimal")
            elif mode == "schema_invalid":
                keep = not r.get("schema_valid", True)
            if not keep:
                continue
            n += 1
            print(
                f"=== step={r['step']} chosen={r['chosen_move']} "
                f"reward={r['reward']:+.3f} regret={r['regret']:.2f} "
                f"legal={r['legal']} optimal={r['optimal']} "
                f"schema_valid={r['schema_valid']} ==="
            )
            print("moves:", r.get("moves"))
            print(r["completion"])
            print()
    print(f"# {n} rollouts matched filter={mode!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step numbers to include (default: all).",
    )
    parser.add_argument(
        "--rollouts",
        action="store_true",
        help="Read rollouts.jsonl instead of train_log.jsonl.",
    )
    parser.add_argument(
        "--filter",
        choices=("all", "populated", "optimal", "wrong", "schema_invalid"),
        default="all",
        help="Only used with --rollouts.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    allowed = (
        {int(s) for s in args.steps.split(",") if s.strip()}
        if args.steps
        else None
    )
    sf = _step_filter(allowed)

    if args.rollouts:
        _print_rollouts(run_dir, sf, args.filter)
    else:
        _print_train_log(run_dir, sf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
