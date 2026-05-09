"""Score the preflight records against the ground-truth verifier.

For each board: compute exhaustive ground-truth tactical sets, compare to
what the model emitted, and report:

- fraction of boards where ground truth is *fully empty* (no tactics exist)
- fraction of boards where the model's emitted sets exactly match ground truth
- per-field false-negative rate (ground truth non-empty, model emitted empty)
- per-field false-positive rate (model emitted something, ground truth empty)

Tells us whether the all-empty pattern in the preflight is the model
correctly describing quiet positions, or under-claiming on tactical ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from env.connect_four_env import ConnectFourEnv
from faithfulness.parse import parse_structured_response
from faithfulness.verifier.claim_verifier import ground_truth_tactical_claims


FIELDS = (
    "self_immediate_win_columns",
    "opponent_immediate_win_columns",
    "unsafe_moves",
    "self_double_threat_moves",
    "self_single_threat_moves",
)


def _env_from_moves(moves):
    env = ConnectFourEnv()
    for m in moves:
        env.make_move(int(m))
    return env


def _is_empty(field_name: str, value) -> bool:
    return len(value) == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = [json.loads(l) for l in Path(args.records).read_text().splitlines() if l.strip()]
    n = len(rows)

    # Per-board flags
    n_gt_all_empty = 0
    n_model_all_empty = 0
    n_exact_match = 0

    # Per-field counts
    field_stats = {
        f: {"gt_nonempty": 0, "model_nonempty": 0, "fn": 0, "fp": 0, "exact": 0}
        for f in FIELDS
    }

    for row in rows:
        moves = row["moves"]
        env = _env_from_moves(moves)
        gt = ground_truth_tactical_claims(env)

        # Re-parse the completion to get the structured tactical_claims dict.
        completion = row.get("completion", "")
        parsed = parse_structured_response(completion, condition="tactical_claims")
        # The Claim objects carry the typed fields; pull out by claim type.
        model_sets = {f: None for f in FIELDS}
        from faithfulness.claims import CLAIM_TYPE_TO_TACTICAL_FIELD, ClaimType

        for claim in parsed.claims:
            field_name = CLAIM_TYPE_TO_TACTICAL_FIELD.get(claim.type)
            if field_name is None:
                continue
            if claim.type is ClaimType.SET_UNSAFE_MOVES:
                model_sets[field_name] = list(claim.fields.get("entries", []))
            else:
                model_sets[field_name] = sorted(claim.fields.get("values", []))

        gt_all_empty = all(_is_empty(f, gt[f]) for f in FIELDS)
        model_all_empty = all(
            _is_empty(f, model_sets[f]) for f in FIELDS if model_sets[f] is not None
        )
        if gt_all_empty:
            n_gt_all_empty += 1
        if model_all_empty:
            n_model_all_empty += 1

        all_match = True
        for f in FIELDS:
            gt_v = gt[f]
            mv = model_sets[f]
            gt_nonempty = not _is_empty(f, gt_v)
            mv_nonempty = (mv is not None) and (not _is_empty(f, mv))
            if gt_nonempty:
                field_stats[f]["gt_nonempty"] += 1
            if mv_nonempty:
                field_stats[f]["model_nonempty"] += 1

            if f == "unsafe_moves":
                # Compare as sets of (move, tuple(sorted_replies))
                def _norm(seq):
                    out = set()
                    for entry in seq:
                        out.add((entry["move"], tuple(sorted(entry.get("opponent_replies", [])))))
                    return out

                gt_norm = _norm(gt_v) if gt_v else set()
                mv_norm = _norm(mv) if mv else set()
                exact = gt_norm == mv_norm
            else:
                exact = sorted(gt_v) == (mv or [])

            if exact:
                field_stats[f]["exact"] += 1
            else:
                all_match = False
            if gt_nonempty and not mv_nonempty:
                field_stats[f]["fn"] += 1
            if mv_nonempty and not gt_nonempty:
                field_stats[f]["fp"] += 1
        if all_match:
            n_exact_match += 1

    summary = {
        "n_boards": n,
        "ground_truth_all_empty_rate": n_gt_all_empty / n if n else 0.0,
        "model_all_empty_rate": n_model_all_empty / n if n else 0.0,
        "exact_match_rate": n_exact_match / n if n else 0.0,
        "by_field": {
            f: {
                "gt_nonempty_rate": s["gt_nonempty"] / n if n else 0.0,
                "model_nonempty_rate": s["model_nonempty"] / n if n else 0.0,
                "exact_rate": s["exact"] / n if n else 0.0,
                "false_negative_rate_given_gt": (
                    s["fn"] / s["gt_nonempty"] if s["gt_nonempty"] else None
                ),
                "false_positive_rate_given_gt_empty": (
                    s["fp"] / max(1, n - s["gt_nonempty"]) if (n - s["gt_nonempty"]) else None
                ),
            }
            for f, s in field_stats.items()
        },
    }

    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
