"""Build a supervised (prompt=x0, completion) dataset from graded rollouts.

Three modes (selected with --mode):

  rltf_sft     The main arm. From graded y1 (revised) + graded y0 (first
               attempt), keep a pair iff:
                 y1 correct AND (y0 incorrect OR tokens(y1) <= ratio*tokens(y0)).
               completion = y1. We train on x0 -> y1 (feedback DROPPED), so
               the model internalizes compression without ever seeing the
               critique at inference.

  sft_y1       Ablation: same y1 source, correctness filter ONLY (no length
               gate). Isolates the effect of the length/compression filter.

  sft_caveman  Baseline: the model's own SHORTEST correct first attempt y0
               per question. completion = y0. No judge, no revision.

Length uses the exact generated token count (`n_gen_tokens`) recorded at
sampling time — for a thinking model this is dominated by the <think> block,
so the gate selects for shorter *thinking*, the primary compression target.

NOTE ON FAITHFULNESS: this is upstream RLTF's *SFT* distillation mode
(correctness-filtered cross-entropy on the improved attempt), NOT RLTF-SD.
See ../FAITHFULNESS.md.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _ntok(row):
    return int(row.get("n_gen_tokens", 0) or 0)


def build_rltf_sft(rows_y1, y0_idx, length_ratio):
    kept, rejected = [], defaultdict(int)
    for r1 in rows_y1:
        r0 = y0_idx.get((r1["id"], r1["sample_idx"]))
        if not r0:
            rejected["no_y0_pair"] += 1
            continue
        if not bool(r1.get("correct")):
            rejected["y1_incorrect"] += 1
            continue
        y0_correct = bool(r0.get("correct"))
        y0_tokens, y1_tokens = _ntok(r0), _ntok(r1)
        if y0_correct and y1_tokens > length_ratio * max(y0_tokens, 1):
            rejected["y1_not_shorter"] += 1
            continue
        kept.append(
            {
                "id": r1["id"],
                "task": r1["task"],
                "prompt": r1["x0"],
                "completion": r1["raw_output"],
                "y0_correct": y0_correct,
                "tokens_y0": y0_tokens,
                "tokens_y1": y1_tokens,
                "x1_mode": r1.get("x1_mode"),
                "feedback_mode": r1.get("feedback_mode"),
            }
        )
    return kept, rejected


def build_sft_y1(rows_y1):
    kept, rejected = [], defaultdict(int)
    for r1 in rows_y1:
        if not bool(r1.get("correct")):
            rejected["y1_incorrect"] += 1
            continue
        kept.append(
            {
                "id": r1["id"],
                "task": r1["task"],
                "prompt": r1["x0"],
                "completion": r1["raw_output"],
            }
        )
    return kept, rejected


def build_sft_caveman(rows_y0):
    """Keep the shortest CORRECT y0 per question id."""
    best = {}
    rejected = defaultdict(int)
    for r0 in rows_y0:
        if not bool(r0.get("correct")):
            rejected["y0_incorrect"] += 1
            continue
        n = _ntok(r0)
        cur = best.get(r0["id"])
        if cur is None or n < cur[0]:
            best[r0["id"]] = (n, r0)
    kept = [
        {
            "id": r0["id"],
            "task": r0["task"],
            "prompt": r0["x0"],
            "completion": r0["raw_output"],
            "tokens": n,
        }
        for n, r0 in best.values()
    ]
    return kept, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, choices=["rltf_sft", "sft_y1", "sft_caveman"]
    )
    parser.add_argument("--y1-graded", help="y1_graded.jsonl (rltf_sft, sft_y1)")
    parser.add_argument("--y0-graded", help="y0_graded.jsonl (rltf_sft, sft_caveman)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--length-ratio", type=float, default=0.8)
    args = parser.parse_args()

    def load(path):
        return [json.loads(l) for l in open(path) if l.strip()]

    if args.mode == "rltf_sft":
        if not (args.y1_graded and args.y0_graded):
            raise SystemExit("rltf_sft needs --y1-graded and --y0-graded")
        y0_idx = {(r["id"], r["sample_idx"]): r for r in load(args.y0_graded)}
        kept, rejected = build_rltf_sft(load(args.y1_graded), y0_idx, args.length_ratio)
    elif args.mode == "sft_y1":
        if not args.y1_graded:
            raise SystemExit("sft_y1 needs --y1-graded")
        kept, rejected = build_sft_y1(load(args.y1_graded))
    else:  # sft_caveman
        if not args.y0_graded:
            raise SystemExit("sft_caveman needs --y0-graded")
        kept, rejected = build_sft_caveman(load(args.y0_graded))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for row in kept:
            fout.write(json.dumps(row) + "\n")

    print(f"mode={args.mode} kept={len(kept)} rejected={dict(rejected)} -> {out_path}")


if __name__ == "__main__":
    main()
