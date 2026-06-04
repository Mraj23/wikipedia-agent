"""Build a supervised (prompt=x0, completion) dataset from graded rollouts.

Three modes (selected with --mode):

  rltf_sft     The main arm. From graded y1 (revised) + graded y0 (first
               attempt), keep a pair iff:
                 y1 correct AND (y0 incorrect OR len(y1) <= ratio*len(y0)).
               completion = y1. We train on x0 -> y1 (feedback DROPPED), so
               the model internalizes compression without ever seeing the
               critique at inference.

  sft_y1       Ablation: same y1 source, correctness filter ONLY (no length
               gate). Isolates the effect of the length/compression filter.

  sft_caveman  Baseline: the model's own SHORTEST correct first attempt y0
               per question. completion = y0. No judge, no revision.

NOTE ON FAITHFULNESS: this is upstream RLTF's *SFT* distillation mode
(correctness-filtered cross-entropy on the improved attempt), NOT RLTF-SD.
True RLTF-SD is an advantage-weighted importance-sampling objective trained
jointly with multi-turn GRPO; see ../FAITHFULNESS.md. Length is measured on
the full output (parser-independent), matching the eval's primary axis.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def _n_tokens(tok, text):
    return len(tok.encode(text or "", add_special_tokens=False))


def build_rltf_sft(rows_y1, y0_idx, tok, length_ratio):
    kept, rejected = [], defaultdict(int)
    for r1 in rows_y1:
        key = (r1["id"], r1["sample_idx"])
        r0 = y0_idx.get(key)
        if not r0:
            rejected["no_y0_pair"] += 1
            continue
        if not bool(r1.get("correct")):
            rejected["y1_incorrect"] += 1
            continue
        y0_correct = bool(r0.get("correct"))
        y0_tokens = _n_tokens(tok, r0.get("raw_output", ""))
        y1_tokens = _n_tokens(tok, r1.get("raw_output", ""))
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
                "output_tokens_y0": y0_tokens,
                "output_tokens_y1": y1_tokens,
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


def build_sft_caveman(rows_y0, tok):
    """Keep the shortest CORRECT y0 per question id."""
    best = {}
    rejected = defaultdict(int)
    for r0 in rows_y0:
        if not bool(r0.get("correct")):
            rejected["y0_incorrect"] += 1
            continue
        n = _n_tokens(tok, r0.get("raw_output", ""))
        cur = best.get(r0["id"])
        if cur is None or n < cur[0]:
            best[r0["id"]] = (n, r0)
    kept = [
        {
            "id": r0["id"],
            "task": r0["task"],
            "prompt": r0["x0"],
            "completion": r0["raw_output"],
            "output_tokens": n,
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
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--length-ratio", type=float, default=0.8)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def load(path):
        return [json.loads(l) for l in open(path) if l.strip()]

    if args.mode == "rltf_sft":
        if not (args.y1_graded and args.y0_graded):
            raise SystemExit("rltf_sft needs --y1-graded and --y0-graded")
        y0_idx = {(r["id"], r["sample_idx"]): r for r in load(args.y0_graded)}
        kept, rejected = build_rltf_sft(
            load(args.y1_graded), y0_idx, tok, args.length_ratio
        )
    elif args.mode == "sft_y1":
        if not args.y1_graded:
            raise SystemExit("sft_y1 needs --y1-graded")
        kept, rejected = build_sft_y1(load(args.y1_graded))
    else:  # sft_caveman
        if not args.y0_graded:
            raise SystemExit("sft_caveman needs --y0-graded")
        kept, rejected = build_sft_caveman(load(args.y0_graded), tok)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for row in kept:
            fout.write(json.dumps(row) + "\n")

    print(f"mode={args.mode} kept={len(kept)} rejected={dict(rejected)} -> {out_path}")


if __name__ == "__main__":
    main()
