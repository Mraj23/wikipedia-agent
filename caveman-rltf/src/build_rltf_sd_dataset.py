"""Filter graded (x0, y0, y1) tuples into RLTF-SD training pairs.

Keep iff:
  y1 is correct AND (y0 is incorrect OR reasoning_tokens(y1) <= ratio *
  reasoning_tokens(y0)).

Each kept row becomes a (prompt=x0, completion=y1) training pair. We
deliberately train on x0 -> y1 (NOT x1 -> y1) so the model internalizes
the compression instruction without seeing feedback at inference (plan §10).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="y1_graded.jsonl")
    parser.add_argument(
        "--y0-graded",
        required=True,
        help="y0_graded.jsonl, used to look up r0 + reasoning_tokens(y0)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--length-ratio",
        type=float,
        default=0.8,
        help="when y0 already correct, keep y1 only if reasoning <= ratio * y0",
    )
    parser.add_argument(
        "--random-feedback",
        action="store_true",
        help="ablation: keep filtering rules but cite c0 as random across rows "
        "(no effect here since we train on x0; for completeness, marker only)",
    )
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    y0_idx = {}
    with open(args.y0_graded) as f:
        for line in f:
            r = json.loads(line)
            y0_idx[(r["id"], r["sample_idx"])] = r

    kept = 0
    rejected = defaultdict(int)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input) as fin, out_path.open("w") as fout:
        for line in fin:
            r1 = json.loads(line)
            key = (r1["id"], r1["sample_idx"])
            r0 = y0_idx.get(key)
            if not r0:
                rejected["no_y0_pair"] += 1
                continue

            y1_correct = bool(r1.get("correct"))
            y0_correct = bool(r0.get("correct"))

            if not y1_correct:
                rejected["y1_incorrect"] += 1
                continue

            y0_tokens = len(
                tok.encode(r0.get("reasoning_text", "") or "", add_special_tokens=False)
            )
            y1_tokens = len(
                tok.encode(r1.get("reasoning_text", "") or "", add_special_tokens=False)
            )

            if y0_correct and y1_tokens > args.length_ratio * max(y0_tokens, 1):
                rejected["y1_not_shorter"] += 1
                continue

            out = {
                "id": r1["id"],
                "sample_idx": r1["sample_idx"],
                "y1_sample_idx": r1["y1_sample_idx"],
                "task": r1["task"],
                "prompt": r1["x0"],
                "completion": r1["raw_output"],
                "y0_text": r0.get("raw_output", ""),
                "y0_correct": y0_correct,
                "y1_correct": y1_correct,
                "reasoning_tokens_y0": y0_tokens,
                "reasoning_tokens_y1": y1_tokens,
                "weight": 1.0,
                "random_feedback_ablation": bool(args.random_feedback),
            }
            fout.write(json.dumps(out) + "\n")
            kept += 1

    print(f"kept: {kept}")
    print(f"rejected: {dict(rejected)}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
