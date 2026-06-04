"""Combine (x0, y0, c0) into the revision prompt x1.

Ablation modes (these decide what feedback, if any, conditions the revision
y1 — which is where feedback actually has its effect, since training is on
x0 -> y1 with the feedback dropped):

  default        : x1 = question + y0 + the matched critique c0
  --shuffle-feedback : x1 uses a critique sampled from a DIFFERENT row. If y1
                   still improves as much as with matched feedback, the
                   critique CONTENT does not matter (only the revise-shorter
                   pressure does). This is the real random-feedback control.
  --no-feedback  : x1 = question + y0 + "revise shorter" (no critique at all).
                   Lower bound = pure rejection sampling of a terser attempt.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from prompts import build_x1, build_x1_no_feedback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="c0.jsonl (feedback_judge output)")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="build x1 without any critique (rejection-sampling control)",
    )
    parser.add_argument(
        "--shuffle-feedback",
        action="store_true",
        help="pair each row with another row's critique (random-feedback control)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.no_feedback and args.shuffle_feedback:
        raise SystemExit("--no-feedback and --shuffle-feedback are mutually exclusive")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.input) if l.strip()]

    if args.no_feedback:
        kept = []
        for row in rows:
            row["y0"] = row["raw_output"]
            row["x1"] = build_x1_no_feedback(row["question"], row["y0"])
            row["x1_mode"] = "no_feedback"
            row.pop("raw_output", None)
            kept.append(row)
    else:
        # Need a real critique for matched + shuffled modes.
        rows = [r for r in rows if r.get("c0")]
        critiques = [r["c0"] for r in rows]
        if args.shuffle_feedback and len(critiques) > 1:
            rng = random.Random(args.seed)
            shifted = critiques[1:] + critiques[:1]  # derangement for n>1
            rng.shuffle(shifted)
            # ensure no row keeps its own critique
            for i in range(len(shifted)):
                if shifted[i] is critiques[i]:
                    shifted[i], shifted[(i + 1) % len(shifted)] = (
                        shifted[(i + 1) % len(shifted)],
                        shifted[i],
                    )
            critiques = shifted
        kept = []
        for row, c0 in zip(rows, critiques):
            row["y0"] = row["raw_output"]
            row["x1"] = build_x1(row["question"], row["y0"], c0)
            row["x1_mode"] = "shuffled_feedback" if args.shuffle_feedback else "feedback"
            row.pop("raw_output", None)
            kept.append(row)

    with out.open("w") as fout:
        for row in kept:
            fout.write(json.dumps(row) + "\n")
    print(f"wrote {len(kept)}/{len(rows)} -> {out}")


if __name__ == "__main__":
    main()
