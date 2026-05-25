"""Combine (x0, y0, c0) into the revision prompt x1."""

import argparse
import json
from pathlib import Path

from prompts import build_x1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="c0.jsonl (output of feedback_judge)")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--require-feedback",
        action="store_true",
        default=True,
        help="drop rows with no c0 (default on)",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    with open(args.input) as fin, out.open("w") as fout:
        for line in fin:
            n_in += 1
            row = json.loads(line)
            if args.require_feedback and not row.get("c0"):
                continue
            row["y0"] = row["raw_output"]
            row["x1"] = build_x1(row["question"], row["y0"], row.get("c0") or "")
            row.pop("raw_output", None)
            fout.write(json.dumps(row) + "\n")
            n_out += 1
    print(f"wrote {n_out}/{n_in} -> {out}")


if __name__ == "__main__":
    main()
