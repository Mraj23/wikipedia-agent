"""Sample BBH Tracking Shuffled Objects and Logical Deduction into JSONL.

Fetches the per-task JSON from suzgunmirac/BIG-Bench-Hard on GitHub. We use
GitHub raw rather than `datasets.load_dataset("lukaemon/bbh", ...)` because
some execution environments block HF Hub egress; the BBH files are tiny
(~150 KB each) and the schema (`canary`, `examples: [{input, target}]`) is
stable.
"""

import argparse
import json
import random
import urllib.request
from pathlib import Path


TASK_MAP = {
    "tracking_shuffled_objects": "tracking_shuffled_objects_seven_objects",
    "logical_deduction": "logical_deduction_seven_objects",
}

BBH_URL = (
    "https://raw.githubusercontent.com/suzgunmirac/"
    "BIG-Bench-Hard/main/bbh/{name}.json"
)


def fetch_bbh(name: str):
    url = BBH_URL.format(name=name)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["examples"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        choices=list(TASK_MAP.keys()),
    )
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    for task in args.tasks:
        rows = fetch_bbh(TASK_MAP[task])
        rng.shuffle(rows)
        rows = rows[: args.n]

        out_path = out_dir / f"{task}.jsonl"
        with out_path.open("w") as f:
            for i, row in enumerate(rows):
                obj = {
                    "id": f"{task}_{i:04d}",
                    "task": task,
                    "question": row["input"],
                    "gold": row["target"],
                }
                f.write(json.dumps(obj) + "\n")
        print(f"wrote {len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
