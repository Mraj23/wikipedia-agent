"""Sample BBH Tracking Shuffled Objects and Logical Deduction into JSONL."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


TASK_MAP = {
    "tracking_shuffled_objects": "tracking_shuffled_objects_seven_objects",
    "logical_deduction": "logical_deduction_seven_objects",
}


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
    parser.add_argument("--dataset", default="lukaemon/bbh")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    for task in args.tasks:
        hf_name = TASK_MAP[task]
        ds = load_dataset(args.dataset, hf_name, split="test")
        rows = list(ds)
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
