"""Load BBH tasks and split into train / eval JSONL.

Reads configs/tasks.yaml for task mapping; produces:
  data/processed/train.jsonl   (used for RLTF rollout generation)
  data/processed/eval.jsonl    (held-out, used by src/eval.py)
"""

import argparse
import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tasks.yaml")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="override task list from config",
    )
    parser.add_argument("--train-n", type=int, default=None)
    parser.add_argument("--eval-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    task_map = cfg["tasks"]
    if args.tasks:
        task_map = {t: task_map[t] for t in args.tasks}
    train_n = args.train_n if args.train_n is not None else cfg["train_n"]
    eval_n = args.eval_n if args.eval_n is not None else cfg["eval_n"]
    seed = args.seed if args.seed is not None else cfg["seed"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"

    rng = random.Random(seed)
    train_rows = []
    eval_rows = []

    for task, hf_name in task_map.items():
        ds = load_dataset(cfg["dataset"], hf_name, split="test")
        rows = list(ds)
        rng.shuffle(rows)
        wanted = train_n + eval_n
        if len(rows) < wanted:
            print(
                f"warning: {task} has {len(rows)} examples, want "
                f"{wanted}; using all available"
            )
        cut_train = min(train_n, len(rows))
        cut_eval = min(eval_n, len(rows) - cut_train)
        for i, row in enumerate(rows[:cut_train]):
            train_rows.append(
                {
                    "id": f"{task}_train_{i:05d}",
                    "task": task,
                    "question": row["input"],
                    "gold": row["target"],
                }
            )
        for i, row in enumerate(rows[cut_train : cut_train + cut_eval]):
            eval_rows.append(
                {
                    "id": f"{task}_eval_{i:05d}",
                    "task": task,
                    "question": row["input"],
                    "gold": row["target"],
                }
            )

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    with train_path.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with eval_path.open("w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(train_rows)} -> {train_path}")
    print(f"wrote {len(eval_rows)} -> {eval_path}")


if __name__ == "__main__":
    main()
