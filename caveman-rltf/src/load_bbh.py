"""Load reasoning tasks and split into train / eval JSONL.

Supports two task types (configured in configs/tasks.yaml):
  - bbh:    BBH multiple-choice tasks, fetched from the suzgunmirac GitHub
            raw mirror (HF Hub egress is blocked in some environments).
            A single 250-row `test` split is shuffled and sliced disjointly
            into eval (first) then train (rest).
  - gsm8k:  grade-school math, fetched from the openai/grade-school-math
            GitHub raw mirror. Uses the dataset's OWN train split for train
            rows and its test split for eval rows (no contamination).

Produces:
  data/processed/train.jsonl   (used for RLTF rollout generation)
  data/processed/eval.jsonl    (held-out, used by src/eval.py)

Split fix: eval is allocated FIRST, so a held-out eval set is never empty
even when a task has fewer rows than train_n + eval_n. (The old logic took
train first and could leave eval_n == 0 on the 250-row BBH tasks.)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

import yaml


BBH_URL = (
    "https://raw.githubusercontent.com/suzgunmirac/"
    "BIG-Bench-Hard/main/bbh/{name}.json"
)
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/{split}.jsonl"
)
GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _fetch_jsonl(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return [json.loads(l) for l in r.read().decode().splitlines() if l.strip()]


def _bbh_rows(name: str):
    examples = _fetch_json(BBH_URL.format(name=name))["examples"]
    return [{"question": e["input"], "gold": e["target"]} for e in examples]


def _gsm8k_rows(split: str):
    rows = []
    for e in _fetch_jsonl(GSM8K_URL.format(split=split)):
        m = GSM8K_ANSWER_RE.search(e["answer"])
        if not m:
            continue
        rows.append({"question": e["question"], "gold": m.group(1).replace(",", "")})
    return rows


def _split_rows(task: str, spec, train_n: int, eval_n: int, rng: random.Random):
    """Return (train_rows, eval_rows) for a single task."""
    ttype = spec["type"] if isinstance(spec, dict) else "bbh"

    if ttype == "gsm8k":
        # Dataset has its own train/test splits — use them directly.
        train_pool = _gsm8k_rows("train")
        eval_pool = _gsm8k_rows("test")
        rng.shuffle(train_pool)
        rng.shuffle(eval_pool)
        train_rows = train_pool[:train_n]
        eval_rows = eval_pool[:eval_n]
    else:
        name = spec["name"] if isinstance(spec, dict) else spec
        rows = _bbh_rows(name)
        rng.shuffle(rows)
        wanted = train_n + eval_n
        if len(rows) < wanted:
            print(
                f"warning: {task} has {len(rows)} rows, want {wanted}; "
                f"allocating eval first, train gets the remainder"
            )
        # Allocate EVAL first so the held-out set is never starved.
        cut_eval = min(eval_n, len(rows))
        cut_train = min(train_n, len(rows) - cut_eval)
        eval_rows = rows[:cut_eval]
        train_rows = rows[cut_eval : cut_eval + cut_train]

    def tag(rows, split):
        return [
            {
                "id": f"{task}_{split}_{i:05d}",
                "task": task,
                "question": r["question"],
                "gold": r["gold"],
            }
            for i, r in enumerate(rows)
        ]

    return tag(train_rows, "train"), tag(eval_rows, "eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tasks.yaml")
    parser.add_argument("--tasks", nargs="+", default=None, help="override task list")
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
    train_rows, eval_rows = [], []
    for task, spec in task_map.items():
        tr, ev = _split_rows(task, spec, train_n, eval_n, rng)
        train_rows.extend(tr)
        eval_rows.extend(ev)
        print(f"  {task}: {len(tr)} train, {len(ev)} eval")

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
    if not eval_rows:
        raise SystemExit("eval set is empty — lower train_n or check task config")


if __name__ == "__main__":
    main()
