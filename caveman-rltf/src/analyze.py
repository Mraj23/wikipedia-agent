"""Aggregate eval JSONLs into a summary CSV.

Groups by (model, condition, prompt_condition, task). For a thinking model we
report BOTH channels:
  - thinking_tokens  -- the "brain"; PRIMARY compression target / Pareto x-axis
  - answer/output tokens -- the "mouth"
plus accuracy with a bootstrap 95% CI, accuracy-per-1k-thinking-tokens, a
compression ratio vs the plain baseline, and per-think-budget accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


THINK_BUDGETS = [64, 128, 256]
KEYS = ["model", "condition", "prompt_condition", "task"]


def _bootstrap_ci(values, n_boot=2000, seed=0):
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="dir of eval JSONLs (or a file)")
    parser.add_argument("--output", required=True, help="output dir")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [in_path] if in_path.is_file() else sorted(in_path.rglob("*.jsonl"))
    rows = []
    for f in files:
        with f.open() as g:
            for line in g:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no rows in {args.input}")
    if "prompt_condition" not in df.columns:
        df["prompt_condition"] = "caveman"
    df["correct"] = df["correct"].astype(bool)
    df["parse_success"] = df["parse_success"].astype(bool)
    df["normalized_answer"] = df["normalized_answer"].fillna("")
    df["is_invalid"] = df["normalized_answer"].astype(str).str.strip() == ""

    records = []
    for key_vals, sub in df.groupby(KEYS):
        acc = sub["correct"].mean()
        acc_lo, acc_hi = _bootstrap_ci(sub["correct"])
        mean_think = sub["thinking_tokens"].mean()
        rec = dict(zip(KEYS, key_vals))
        rec.update(
            {
                "n_examples": len(sub),
                "accuracy": acc,
                "accuracy_ci_lo": acc_lo,
                "accuracy_ci_hi": acc_hi,
                "mean_thinking_tokens": mean_think,
                "median_thinking_tokens": sub["thinking_tokens"].median(),
                "mean_answer_tokens": sub["answer_tokens"].mean(),
                "mean_output_tokens": sub["output_tokens"].mean(),
                "mean_total_output_tokens": sub["total_output_tokens"].mean(),
                "accuracy_per_1k_thinking_tokens": (
                    1000.0 * acc / mean_think if mean_think else float("nan")
                ),
                "has_thinking_rate": sub["has_thinking"].mean()
                if "has_thinking" in sub
                else float("nan"),
                "parse_success_rate": sub["parse_success"].mean(),
                "invalid_answer_rate": sub["is_invalid"].mean(),
            }
        )
        records.append(rec)

    agg = pd.DataFrame(records)

    base = (
        agg[agg["prompt_condition"] == "plain"]
        .groupby(["model", "task"])["mean_thinking_tokens"]
        .mean()
        .to_dict()
    )
    agg["think_compression_vs_plain"] = agg.apply(
        lambda r: (
            base.get((r["model"], r["task"]), float("nan")) / r["mean_thinking_tokens"]
            if r["mean_thinking_tokens"]
            else float("nan")
        ),
        axis=1,
    )

    for budget in THINK_BUDGETS:
        col = f"accuracy_under_{budget}_thinking_tokens"
        b = (
            df.assign(_in=df["thinking_tokens"] <= budget)
            .assign(_c=lambda x: x["correct"] & x["_in"])
            .groupby(KEYS)
            .agg(_acc=("_c", "mean"))
            .reset_index()
            .rename(columns={"_acc": col})
        )
        agg = agg.merge(b, on=KEYS, how="left")

    summary_path = out_dir / "summary.csv"
    agg.to_csv(summary_path, index=False)
    (out_dir / "summary_by_task.csv").write_text(
        agg.sort_values(["task", "condition", "prompt_condition"]).to_csv(index=False)
    )
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
