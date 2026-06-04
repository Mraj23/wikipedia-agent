"""Aggregate eval JSONLs into a summary CSV.

Groups by (model, condition, prompt_condition, task). Reports accuracy with
a bootstrap 95% CI, output-token stats, accuracy-per-1k-tokens, a compression
ratio vs the plain baseline, and per-budget accuracy.

Primary length axis is `total_output_tokens` (parser-independent, and the
quantity Caveman targets). `reasoning_tokens` is reported too but only over
rows where the "Answer:" label parsed (parse_success), since terse outputs
that drop the label make the raw reasoning-token count unreliable.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


BUDGET_THRESHOLDS = [32, 64, 96]
KEYS = ["model", "condition", "prompt_condition", "task"]


def _bootstrap_ci(values, n_boot=2000, seed=0):
    """Return (lo, hi) 95% bootstrap CI for the mean of a 0/1 or numeric series."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (lo, hi)


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
    # Back-compat: older evals may lack prompt_condition.
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
        mean_out = sub["total_output_tokens"].mean()
        parsed = sub[sub["parse_success"]]
        rec = dict(zip(KEYS, key_vals))
        rec.update(
            {
                "n_examples": len(sub),
                "accuracy": acc,
                "accuracy_ci_lo": acc_lo,
                "accuracy_ci_hi": acc_hi,
                "mean_output_tokens": mean_out,
                "median_output_tokens": sub["total_output_tokens"].median(),
                "mean_reasoning_tokens_parsed": (
                    parsed["reasoning_tokens"].mean() if len(parsed) else float("nan")
                ),
                "accuracy_per_1k_output_tokens": (
                    1000.0 * acc / mean_out if mean_out else float("nan")
                ),
                "parse_success_rate": sub["parse_success"].mean(),
                "invalid_answer_rate": sub["is_invalid"].mean(),
            }
        )
        records.append(rec)

    agg = pd.DataFrame(records)

    # Compression ratio vs the plain-prompt baseline of the same model/task.
    base = (
        agg[agg["prompt_condition"] == "plain"]
        .groupby(["model", "task"])["mean_output_tokens"]
        .mean()
        .to_dict()
    )
    agg["compression_ratio_vs_plain"] = agg.apply(
        lambda r: (
            base.get((r["model"], r["task"]), float("nan")) / r["mean_output_tokens"]
            if r["mean_output_tokens"]
            else float("nan")
        ),
        axis=1,
    )

    for budget in BUDGET_THRESHOLDS:
        col = f"accuracy_under_{budget}_output_tokens"
        budget_acc = (
            df.assign(_in_budget=df["total_output_tokens"] <= budget)
            .assign(_correct=lambda x: x["correct"] & x["_in_budget"])
            .groupby(KEYS)
            .agg(_acc=("_correct", "mean"))
            .reset_index()
            .rename(columns={"_acc": col})
        )
        agg = agg.merge(budget_acc, on=KEYS, how="left")

    summary_path = out_dir / "summary.csv"
    agg.to_csv(summary_path, index=False)
    (out_dir / "summary_by_task.csv").write_text(
        agg.sort_values(["task", "condition", "prompt_condition"]).to_csv(index=False)
    )
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
