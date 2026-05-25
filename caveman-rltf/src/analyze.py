"""Aggregate eval JSONLs into a summary CSV with per-budget accuracy."""

import argparse
import json
from pathlib import Path

import pandas as pd


BUDGET_THRESHOLDS = [32, 64, 96]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="dir of eval JSONL files (or a single file)",
    )
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
    df["correct"] = df["correct"].astype(bool)
    df["parse_success"] = df["parse_success"].astype(bool)
    df["normalized_answer"] = df["normalized_answer"].fillna("")
    df["is_invalid"] = df["normalized_answer"].str.strip() == ""

    keys = ["model", "condition", "task"]
    agg = (
        df.groupby(keys)
        .agg(
            n_examples=("id", "count"),
            accuracy=("correct", "mean"),
            mean_reasoning_tokens=("reasoning_tokens", "mean"),
            median_reasoning_tokens=("reasoning_tokens", "median"),
            mean_total_output_tokens=("total_output_tokens", "mean"),
            parse_success_rate=("parse_success", "mean"),
            invalid_answer_rate=("is_invalid", "mean"),
        )
        .reset_index()
    )

    for budget in BUDGET_THRESHOLDS:
        col = f"accuracy_under_{budget}_tokens"
        budget_acc = (
            df.assign(_in_budget=df["reasoning_tokens"] <= budget)
            .assign(_correct=lambda x: x["correct"] & x["_in_budget"])
            .groupby(keys)
            .agg(_acc=("_correct", "mean"))
            .reset_index()
            .rename(columns={"_acc": col})
        )
        agg = agg.merge(budget_acc, on=keys, how="left")

    summary_path = out_dir / "summary.csv"
    agg.to_csv(summary_path, index=False)
    (out_dir / "summary_by_task.csv").write_text(
        agg.sort_values(["task", "condition"]).to_csv(index=False)
    )
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
