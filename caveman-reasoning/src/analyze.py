"""Aggregate graded+tokenized rows into a summary CSV."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    df["normalized_answer"] = df["normalized_answer"].fillna("")
    df["is_invalid"] = df["normalized_answer"].str.strip() == ""

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = ["model", "task", "condition"]
    summary = (
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

    summary["accuracy_per_100_reasoning_tokens"] = (
        summary["accuracy"]
        / summary["mean_reasoning_tokens"].clip(lower=1)
        * 100
    )

    summary.to_csv(out_path, index=False)

    by_task_path = out_path.parent / "summary_by_task.csv"
    summary.sort_values(["task", "condition"]).to_csv(by_task_path, index=False)

    errors_path = out_path.parent / "errors.jsonl"
    df[~df["correct"]].to_json(errors_path, orient="records", lines=True)

    print(f"summary -> {out_path}")
    print(f"by task -> {by_task_path}")
    print(f"errors -> {errors_path}")


if __name__ == "__main__":
    main()
