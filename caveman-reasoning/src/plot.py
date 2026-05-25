"""Pareto and savings plots from summary CSV."""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_pareto(df: pd.DataFrame, x_col: str, x_label: str, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for task, sub in df.groupby("task"):
        ax.scatter(sub[x_col], sub["accuracy"], s=60, label=task)
        for _, row in sub.iterrows():
            ax.annotate(
                row["condition"],
                (row[x_col], row["accuracy"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_savings(df: pd.DataFrame, baseline: str, out_path: Path):
    base = df[df["condition"] == baseline].set_index(["model", "task"])
    fig, ax = plt.subplots(figsize=(8, 6))
    for (model, task), sub in df.groupby(["model", "task"]):
        if (model, task) not in base.index:
            continue
        b = base.loc[(model, task)]
        b_tokens = max(float(b["mean_reasoning_tokens"]), 1.0)
        for _, row in sub.iterrows():
            if row["condition"] == baseline:
                continue
            pct_tok_red = (b_tokens - row["mean_reasoning_tokens"]) / b_tokens * 100
            acc_diff = row["accuracy"] - b["accuracy"]
            ax.scatter(pct_tok_red, acc_diff)
            ax.annotate(
                f"{task}/{row['condition']}",
                (pct_tok_red, acc_diff),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel(f"% reduction in reasoning tokens vs {baseline}")
    ax.set_ylabel(f"accuracy diff vs {baseline}")
    ax.set_title(f"Token savings vs accuracy drop ({baseline} baseline)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="primary pareto plot path; companion plots saved alongside",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_path = Path(args.output)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_pareto(
        df,
        "mean_reasoning_tokens",
        "mean reasoning tokens",
        out_path,
        "Accuracy vs reasoning tokens",
    )
    plot_pareto(
        df,
        "mean_total_output_tokens",
        "mean total output tokens",
        out_dir / "pareto_total_tokens.png",
        "Accuracy vs total output tokens",
    )
    plot_savings(df, "normal_cot", out_dir / "savings_vs_normal_cot.png")

    for task, sub in df.groupby("task"):
        plot_pareto(
            sub,
            "mean_reasoning_tokens",
            "mean reasoning tokens",
            out_dir / f"pareto_{task}.png",
            f"Accuracy vs reasoning tokens — {task}",
        )

    print(f"plots -> {out_dir}")


if __name__ == "__main__":
    main()
