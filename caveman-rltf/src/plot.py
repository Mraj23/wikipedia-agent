"""Pareto plot per task and combined: accuracy vs mean reasoning tokens."""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="summary.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    for task, sub in df.groupby("task"):
        ax.scatter(sub["mean_reasoning_tokens"], sub["accuracy"], s=80, label=task)
        for _, r in sub.iterrows():
            ax.annotate(
                r["condition"],
                (r["mean_reasoning_tokens"], r["accuracy"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("mean reasoning tokens")
    ax.set_ylabel("exact-match accuracy")
    ax.set_title("Caveman RLTF: accuracy vs reasoning tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_combined.png", dpi=150)
    plt.close(fig)

    for task, sub in df.groupby("task"):
        fig, ax = plt.subplots(figsize=(8, 6))
        for cond, csub in sub.groupby("condition"):
            ax.scatter(
                csub["mean_reasoning_tokens"], csub["accuracy"], s=80, label=cond
            )
        ax.set_xlabel("mean reasoning tokens")
        ax.set_ylabel("exact-match accuracy")
        ax.set_title(task)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"pareto_{task}.png", dpi=150)
        plt.close(fig)

    print(f"plots -> {out_dir}")


if __name__ == "__main__":
    main()
