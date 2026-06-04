"""Pareto plot: accuracy vs mean OUTPUT tokens, per task and combined.

X-axis is total output tokens (the quantity Caveman targets and the
parser-independent length metric). Each point is a (condition,
prompt_condition) arm; accuracy error bars are the bootstrap 95% CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _label(row):
    return f"{row['condition']}/{row['prompt_condition']}"


def _scatter(ax, sub):
    yerr = [
        (sub["accuracy"] - sub["accuracy_ci_lo"]).clip(lower=0),
        (sub["accuracy_ci_hi"] - sub["accuracy"]).clip(lower=0),
    ]
    ax.errorbar(
        sub["mean_output_tokens"],
        sub["accuracy"],
        yerr=yerr,
        fmt="o",
        ms=7,
        capsize=3,
        alpha=0.8,
    )
    for _, r in sub.iterrows():
        ax.annotate(
            _label(r),
            (r["mean_output_tokens"], r["accuracy"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="summary.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    for task, sub in df.groupby("task"):
        _scatter(ax, sub)
    ax.set_xlabel("mean output tokens")
    ax.set_ylabel("exact-match accuracy")
    ax.set_title("Caveman RLTF-SFT: accuracy vs output tokens (all tasks)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_combined.png", dpi=150)
    plt.close(fig)

    for task, sub in df.groupby("task"):
        fig, ax = plt.subplots(figsize=(9, 6))
        _scatter(ax, sub)
        ax.set_xlabel("mean output tokens")
        ax.set_ylabel("exact-match accuracy")
        ax.set_title(str(task))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"pareto_{task}.png", dpi=150)
        plt.close(fig)

    print(f"plots -> {out_dir}")


if __name__ == "__main__":
    main()
