"""Pareto plots: accuracy vs tokens, per task and combined.

Two x-axes are drawn:
  - thinking tokens  (pareto_think_*.png)  -- the "brain" / primary result
  - answer tokens    (pareto_answer_*.png) -- the visible "mouth"
Accuracy error bars are the bootstrap 95% CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _label(row):
    return f"{row['condition']}/{row['prompt_condition']}"


def _scatter(ax, sub, xcol):
    yerr = [
        (sub["accuracy"] - sub["accuracy_ci_lo"]).clip(lower=0),
        (sub["accuracy_ci_hi"] - sub["accuracy"]).clip(lower=0),
    ]
    ax.errorbar(sub[xcol], sub["accuracy"], yerr=yerr, fmt="o", ms=7, capsize=3, alpha=0.8)
    for _, r in sub.iterrows():
        ax.annotate(
            _label(r), (r[xcol], r["accuracy"]),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )


def _make(df, out_dir, xcol, tag, xlabel):
    fig, ax = plt.subplots(figsize=(10, 7))
    for _, sub in df.groupby("task"):
        _scatter(ax, sub, xcol)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("exact-match accuracy")
    ax.set_title(f"Caveman RLTF-SFT: accuracy vs {xlabel} (all tasks)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"pareto_{tag}_combined.png", dpi=150)
    plt.close(fig)

    for task, sub in df.groupby("task"):
        fig, ax = plt.subplots(figsize=(9, 6))
        _scatter(ax, sub, xcol)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("exact-match accuracy")
        ax.set_title(str(task))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"pareto_{tag}_{task}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="summary.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _make(df, out_dir, "mean_thinking_tokens", "think", "mean thinking tokens")
    _make(df, out_dir, "mean_answer_tokens", "answer", "mean answer tokens")
    print(f"plots -> {out_dir}")


if __name__ == "__main__":
    main()
