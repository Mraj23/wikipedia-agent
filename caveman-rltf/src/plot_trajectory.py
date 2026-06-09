"""Plot the training trajectory in (thinking tokens, accuracy) space.

Reads a trainer's trajectory.jsonl (one row per checkpoint eval) and draws
accuracy vs mean thinking tokens, as a path colored by training step — the
AIME-style "evolution" curve (expand / compress arms).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="trajectory.jsonl")
    ap.add_argument("--output", required=True, help="output png")
    ap.add_argument("--title", default="Thinking-length evolution")
    args = ap.parse_args()

    pts = [json.loads(l) for l in open(args.input) if l.strip()]
    pts.sort(key=lambda p: p["step"])
    # Prefer the trustworthy metrics (completed-within-budget accuracy, total
    # output tokens); fall back to legacy keys for older runs.
    def gx(p):
        return p.get("mean_total_tokens", p.get("mean_thinking_tokens"))
    def gy(p):
        return 100.0 * p.get("completed_accuracy", p.get("accuracy", 0.0))
    x = np.array([gx(p) for p in pts])
    y = np.array([gy(p) for p in pts])
    steps = np.array([p["step"] for p in pts])

    fig, ax = plt.subplots(figsize=(8, 7))
    if len(pts) >= 2:
        segs = np.stack([np.column_stack([x[:-1], y[:-1]]),
                         np.column_stack([x[1:], y[1:]])], axis=1)
        lc = LineCollection(segs, cmap="cool", linewidth=2.5)
        lc.set_array(steps[:-1])
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label("training step")
    sc = ax.scatter(x, y, c=steps, cmap="cool", s=60, zorder=3)
    for xi, yi, p in zip(x, y, pts):
        ax.annotate(f"s{p['step']}", (xi, yi),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("mean total output tokens")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.margins(0.1)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"trajectory plot -> {args.output}")


if __name__ == "__main__":
    main()
