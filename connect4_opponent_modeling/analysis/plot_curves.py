"""Plotting utilities for learning curves and results visualization.

Generates publication-quality plots for:
- GTBench win rate vs training step
- Probe accuracy vs training step
- Final results bar chart grouped by benchmark
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import pandas as pd
except ImportError:
    pd = None


# Condition colors and labels
CONDITION_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#9467bd",
    "F": "#8c564b",
}
CONDITION_LABELS = {
    "A": "A: SFT Only",
    "B": "B: Self-Play RL",
    "C": "C: Solver-RL",
    "D": "D: Solver-RL + Future",
    "E": "E: Solver-RL + Opponent",
    "F": "F: Prompt-Only",
}


def _load_results(results_dir: str) -> List[Dict]:
    """Load all result JSON files.

    Args:
        results_dir: Directory containing *_eval_*.json files.

    Returns:
        List of result dicts.
    """
    results_path = Path(results_dir)
    all_results = []
    for json_file in sorted(results_path.glob("*_eval_*.json")):
        with open(json_file) as f:
            all_results.append(json.load(f))
    return all_results


def _ensure_matplotlib() -> None:
    """Raise ImportError if matplotlib is not available."""
    if plt is None:
        raise ImportError("matplotlib is required. pip install matplotlib")


def plot_transfer_curves(results_dir: str, output_dir: str) -> None:
    """Plot GTBench win rate vs training step, one line per condition.

    Args:
        results_dir: Directory with result JSONs.
        output_dir: Directory to save figures.
    """
    _ensure_matplotlib()

    results = _load_results(results_dir)
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in ["A", "B", "C", "D", "E", "F"]:
        cond_results = [r for r in results if r.get("condition") == condition]
        if not cond_results:
            continue

        steps = []
        win_rates = []
        for r in sorted(cond_results, key=lambda x: x.get("step", 0)):
            step = r.get("step", 0)
            gt = r.get("gtbench", {})
            if "breakthrough" in gt:
                wr = gt["breakthrough"].get("win_rate")
            elif "win_rate" in gt:
                wr = gt.get("win_rate")
            else:
                continue
            if wr is not None:
                steps.append(step)
                win_rates.append(wr)

        if steps:
            ax.plot(
                steps, win_rates,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
                marker="o", linewidth=2,
            )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("GTBench Win Rate", fontsize=12)
    ax.set_title("Transfer Performance (GTBench) vs Training Step", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(output_path / "transfer_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved transfer_curves.png to {output_path}")


def plot_probe_curves(results_dir: str, output_dir: str) -> None:
    """Plot probe accuracy vs training step.

    Args:
        results_dir: Directory with result JSONs.
        output_dir: Directory to save figures.
    """
    _ensure_matplotlib()

    results = _load_results(results_dir)
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in ["A", "B", "C", "D", "E", "F"]:
        cond_results = [r for r in results if r.get("condition") == condition]
        if not cond_results:
            continue

        steps = []
        accuracies = []
        for r in sorted(cond_results, key=lambda x: x.get("step", 0)):
            step = r.get("step", 0)
            probe = r.get("probe", {})
            acc = probe.get("overall_accuracy")
            if acc is not None:
                steps.append(step)
                accuracies.append(acc)

        if steps:
            ax.plot(
                steps, accuracies,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
                marker="s", linewidth=2,
            )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Probe Accuracy", fontsize=12)
    ax.set_title("Opponent Modeling Probe Accuracy vs Training Step", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(output_path / "probe_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved probe_curves.png to {output_path}")


def plot_main_table(results_dir: str, output_dir: str) -> None:
    """Plot final results as a grouped bar chart.

    Args:
        results_dir: Directory with result JSONs.
        output_dir: Directory to save figures.
    """
    _ensure_matplotlib()

    results = _load_results(results_dir)
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the latest result per condition
    latest = {}
    for r in results:
        cond = r.get("condition", "?")
        if cond not in latest or r.get("step", 0) >= latest[cond].get("step", 0):
            latest[cond] = r

    if not latest:
        print("No results to plot.")
        return

    conditions = sorted(latest.keys())
    benchmarks = ["Pons Optimal %", "Probe Accuracy", "GTBench NRA", "GSM8K"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(conditions))
    width = 0.18

    for i, bench in enumerate(benchmarks):
        values = []
        for cond in conditions:
            r = latest[cond]
            if bench == "Pons Optimal %":
                val = r.get("pons_benchmark", {}).get("overall_pct_optimal", 0)
            elif bench == "Probe Accuracy":
                val = r.get("probe", {}).get("overall_accuracy", 0)
            elif bench == "GTBench NRA":
                gt = r.get("gtbench", {})
                val = gt.get("average_nra", 0)
                val = (val + 1) / 2  # normalize from [-1,1] to [0,1]
            elif bench == "GSM8K":
                val = r.get("gsm8k", {}).get("accuracy", 0)
            else:
                val = 0
            values.append(max(0, val) if val is not None else 0)

        ax.bar(x + i * width, values, width, label=bench, alpha=0.85)

    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Final Results by Condition and Benchmark", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path / "main_results.png", dpi=150)
    plt.close(fig)
    print(f"Saved main_results.png to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curves and results")
    parser.add_argument("--results", type=str, default="results/", help="Results directory")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    args = parser.parse_args()

    print("=== Generating Plots ===\n")
    try:
        plot_transfer_curves(args.results, args.output)
        plot_probe_curves(args.results, args.output)
        plot_main_table(args.results, args.output)
        print("\nAll plots generated successfully.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
    except Exception as e:
        print(f"Error generating plots: {e}")
