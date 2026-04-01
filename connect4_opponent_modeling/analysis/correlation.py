"""Correlation analysis between probe accuracy and transfer performance.

Computes Pearson correlation between mechanistic probe accuracy
and GTBench win rate to test the hypothesis that opponent modeling
representations cause transfer.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy import stats
except ImportError:
    stats = None


def load_all_results(results_dir: str) -> "pd.DataFrame":
    """Load all evaluation result JSON files into a DataFrame.

    Args:
        results_dir: Directory containing *_eval_*.json files.

    Returns:
        DataFrame with columns: condition, step, probe_accuracy,
        gtbench_win_rate, gsm8k_accuracy.

    Raises:
        ImportError: If pandas is not installed.
    """
    if pd is None:
        raise ImportError("pandas is required. pip install pandas")

    results_path = Path(results_dir)
    records = []

    for json_file in sorted(results_path.glob("*_eval_*.json")):
        with open(json_file) as f:
            data = json.load(f)

        record = {
            "condition": data.get("condition", "unknown"),
            "step": data.get("step", 0),
            "timestamp": data.get("timestamp", ""),
            "file": json_file.name,
        }

        # Probe accuracy
        probe = data.get("probe", {})
        record["probe_accuracy"] = probe.get("overall_accuracy", None)

        # GTBench win rate
        gtbench = data.get("gtbench", {})
        if "breakthrough" in gtbench:
            record["gtbench_win_rate"] = gtbench["breakthrough"].get("win_rate", None)
        elif "win_rate" in gtbench:
            record["gtbench_win_rate"] = gtbench.get("win_rate", None)
        else:
            record["gtbench_win_rate"] = None

        # GSM8K accuracy
        gsm8k = data.get("gsm8k", {})
        record["gsm8k_accuracy"] = gsm8k.get("accuracy", None)

        # Pons benchmark
        pons = data.get("pons_benchmark", {})
        record["pons_pct_optimal"] = pons.get("overall_pct_optimal", None)

        records.append(record)

    return pd.DataFrame(records)


def compute_probe_transfer_correlation(df: "pd.DataFrame") -> Dict:
    """Compute Pearson correlation between probe accuracy and GTBench win rate.

    Args:
        df: DataFrame from load_all_results.

    Returns:
        Dict with pearson_r, p_value, n, and interpretation.

    Raises:
        ImportError: If scipy is not installed.
    """
    if stats is None:
        raise ImportError("scipy is required. pip install scipy")

    # Filter to rows with both values
    valid = df.dropna(subset=["probe_accuracy", "gtbench_win_rate"])
    if len(valid) < 3:
        return {
            "pearson_r": None,
            "p_value": None,
            "n": len(valid),
            "interpretation": "Insufficient data points (need >= 3)",
        }

    r, p = stats.pearsonr(valid["probe_accuracy"], valid["gtbench_win_rate"])

    if p < 0.01:
        sig = "highly significant"
    elif p < 0.05:
        sig = "significant"
    else:
        sig = "not significant"

    if abs(r) > 0.7:
        strength = "strong"
    elif abs(r) > 0.4:
        strength = "moderate"
    else:
        strength = "weak"

    return {
        "pearson_r": float(r),
        "p_value": float(p),
        "n": len(valid),
        "interpretation": f"{strength} {'positive' if r > 0 else 'negative'} correlation ({sig})",
    }


def print_correlation_table(results_dir: str) -> None:
    """Print a formatted correlation analysis table.

    Args:
        results_dir: Directory containing result JSON files.
    """
    df = load_all_results(results_dir)

    if df.empty:
        print("No results found.")
        return

    print("\n=== Results Summary ===\n")
    print(df[["condition", "probe_accuracy", "gtbench_win_rate", "gsm8k_accuracy"]].to_string(index=False))

    print("\n=== Probe → Transfer Correlation ===\n")
    corr = compute_probe_transfer_correlation(df)
    if corr["pearson_r"] is not None:
        print(f"  Pearson r: {corr['pearson_r']:.4f}")
        print(f"  p-value:   {corr['p_value']:.4f}")
        print(f"  N:         {corr['n']}")
        print(f"  Interpretation: {corr['interpretation']}")
    else:
        print(f"  {corr['interpretation']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlation analysis")
    parser.add_argument("--results", type=str, default="results/", help="Results directory")
    args = parser.parse_args()

    print_correlation_table(args.results)
