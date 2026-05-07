"""Summarize narrow-experiment result JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


def _condition_from_label(label: str) -> str:
    if label.startswith("Value_"):
        return "Value"
    if label.startswith("OpponentNextMove_"):
        return "OpponentNextMove"
    return label


def _mean(vals: List[float]) -> float:
    return statistics.fmean(vals) if vals else 0.0


def _stdev(vals: List[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def summarize(results_dir: Path) -> Dict:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        summary = data.get("summary", {})
        label = summary.get("model_label", path.stem)
        condition = _condition_from_label(label)
        by_split = summary.get("by_split", {})
        rows.append(
            {
                "file": str(path),
                "label": label,
                "condition": condition,
                "primary_mean_move_quality": summary.get("primary_mean_move_quality", 0.0),
                "iid_move_quality": by_split.get("iid", {}).get("mean_move_quality", 0.0),
                "hard_move_quality": by_split.get("hard", {}).get("mean_move_quality", 0.0),
                "iid_reply_quality": by_split.get("iid", {}).get(
                    "mean_opponent_reply_quality", 0.0
                ),
                "hard_reply_quality": by_split.get("hard", {}).get(
                    "mean_opponent_reply_quality", 0.0
                ),
                "iid_answer_valid": by_split.get("iid", {}).get("answer_valid_rate", 0.0),
                "hard_answer_valid": by_split.get("hard", {}).get("answer_valid_rate", 0.0),
            }
        )

    grouped: Dict[str, Dict] = {}
    for condition in sorted({row["condition"] for row in rows}):
        cond_rows = [row for row in rows if row["condition"] == condition]
        grouped[condition] = {
            "n_runs": len(cond_rows),
            "primary_mean_move_quality_mean": _mean(
                [row["primary_mean_move_quality"] for row in cond_rows]
            ),
            "primary_mean_move_quality_stdev": _stdev(
                [row["primary_mean_move_quality"] for row in cond_rows]
            ),
            "iid_move_quality_mean": _mean([row["iid_move_quality"] for row in cond_rows]),
            "hard_move_quality_mean": _mean([row["hard_move_quality"] for row in cond_rows]),
            "iid_reply_quality_mean": _mean([row["iid_reply_quality"] for row in cond_rows]),
            "hard_reply_quality_mean": _mean([row["hard_reply_quality"] for row in cond_rows]),
            "iid_answer_valid_mean": _mean([row["iid_answer_valid"] for row in cond_rows]),
            "hard_answer_valid_mean": _mean([row["hard_answer_valid"] for row in cond_rows]),
        }

    return {"runs": rows, "by_condition": grouped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize narrow experiment results.")
    parser.add_argument("--results_dir", default="experiments/opponent_next_move/results")
    parser.add_argument(
        "--output",
        default="experiments/opponent_next_move/results/summary.json",
    )
    args = parser.parse_args()

    result = summarize(Path(args.results_dir))
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["by_condition"], indent=2, default=str))


if __name__ == "__main__":
    main()
