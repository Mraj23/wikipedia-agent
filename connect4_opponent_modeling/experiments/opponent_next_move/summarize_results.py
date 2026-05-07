"""Summarize narrow-experiment result JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.connect_four_env import ConnectFourEnv


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


def _validate_result_position(row: Dict) -> Tuple[bool, str]:
    moves = str(row.get("moves", ""))
    if any(ch not in "0123456" for ch in moves):
        return False, "move_sequence_contains_non_column_digit"

    env = ConnectFourEnv()
    try:
        env.from_move_sequence([int(ch) for ch in moves])
    except ValueError as exc:
        return False, f"illegal_move_sequence: {exc}"

    if env.to_move_sequence() != moves:
        return False, "move_sequence_replay_mismatch"
    if row.get("move_count") is not None and int(row["move_count"]) != len(moves):
        return False, "move_count_mismatch"
    if env.is_terminal():
        return False, "terminal_position"
    if len(env.legal_moves()) < 2:
        return False, "fewer_than_two_legal_moves"
    return True, "ok"


def _position_stats(data: Dict, split: str) -> Dict[str, float]:
    by_split = data.get("summary", {}).get("by_split", {})
    split_summary = by_split.get(split, {})
    if "position_valid_rate" in split_summary:
        return {
            "position_valid_rate": split_summary.get("position_valid_rate", 0.0),
            "position_invalid_count": split_summary.get("position_invalid_count", 0),
        }

    rows = [row for row in data.get("records", []) if row.get("split") == split]
    if not rows:
        return {"position_valid_rate": 0.0, "position_invalid_count": 0}

    validity = [_validate_result_position(row)[0] for row in rows]
    return {
        "position_valid_rate": _mean([float(valid) for valid in validity]),
        "position_invalid_count": sum(1 for valid in validity if not valid),
    }


def summarize(results_dir: Path) -> Dict:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        data = json.loads(path.read_text())
        summary = data.get("summary", {})
        label = summary.get("model_label", path.stem)
        condition = _condition_from_label(label)
        by_split = summary.get("by_split", {})
        iid_position_stats = _position_stats(data, "iid")
        hard_position_stats = _position_stats(data, "hard")
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
                "iid_position_valid": iid_position_stats["position_valid_rate"],
                "hard_position_valid": hard_position_stats["position_valid_rate"],
                "iid_position_invalid_count": iid_position_stats["position_invalid_count"],
                "hard_position_invalid_count": hard_position_stats["position_invalid_count"],
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
            "iid_position_valid_mean": _mean([row["iid_position_valid"] for row in cond_rows]),
            "hard_position_valid_mean": _mean([row["hard_position_valid"] for row in cond_rows]),
            "iid_position_invalid_count_total": sum(
                row["iid_position_invalid_count"] for row in cond_rows
            ),
            "hard_position_invalid_count_total": sum(
                row["hard_position_invalid_count"] for row in cond_rows
            ),
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
