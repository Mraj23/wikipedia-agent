"""Evaluate Pons-normalized move quality for the narrow experiment."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from eval.model_loader import create_model_fn
from training.prompts import format_prompt, parse_response, validate_response


def _norm_from_scores(scores: Dict[int, int], move: Optional[int]) -> float:
    if move is None or move not in scores:
        return 0.0
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return 1.0
    return (scores[move] - min_score) / (max_score - min_score)


def _opponent_reply_quality(
    solver: PonsSolver,
    env: ConnectFourEnv,
    model_move: Optional[int],
    predicted_reply: Optional[int],
) -> Optional[float]:
    if model_move is None or model_move not in env.legal_moves():
        return 0.0

    next_env = env.copy()
    next_env.make_move(model_move)
    if next_env.is_terminal():
        return None

    reply_scores = solver.analyze(next_env)
    return _norm_from_scores(reply_scores, predicted_reply)


def _load_records(path: Path, max_positions_per_split: Optional[int]) -> List[Dict]:
    counts: Dict[str, int] = {}
    records: List[Dict] = []
    with path.open() as handle:
        for line in handle:
            rec = json.loads(line)
            split = rec.get("split", "unknown")
            if max_positions_per_split is not None:
                if counts.get(split, 0) >= max_positions_per_split:
                    continue
            counts[split] = counts.get(split, 0) + 1
            records.append(rec)
    return records


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else 0.0


def evaluate(
    *,
    model: str,
    condition: str,
    banks_path: Path,
    output: Path,
    model_label: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    max_positions_per_split: Optional[int],
) -> Dict:
    solver = PonsSolver(strict=True)
    if not solver.is_available():
        raise RuntimeError("Pons solver is required. Run scripts/bootstrap_gpu.sh first.")

    model_fn = create_model_fn(
        model,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    records = _load_records(banks_path, max_positions_per_split)

    outputs: List[Dict] = []
    for idx, rec in enumerate(records, start=1):
        env = ConnectFourEnv()
        env.from_move_sequence([int(c) for c in rec["moves"]])
        scores = {int(col): int(score) for col, score in rec["scores"].items()}

        prompt = format_prompt(condition, env)
        response = model_fn(prompt)
        parsed = parse_response(response, condition)
        schema_valid, schema_reason = validate_response(parsed, condition, env.legal_moves())

        move = parsed.get("move")
        answer_valid = move is not None and move in env.legal_moves()
        move_quality = _norm_from_scores(scores, move)
        opponent_quality = _opponent_reply_quality(
            solver,
            env,
            move if answer_valid else None,
            parsed.get("opponent_prediction"),
        )

        best_moves = [int(col) for col in rec.get("best_moves", [])]
        outputs.append(
            {
                "index": idx,
                "split": rec.get("split", "unknown"),
                "moves": rec["moves"],
                "move_count": rec.get("move_count"),
                "score_spread": rec.get("score_spread"),
                "model_move": move,
                "answer_valid": answer_valid,
                "schema_valid": schema_valid,
                "schema_reason": schema_reason,
                "move_quality": move_quality,
                "optimal": answer_valid and move in best_moves,
                "opponent_prediction": parsed.get("opponent_prediction"),
                "opponent_reply_quality": opponent_quality,
                "response": response[:1000],
            }
        )

    by_split = {}
    for split in sorted({row["split"] for row in outputs}):
        rows = [row for row in outputs if row["split"] == split]
        opp_rows = [row for row in rows if row["opponent_reply_quality"] is not None]
        by_split[split] = {
            "n": len(rows),
            "mean_move_quality": _mean(row["move_quality"] for row in rows),
            "pct_optimal": _mean(float(row["optimal"]) for row in rows),
            "answer_valid_rate": _mean(float(row["answer_valid"]) for row in rows),
            "schema_valid_rate": _mean(float(row["schema_valid"]) for row in rows),
            "mean_opponent_reply_quality": _mean(
                row["opponent_reply_quality"] for row in opp_rows
            ),
            "opponent_reply_n": len(opp_rows),
        }

    summary = {
        "model": model,
        "model_label": model_label,
        "condition": condition,
        "banks_path": str(banks_path),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "total": len(outputs),
        "by_split": by_split,
        "primary_mean_move_quality": _mean(
            item["mean_move_quality"] for item in by_split.values()
        ),
    }

    result = {"summary": summary, "records": outputs}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normalized Connect Four move quality.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", default="BaseScaffold")
    parser.add_argument("--model_label", default=None)
    parser.add_argument(
        "--banks",
        default="experiments/opponent_next_move/data/connect4_eval_banks.jsonl",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_positions_per_split", type=int, default=None)
    args = parser.parse_args()

    result = evaluate(
        model=args.model,
        condition=args.condition,
        banks_path=Path(args.banks),
        output=Path(args.output),
        model_label=args.model_label or args.condition,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_positions_per_split=args.max_positions_per_split,
    )
    print(json.dumps(result["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
