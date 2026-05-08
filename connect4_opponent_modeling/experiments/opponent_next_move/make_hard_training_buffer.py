"""Create a Pons-filtered hard-position training buffer.

The trainer's PositionBuffer format is intentionally simple: a JSON list of
move-sequence strings. This script keeps that format but filters sampled
Connect Four positions so future GRPO groups are more likely to have useful
reward variance.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver


def _sample_random_position(
    rng: random.Random,
    *,
    min_moves: int,
    max_moves: int,
    center_bias: float,
) -> Optional[ConnectFourEnv]:
    env = ConnectFourEnv()
    target_len = rng.randint(min_moves, max_moves)
    center_weights = {0: 1.0, 1: 1.4, 2: 2.0, 3: 2.6, 4: 2.0, 5: 1.4, 6: 1.0}

    for _ in range(target_len):
        if env.is_terminal() or not env.legal_moves():
            break
        legal = env.legal_moves()
        if rng.random() < center_bias:
            weights = [center_weights[col] for col in legal]
            move = rng.choices(legal, weights=weights, k=1)[0]
        else:
            move = rng.choice(legal)
        env.make_move(move)

    if env.is_terminal() or len(env.legal_moves()) < 2:
        return None
    return env


def _phase(move_count: int) -> str:
    if move_count <= 8:
        return "opening"
    if move_count <= 20:
        return "middle"
    return "late"


def _score_stats(scores: Dict[int, int]) -> Tuple[int, int, List[int]]:
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    best_moves = [col for col, score in scores.items() if score == max_score]
    return max_score - min_score, max_score, best_moves


def build_hard_buffer(
    output: Path,
    *,
    n_positions: int,
    seed: int,
    min_moves: int,
    max_moves: int,
    min_spread: int,
    max_best_moves: int,
    max_attempts: int,
    center_bias: float,
    metadata_output: Optional[Path],
) -> Dict:
    solver = PonsSolver(strict=True)
    if not solver.is_available():
        raise RuntimeError("Pons solver is required. Run scripts/bootstrap_gpu.sh first.")

    rng = random.Random(seed)
    seen = set()
    positions: List[str] = []
    records: List[Dict] = []
    phase_counts: Dict[str, int] = {"opening": 0, "middle": 0, "late": 0}
    best_col_counts: Dict[int, int] = {col: 0 for col in range(7)}
    attempts = 0
    start = time.time()
    last_log = 0

    while len(positions) < n_positions and attempts < max_attempts:
        attempts += 1
        env = _sample_random_position(
            rng,
            min_moves=min_moves,
            max_moves=max_moves,
            center_bias=center_bias,
        )
        if env is None:
            continue

        moves = env.to_move_sequence()
        if moves in seen:
            continue
        seen.add(moves)

        scores = solver.analyze(env)
        if len(scores) < 2:
            continue
        spread, max_score, best_moves = _score_stats(scores)
        if spread < min_spread:
            continue
        if len(best_moves) > max_best_moves:
            continue

        positions.append(moves)
        phase_name = _phase(len(moves))
        phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
        for col in best_moves:
            best_col_counts[col] = best_col_counts.get(col, 0) + 1
        records.append(
            {
                "moves": moves,
                "move_count": len(moves),
                "phase": phase_name,
                "score_spread": spread,
                "max_score": max_score,
                "legal_moves": sorted(scores),
                "best_moves": sorted(best_moves),
                "scores": {str(col): score for col, score in sorted(scores.items())},
            }
        )

        if len(positions) - last_log >= max(1, n_positions // 10):
            elapsed = time.time() - start
            rate = len(positions) / elapsed if elapsed else 0.0
            eta = (n_positions - len(positions)) / rate if rate else 0.0
            print(
                f"  Hard buffer: {len(positions)}/{n_positions} "
                f"({100 * len(positions) / n_positions:.0f}%) | "
                f"attempts={attempts} | {rate:.1f}/s | eta={eta:.0f}s",
                flush=True,
            )
            last_log = len(positions)

    if len(positions) < n_positions:
        raise RuntimeError(
            f"Only built {len(positions)}/{n_positions} positions after {attempts} attempts. "
            "Try lowering --min_spread or --max_best_moves, or raising --max_attempts."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"{output} already exists.")
    output.write_text(json.dumps(positions, indent=2))

    spreads = [record["score_spread"] for record in records]
    summary = {
        "output": str(output),
        "n_positions": len(positions),
        "attempts": attempts,
        "seed": seed,
        "min_moves": min_moves,
        "max_moves": max_moves,
        "min_spread": min_spread,
        "max_best_moves": max_best_moves,
        "center_bias": center_bias,
        "phase_counts": phase_counts,
        "best_col_counts": best_col_counts,
        "spread_min": min(spreads),
        "spread_mean": statistics.fmean(spreads),
        "spread_median": statistics.median(spreads),
        "spread_max": max(spreads),
    }

    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        if metadata_output.exists():
            raise FileExistsError(f"{metadata_output} already exists.")
        metadata_output.write_text(
            json.dumps({"summary": summary, "records": records}, indent=2, sort_keys=True)
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hard Connect Four training buffer.")
    parser.add_argument(
        "--output",
        default="experiments/opponent_next_move/data/hard_position_buffer.json",
    )
    parser.add_argument("--metadata_output", default=None)
    parser.add_argument("--n_positions", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--min_moves", type=int, default=6)
    parser.add_argument("--max_moves", type=int, default=32)
    parser.add_argument("--min_spread", type=int, default=8)
    parser.add_argument("--max_best_moves", type=int, default=2)
    parser.add_argument("--max_attempts", type=int, default=100000)
    parser.add_argument("--center_bias", type=float, default=0.35)
    args = parser.parse_args()

    metadata_output = Path(args.metadata_output) if args.metadata_output else None
    summary = build_hard_buffer(
        Path(args.output),
        n_positions=args.n_positions,
        seed=args.seed,
        min_moves=args.min_moves,
        max_moves=args.max_moves,
        min_spread=args.min_spread,
        max_best_moves=args.max_best_moves,
        max_attempts=args.max_attempts,
        center_bias=args.center_bias,
        metadata_output=metadata_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
