"""CLI: generate the faithfulness eval set and lock it to disk.

Usage:
    python -m faithfulness.scripts.generate_eval_set \
        --output faithfulness/data/eval_boards.jsonl \
        --n-per-category 100 \
        --seed 42

Once locked, the file must not be regenerated. See
faithfulness/eval/board_generator.py::lock_eval_set for the immutability
contract (mirrors data/probe_positions_locked.jsonl).
"""

import argparse
import logging
import sys

from env.pons_wrapper import PonsSolver
from faithfulness.eval.board_generator import lock_eval_set


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="faithfulness/data/eval_boards.jsonl",
        help="Output JSONL path (will refuse to overwrite).",
    )
    parser.add_argument("--n-per-category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow Pons minimax fallback (NOT recommended for locking).",
    )
    parser.add_argument(
        "--fallback-depth",
        type=int,
        default=8,
        help="Minimax search depth when Pons binary is unavailable. "
        "Lower = faster but less accurate (4 for development, 8 for production).",
    )
    parser.add_argument(
        "--candidate-games",
        type=int,
        default=1500,
        help="Number of random self-play games to seed the candidate pool.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    solver = PonsSolver(strict=not args.allow_fallback, fallback_depth=args.fallback_depth)
    lock_eval_set(
        args.output,
        n_per_category=args.n_per_category,
        seed=args.seed,
        solver=solver,
        candidate_games=args.candidate_games,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
