"""CLI: generate a large training-position pool for faithfulness RL.

Unlike the eval set, the training pool is regenerable and overwrites by
default. It contains only deterministic strategic-tag metadata — no Pons
scores cached. The trainer is expected to call the solver at training time
for the regret reward (cheaper than pre-scoring tens of thousands of
positions that may not all get sampled).

Usage:
    python -m faithfulness.scripts.generate_training_pool \
        --output faithfulness/data/training_positions.jsonl \
        --n-positions 50000 --seed 42
"""

import argparse
import json
import logging
import sys

from faithfulness.eval.training_pool import (
    generate_training_pool,
    stratify_summary,
    write_training_pool,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="faithfulness/data/training_positions.jsonl",
        help="Output JSONL path. Overwrites existing file.",
    )
    parser.add_argument("--n-positions", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Hard cap on random self-play games. Default scales with n-positions.",
    )
    parser.add_argument("--min-ply", type=int, default=1)
    parser.add_argument("--max-ply", type=int, default=41)
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip de-duplication by move sequence (faster, may include repeats).",
    )
    parser.add_argument(
        "--cap-per-tag",
        type=int,
        default=None,
        help="Optional per-position-tag cap to balance rare tags.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    target_per_tag = None
    if args.cap_per_tag is not None:
        # All five PositionTag values uniformly capped.
        target_per_tag = {
            "has_immediate_win": args.cap_per_tag,
            "must_block_threat": args.cap_per_tag,
            "has_double_threat_move": args.cap_per_tag,
            "has_forcing_threat": args.cap_per_tag,
            "quiet": args.cap_per_tag,
        }

    records = generate_training_pool(
        n_positions=args.n_positions,
        seed=args.seed,
        max_games=args.max_games,
        dedup=not args.no_dedup,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        target_per_tag=target_per_tag,
    )

    write_training_pool(records, args.output)
    summary = stratify_summary(records)
    logging.info("Wrote %d positions to %s", len(records), args.output)
    logging.info("Stratification: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
