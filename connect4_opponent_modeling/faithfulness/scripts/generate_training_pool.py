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

MIX_PRESETS = {
    "tactical": {
        "must_block_threat": 8000,
        "has_immediate_win": 5000,
        "has_double_threat_move": 5000,
        "has_forcing_threat": 5000,
        "quiet": 2000,
    },
}


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
    parser.add_argument(
        "--mix",
        choices=sorted(MIX_PRESETS),
        default=None,
        help=(
            "Stratified preset of per-tag targets. 'tactical' biases the pool "
            "toward must_block_threat / has_immediate_win / has_double_threat_move "
            "/ has_forcing_threat to raise GRPO reward variance. Overrides "
            "--n-positions and --cap-per-tag."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    target_per_tag = None
    n_positions = args.n_positions
    max_games = args.max_games

    if args.mix is not None:
        target_per_tag = dict(MIX_PRESETS[args.mix])
        n_positions = sum(target_per_tag.values())
        # Tactical positions are rare in random self-play; raise the games
        # ceiling so rare tags can hit their targets.
        if max_games is None:
            max_games = max(20_000, n_positions * 2)
    elif args.cap_per_tag is not None:
        target_per_tag = {
            "has_immediate_win": args.cap_per_tag,
            "must_block_threat": args.cap_per_tag,
            "has_double_threat_move": args.cap_per_tag,
            "has_forcing_threat": args.cap_per_tag,
            "quiet": args.cap_per_tag,
        }

    records = generate_training_pool(
        n_positions=n_positions,
        seed=args.seed,
        max_games=max_games,
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
