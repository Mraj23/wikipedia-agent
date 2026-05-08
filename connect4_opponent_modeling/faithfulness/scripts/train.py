"""CLI entry point for the Tinker faithfulness GRPO loop.

Usage:
    TINKER_API_KEY=... python -m faithfulness.scripts.train \
        --base-model Qwen/Qwen3-4B \
        --n-steps 2000 \
        --batch-size 32 --group-size 8 \
        --log-path faithfulness/data/runs/main

Set --truth-lambda > 0 for the optional reward-shaping ablation that adds
`+ lambda * mean(claim_truth)` to the per-rollout reward.
"""

from __future__ import annotations

import argparse
import sys

from faithfulness.rl.trainer import TrainerConfig, train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--truth-lambda", type=float, default=0.0)
    parser.add_argument("--log-path", default="faithfulness/data/runs/default")
    parser.add_argument(
        "--eval-set-path",
        default="faithfulness/data/eval_boards.jsonl",
    )
    args = parser.parse_args()

    cfg = TrainerConfig(
        base_model=args.base_model,
        base_url=args.base_url,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        group_size=args.group_size,
        n_steps=args.n_steps,
        save_every=args.save_every,
        seed=args.seed,
        truth_lambda=args.truth_lambda,
        log_path=args.log_path,
        eval_set_path=args.eval_set_path,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
