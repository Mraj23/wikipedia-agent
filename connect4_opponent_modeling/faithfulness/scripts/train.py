"""CLI entry point for the Tinker faithfulness GRPO loop.

Usage:
    TINKER_API_KEY=... python -m faithfulness.scripts.train \
        --base-model Qwen/Qwen3-4B-Instruct-2507 \
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
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--condition",
        choices=("claims_rationale", "move_only"),
        default="claims_rationale",
        help="Training output format. Evaluation can still use the full claims prompt.",
    )
    parser.add_argument(
        "--renderer",
        default="qwen3_instruct",
        help="Tinker cookbook renderer name, e.g. qwen3_instruct or qwen3.",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--final-ttl-seconds",
        type=int,
        default=None,
        help="TTL for final/eval sampler checkpoints. Tinker default when omitted.",
    )
    parser.add_argument(
        "--rollout-ttl-seconds",
        type=int,
        default=3600,
        help="TTL for short-lived rollout sampler checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--truth-lambda", type=float, default=0.0)
    parser.add_argument("--log-path", default="faithfulness/data/runs/default")
    parser.add_argument(
        "--eval-set-path",
        default="faithfulness/data/eval_boards.jsonl",
    )
    parser.add_argument(
        "--training-pool-path",
        default="faithfulness/data/training_positions.jsonl",
    )
    args = parser.parse_args()

    cfg = TrainerConfig(
        base_model=args.base_model,
        condition=args.condition,
        renderer=args.renderer,
        base_url=args.base_url,
        project_id=args.project_id,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        group_size=args.group_size,
        n_steps=args.n_steps,
        save_every=args.save_every,
        final_ttl_seconds=args.final_ttl_seconds,
        rollout_ttl_seconds=args.rollout_ttl_seconds,
        seed=args.seed,
        truth_lambda=args.truth_lambda,
        log_path=args.log_path,
        eval_set_path=args.eval_set_path,
        training_pool_path=args.training_pool_path,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
