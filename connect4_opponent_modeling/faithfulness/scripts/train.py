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
        choices=("claims_rationale", "move_only", "tactical_claims"),
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
    parser.add_argument("--target-accepted-groups", type=int, default=24)
    parser.add_argument("--candidate-batch-multiplier", type=int, default=4)
    parser.add_argument("--max-group-attempts", type=int, default=4)
    parser.add_argument("--min-reward-std", type=float, default=1e-6)
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
        "--fast-eval-every",
        type=int,
        default=0,
        help="Run a small no-causal held-out eval every N steps. 0 disables.",
    )
    parser.add_argument(
        "--fast-eval-n-boards",
        type=int,
        default=25,
        help="Number of held-out boards for each fast eval.",
    )
    parser.add_argument(
        "--disable-fast-eval",
        action="store_true",
        help="Force fast eval off even if --fast-eval-every is supplied.",
    )
    parser.add_argument(
        "--training-pool-path",
        default="faithfulness/data/training_positions.jsonl",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Base sampling temperature. Floor for the dynamic-temperature decay.",
    )
    parser.add_argument(
        "--temperature-max",
        type=float,
        default=1.5,
        help="Cap for dynamic-temperature bumps when most_common_move_pct exceeds threshold.",
    )
    parser.add_argument(
        "--temperature-diversity-threshold",
        type=float,
        default=0.8,
        help=(
            "If the fraction of completions choosing the most-common column "
            "exceeds this in a step, temperature is bumped by +0.1."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        default="faithfulness",
        help="Weights & Biases project. Used when WANDB_API_KEY is set.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Weights & Biases run name. Defaults to log-path basename.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging even if WANDB_API_KEY is set.",
    )
    parser.add_argument(
        "--rollout-dump-every",
        type=int,
        default=5,
        help="Write per-rollout records to rollouts.jsonl every N steps. 0 disables.",
    )
    parser.add_argument(
        "--kl-to-base-beta",
        type=float,
        default=0.0,
        help="If > 0, subtract beta * KL(policy || base) from per-rollout reward.",
    )
    parser.add_argument(
        "--diversity-bonus-beta",
        type=float,
        default=0.0,
        help="If > 0, add beta * (1 - within-group share of chosen_move) to reward.",
    )
    parser.add_argument(
        "--rollout-dump-max-per-step",
        type=int,
        default=0,
        help="Cap the number of rollouts dumped per dump step. 0 means unlimited.",
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
        target_accepted_groups=args.target_accepted_groups,
        candidate_batch_multiplier=args.candidate_batch_multiplier,
        max_group_attempts=args.max_group_attempts,
        min_reward_std=args.min_reward_std,
        final_ttl_seconds=args.final_ttl_seconds,
        rollout_ttl_seconds=args.rollout_ttl_seconds,
        fast_eval_every=0 if args.disable_fast_eval else args.fast_eval_every,
        fast_eval_n_boards=args.fast_eval_n_boards,
        seed=args.seed,
        truth_lambda=args.truth_lambda,
        log_path=args.log_path,
        eval_set_path=args.eval_set_path,
        training_pool_path=args.training_pool_path,
        temperature=args.temperature,
        temperature_max=args.temperature_max,
        temperature_diversity_threshold=args.temperature_diversity_threshold,
        wandb_enabled=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        rollout_dump_every=args.rollout_dump_every,
        rollout_dump_max_per_step=args.rollout_dump_max_per_step,
        kl_to_base_beta=args.kl_to_base_beta,
        diversity_bonus_beta=args.diversity_bonus_beta,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
