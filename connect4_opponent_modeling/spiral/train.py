"""CLI entry point for SPIRAL/GRPO training.

Usage:
    python -m spiral.train --condition B --model checkpoints/condition_a --log_dir logs/condition_b
"""

import argparse
import logging
import sys
from pathlib import Path

from training.grpo_config import get_config
from spiral.grpo_trainer import GRPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPIRAL: GRPO RL training for Connect Four opponent modeling"
    )
    parser.add_argument(
        "--condition", type=str, required=True,
        choices=["B", "C", "D", "E", "G"],
        help="Experimental condition (B, C, D, E, or G)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B",
        help="Model name/path (HuggingFace ID or local checkpoint)",
    )
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Directory for training logs (default: logs/condition_{letter})",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to grpo_config.py (unused, config loaded via get_config)",
    )
    parser.add_argument(
        "--game_steps", type=int, default=None,
        help="Override game_steps from config (useful for smoke tests)",
    )
    parser.add_argument(
        "--group_size", type=int, default=None,
        help="Override group_size from config (useful for smoke tests)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--use_vllm", action="store_true",
        help="Use vLLM for fast generation (GPU only, requires vllm installed)",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging for real-time monitoring",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="connect4-opponent-modeling",
        help="W&B project name (default: connect4-opponent-modeling)",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="W&B run name (default: condition_{letter})",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("spiral")

    # Model path: can be a HuggingFace model ID or local checkpoint path
    model_path = args.model
    local_path = Path(model_path)
    if local_path.exists():
        logger.info("Using local model: %s", model_path)
    else:
        logger.info("Using HuggingFace model: %s (will download on first use)", model_path)

    # Load config
    config = get_config(args.condition, model_path)
    if args.game_steps is not None:
        config.game_steps = args.game_steps
    if args.group_size is not None:
        config.group_size = args.group_size
    if args.seed is not None:
        config.seed = args.seed
    if args.use_vllm:
        config.use_vllm = True
    if args.wandb:
        config.use_wandb = True
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name

    # Log directory
    log_dir = args.log_dir or f"logs/condition_{args.condition.lower()}"

    logger.info("=== SPIRAL Training ===")
    logger.info("Condition: %s", config.condition)
    logger.info("Model: %s", config.model_path)
    logger.info("Game steps: %d", config.game_steps)
    logger.info("Group size: %d", config.group_size)
    logger.info("KL coef: %f", config.kl_coef)
    logger.info("Clip ratio: %f", config.clip_ratio)
    logger.info("Use RAE: %s", config.use_rae)
    logger.info("Reward weights: %s", config.reward_weights)
    logger.info("W&B: %s", "enabled" if config.use_wandb else "disabled")
    logger.info("Log dir: %s", log_dir)

    # Train
    trainer = GRPOTrainer(config=config, log_dir=log_dir)
    trainer.train()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
