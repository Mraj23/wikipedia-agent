"""GRPO hyperparameters for all experimental conditions.

Provides condition-specific configurations with correct reward weights.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class GRPOConfig:
    """Configuration for GRPO (Group Relative Policy Optimization) training.

    Attributes:
        condition: Experimental condition label ('B', 'C', 'D', or 'E').
        model_path: Path to SFT warmup checkpoint.
        game_steps: Total training game steps.
        group_size: Number of responses sampled per prompt for GRPO.
        clip_ratio: PPO-style clipping ratio.
        kl_coef: KL divergence penalty coefficient.
        lr: Learning rate.
        max_tokens: Maximum generation tokens.
        eval_every: Run evaluation every N steps.
        opponent_depth: Minimax depth for opponent in conditions C/D/E.
        use_rae: Whether to use Role-conditioned Advantage Estimation (SPIRAL).
        reward_weights: Reward component weights, populated per condition.
    """

    condition: str = "B"
    model_path: str = "checkpoints/condition_a"
    game_steps: int = 5000
    group_size: int = 8
    clip_ratio: float = 0.2
    kl_coef: float = 0.001
    lr: float = 3e-6
    max_tokens: int = 512
    eval_every: int = 1000
    opponent_depth: int = 6
    use_rae: bool = True
    reward_weights: Optional[Dict[str, float]] = field(default=None)


# Condition-specific reward weight presets
_REWARD_WEIGHTS: Dict[str, Dict[str, float]] = {
    "B": {},  # sparse win/loss only, no weighted components
    "C": {"move": 0.6, "terminal": 0.3, "format": 0.1},
    "D": {"move": 0.5, "future": 0.2, "terminal": 0.2, "format": 0.1},
    "E": {"move": 0.5, "pred": 0.2, "terminal": 0.2, "format": 0.1},
}


def get_config(
    condition: str, model_path: str = "checkpoints/condition_a"
) -> GRPOConfig:
    """Return a condition-specific GRPO configuration.

    Args:
        condition: One of 'B', 'C', 'D', 'E'.
        model_path: Path to the SFT warmup checkpoint.

    Returns:
        GRPOConfig with condition-appropriate reward weights.

    Raises:
        ValueError: If condition is not one of B-E.
    """
    if condition not in _REWARD_WEIGHTS:
        raise ValueError(
            f"Invalid condition '{condition}'. Must be one of: {list(_REWARD_WEIGHTS.keys())}"
        )

    return GRPOConfig(
        condition=condition,
        model_path=model_path,
        reward_weights=_REWARD_WEIGHTS[condition],
    )


if __name__ == "__main__":
    print("=== GRPO Configs ===\n")
    for cond in ["B", "C", "D", "E"]:
        cfg = get_config(cond)
        print(f"Condition {cond}:")
        print(f"  reward_weights: {cfg.reward_weights}")
        print(f"  game_steps: {cfg.game_steps}")
        print(f"  group_size: {cfg.group_size}")
        print(f"  opponent_depth: {cfg.opponent_depth}")
        print(f"  use_rae: {cfg.use_rae}")
        print()
