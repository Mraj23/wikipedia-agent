"""Role-conditioned Advantage Estimation (RAE) for SPIRAL.

Standard GRPO normalizes advantages across the full reward. RAE splits
rewards into per-component advantages and normalizes each independently,
then combines with the condition's reward weights. This prevents high-variance
components from dominating the gradient signal.
"""

import torch
from typing import Dict, List, Optional


def compute_advantages(
    rewards: List[float],
    reward_components: Optional[List[Dict[str, float]]],
    reward_weights: Dict[str, float],
    use_rae: bool = True,
) -> torch.Tensor:
    """Compute GRPO advantages, optionally with RAE.

    Args:
        rewards: Total reward per completion in the group.
        reward_components: Per-completion dicts of {component_name: value}.
            Required when use_rae=True and reward_weights is non-empty.
            Each dict should have the same keys as reward_weights.
        reward_weights: Condition's reward component weights (e.g. {"move": 0.6, ...}).
        use_rae: Whether to use Role-conditioned Advantage Estimation.

    Returns:
        Tensor of advantages, shape (group_size,).
    """
    rewards_t = torch.tensor(rewards, dtype=torch.float32)

    if not use_rae or not reward_weights or reward_components is None:
        return _standard_advantages(rewards_t)

    return _rae_advantages(reward_components, reward_weights)


def _standard_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Standard GRPO advantage: (r_i - mean) / (std + eps).

    Args:
        rewards: Tensor of rewards, shape (group_size,).

    Returns:
        Tensor of advantages, shape (group_size,).
    """
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + 1e-8)


def _rae_advantages(
    reward_components: List[Dict[str, float]],
    reward_weights: Dict[str, float],
) -> torch.Tensor:
    """Role-conditioned Advantage Estimation.

    Normalizes each reward component independently, then combines
    with the condition's reward weights.

    Args:
        reward_components: Per-completion component values.
        reward_weights: Weight per component.

    Returns:
        Tensor of advantages, shape (group_size,).
    """
    group_size = len(reward_components)
    combined = torch.zeros(group_size, dtype=torch.float32)

    for component_name, weight in reward_weights.items():
        values = torch.tensor(
            [rc.get(component_name, 0.0) for rc in reward_components],
            dtype=torch.float32,
        )
        # Normalize this component independently
        mean = values.mean()
        std = values.std()
        # Skip components with zero variance (e.g., format reward always 1.0).
        # These provide no gradient signal and would produce ~1e8 spikes.
        if std < 1e-6:
            continue
        normalized = (values - mean) / (std + 1e-8)
        combined += weight * normalized

    return combined
