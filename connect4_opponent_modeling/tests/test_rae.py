"""Tests for Role-conditioned Advantage Estimation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from spiral.rae import compute_advantages, _standard_advantages, _rae_advantages


def test_standard_advantages_zero_mean():
    """Standard advantages should have approximately zero mean."""
    rewards = torch.tensor([0.2, 0.5, 0.8, 0.3])
    adv = _standard_advantages(rewards)
    assert abs(adv.mean().item()) < 1e-5


def test_standard_advantages_uniform_rewards():
    """Uniform rewards should yield near-zero advantages."""
    rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])
    adv = _standard_advantages(rewards)
    assert torch.all(torch.abs(adv) < 1e-4)


def test_rae_with_single_component():
    """RAE with one component should match standard GRPO."""
    rewards = [0.2, 0.5, 0.8, 0.3]
    components = [{"move": r} for r in rewards]
    weights = {"move": 1.0}

    rae_adv = compute_advantages(rewards, components, weights, use_rae=True)
    std_adv = compute_advantages(rewards, None, {}, use_rae=False)

    assert torch.allclose(rae_adv, std_adv, atol=1e-5)


def test_rae_with_multiple_components():
    """RAE should produce valid advantages with multiple components."""
    components = [
        {"move": 0.8, "terminal": 0.0, "format": 1.0},
        {"move": 0.3, "terminal": 0.0, "format": 1.0},
        {"move": 0.6, "terminal": 1.0, "format": 0.0},
        {"move": 0.9, "terminal": 0.0, "format": 1.0},
    ]
    weights = {"move": 0.6, "terminal": 0.3, "format": 0.1}
    rewards = [0.6 * c["move"] + 0.3 * c["terminal"] + 0.1 * c["format"] for c in components]

    adv = compute_advantages(rewards, components, weights, use_rae=True)
    assert adv.shape == (4,)
    assert not torch.any(torch.isnan(adv))


def test_condition_b_degenerates_to_standard():
    """Condition B (empty weights) should use standard GRPO."""
    rewards = [1.0, 0.0, 0.5, 0.0]
    adv = compute_advantages(rewards, None, {}, use_rae=True)
    std_adv = compute_advantages(rewards, None, {}, use_rae=False)
    assert torch.allclose(adv, std_adv, atol=1e-5)


def test_use_rae_false_ignores_components():
    """When use_rae=False, components should be ignored."""
    rewards = [0.2, 0.8]
    components = [{"move": 0.2, "terminal": 0.0}, {"move": 0.8, "terminal": 1.0}]
    weights = {"move": 0.6, "terminal": 0.3}

    adv_no_rae = compute_advantages(rewards, components, weights, use_rae=False)
    adv_no_comp = compute_advantages(rewards, None, {}, use_rae=False)
    assert torch.allclose(adv_no_rae, adv_no_comp, atol=1e-5)


def test_advantages_dtype():
    """Advantages should be float32."""
    rewards = [0.5, 0.3, 0.7]
    adv = compute_advantages(rewards, None, {}, use_rae=False)
    assert adv.dtype == torch.float32
