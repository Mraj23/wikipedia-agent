"""Tests for the KL-to-base + within-group diversity reward shaping."""

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from faithfulness.rl.trainer import (
    TrainerConfig,
    apply_kl_penalty,
    compute_diversity_bonus,
)


def test_diversity_bonus_minority_gets_full_bonus():
    # 4 rollouts, three picked move 3 and one picked move 5.
    moves = [3, 3, 3, 5]
    # The minority (index 3) has share 1/4 -> bonus = beta * (1 - 0.25) = 0.075
    assert compute_diversity_bonus(moves, my_index=3, beta=0.1) == pytest.approx(0.075)
    # Majority (index 0) has share 3/4 -> bonus = beta * 0.25 = 0.025
    assert compute_diversity_bonus(moves, my_index=0, beta=0.1) == pytest.approx(0.025)


def test_diversity_bonus_none_chosen_move_yields_zero():
    moves = [3, None, 3, 4]
    # None entries skipped from "present"; index 1's chosen_move is None -> 0.
    assert compute_diversity_bonus(moves, my_index=1, beta=0.5) == 0.0


def test_diversity_bonus_zero_beta_short_circuits():
    moves = [1, 2, 3, 1]
    assert compute_diversity_bonus(moves, my_index=0, beta=0.0) == 0.0


def test_diversity_bonus_unanimous_group_gets_zero():
    moves = [4, 4, 4, 4]
    assert compute_diversity_bonus(moves, my_index=0, beta=0.2) == 0.0


def test_apply_kl_penalty_basic():
    # policy logprob > base -> positive KL -> reward decreases.
    adjusted, kl = apply_kl_penalty(
        reward=1.0,
        sum_policy_logprob=-5.0,
        sum_base_logprob=-7.0,
        beta=0.1,
    )
    assert kl == pytest.approx(2.0)
    assert adjusted == pytest.approx(1.0 - 0.1 * 2.0)


def test_apply_kl_penalty_negative_kl_increases_reward():
    # If base assigns higher likelihood than policy, KL is negative under this
    # one-sample MC estimator; the penalty term flips sign.
    adjusted, kl = apply_kl_penalty(
        reward=0.5,
        sum_policy_logprob=-9.0,
        sum_base_logprob=-7.0,
        beta=0.05,
    )
    assert kl == pytest.approx(-2.0)
    assert adjusted == pytest.approx(0.5 - 0.05 * -2.0)


def test_apply_kl_penalty_zero_beta_returns_raw():
    adjusted, kl = apply_kl_penalty(
        reward=0.42,
        sum_policy_logprob=-1.0,
        sum_base_logprob=-2.0,
        beta=0.0,
    )
    assert adjusted == 0.42
    assert kl == pytest.approx(1.0)


def test_trainer_skips_reference_client_when_beta_zero():
    """When kl_to_base_beta == 0, the trainer must not try to build a base
    sampling client. We assert this by inspecting the source: the construction
    is gated on `cfg.kl_to_base_beta > 0.0`."""
    import inspect

    from faithfulness.rl import trainer as trainer_mod

    src = inspect.getsource(trainer_mod._train_async)
    assert "if cfg.kl_to_base_beta > 0.0" in src
    assert "reference_sampling_client = None" in src

    # And the dataclass default is off.
    cfg = TrainerConfig()
    assert cfg.kl_to_base_beta == 0.0
    assert cfg.diversity_bonus_beta == 0.0
