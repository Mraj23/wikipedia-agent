"""Tests for trainer group-selection diagnostics without Tinker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from faithfulness.rl.trainer import group_selection_diagnostics


def test_group_selection_accepts_reward_variance():
    diag = group_selection_diagnostics(
        rewards=[0.2, 0.2, -0.5, 0.2],
        moves=[3, 3, 4, 3],
        min_reward_std=1e-6,
    )
    assert diag["accepted"] is True
    assert diag["skip_reason"] is None
    assert diag["unique_moves"] == 2
    assert diag["most_common_move_pct"] == 0.75


def test_group_selection_rejects_identical_rewards_even_when_wrong_move_repeats():
    diag = group_selection_diagnostics(
        rewards=[-0.7, -0.7, -0.7, -0.7],
        moves=[2, 2, 2, 2],
        min_reward_std=1e-6,
    )
    assert diag["accepted"] is False
    assert diag["skip_reason"] == "zero_reward_variance"
    assert diag["identical_move_group"] is True
    assert diag["unique_moves"] == 1
