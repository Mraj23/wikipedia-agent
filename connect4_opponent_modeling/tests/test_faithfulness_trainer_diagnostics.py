"""Tests for trainer group-selection diagnostics without Tinker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from faithfulness.rl.trainer import balanced_eval_subset, group_selection_diagnostics


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


def test_balanced_eval_subset_round_robins_categories():
    records = (
        [{"category": "a", "i": i} for i in range(10)]
        + [{"category": "b", "i": i} for i in range(10, 20)]
        + [{"category": "c", "i": i} for i in range(20, 30)]
    )
    subset = balanced_eval_subset(records, 9, seed=123)
    counts = {}
    for item in subset:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    assert counts == {"a": 3, "b": 3, "c": 3}


def test_balanced_eval_subset_avoids_prefix_category_bias():
    records = (
        [{"category": "a", "i": i} for i in range(20)]
        + [{"category": "b", "i": i} for i in range(20, 40)]
    )
    subset = balanced_eval_subset(records, 10, seed=0)
    categories = {item["category"] for item in subset}
    assert categories == {"a", "b"}
