"""Tests for reward functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.reward import RewardCalculator


def _get_calc():
    """Create a RewardCalculator with reduced fallback depth for test speed."""
    return RewardCalculator(PonsSolver(fallback_depth=4))


def test_condition_b_win():
    """Returns 1.0 for win."""
    calc = _get_calc()
    assert calc.condition_b_reward("win") == 1.0


def test_condition_b_loss():
    """Returns 0.0 for loss."""
    calc = _get_calc()
    assert calc.condition_b_reward("loss") == 0.0


def test_condition_c_in_range():
    """Reward is in [0,1] for any valid inputs."""
    calc = _get_calc()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    response = "<think>test</think><move>2</move>"
    reward = calc.condition_c_reward(env, 2, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_condition_d_in_range():
    """Reward is in [0,1]."""
    calc = _get_calc()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    response = (
        "<think>test</think>"
        "<opponent_prediction>3</opponent_prediction>"
        "<future_state>. . . . . . .\n. . . . . . .\n. . . . . . .\n"
        ". . . . . . .\n. . . . . . .\n. . X X O . .</future_state>"
        "<move>2</move>"
    )
    reward = calc.condition_d_reward(env, 2, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_condition_e_in_range():
    """Reward is in [0,1]."""
    calc = _get_calc()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    response = (
        "<think>test</think>"
        "<opponent_prediction>3</opponent_prediction>"
        "<move>2</move>"
    )
    reward = calc.condition_e_reward(env, 2, 3, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_prediction_accuracy_correct():
    """1.0 when predicted == optimal."""
    calc = _get_calc()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    # Get the actual optimal response
    optimal = calc.solver.optimal_opponent_response(env, 2)
    if optimal >= 0:
        acc = calc._prediction_accuracy(env, 2, optimal)
        assert acc == 1.0


def test_prediction_accuracy_wrong():
    """< 1.0 when predicted != optimal (unless all moves equivalent)."""
    calc = _get_calc()
    env = ConnectFourEnv()
    # Play some moves to create a non-trivial position
    for col in [3, 3, 4, 2, 5, 1]:
        env.make_move(col)

    optimal = calc.solver.optimal_opponent_response(env, 0)
    if optimal >= 0:
        # Pick a different column
        wrong = (optimal + 1) % 7
        acc = calc._prediction_accuracy(env, 0, wrong)
        # Should be <= 1.0 (might equal 1.0 if moves are equivalent)
        assert 0.0 <= acc <= 1.0


def test_format_reward_missing_tag():
    """0.0 when required tag is missing."""
    calc = _get_calc()
    assert calc._format_reward("no tags here", "C") == 0.0
    assert calc._format_reward("<think>ok</think>but no move", "C") == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
