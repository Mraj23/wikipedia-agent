"""Tests for reward functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.reward import RewardCalculator


def _get_calc():
    return RewardCalculator(PonsSolver(fallback_depth=4))


def test_condition_b_win():
    calc = _get_calc()
    assert calc.condition_b_reward("win") == 1.0


def test_condition_b_loss():
    calc = _get_calc()
    assert calc.condition_b_reward("loss") == 0.0


def test_condition_c_in_range():
    calc = _get_calc()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    response = "<reasoning>test</reasoning><answer>2</answer>"
    reward = calc.condition_c_reward(env, 2, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_condition_d_in_range():
    calc = _get_calc()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    response = (
        "<reasoning>test</reasoning>"
        "<future_state>. . . . . . .\n. . . . . . .\n. . . . . . .\n"
        ". . . . . . .\n. . . . . . .\n. . X X O . .</future_state>"
        "<answer>2</answer>"
    )
    reward = calc.condition_d_reward(env, 2, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_condition_e_in_range():
    calc = _get_calc()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    response = (
        "<reasoning>test</reasoning>"
        "<opponent_prediction>3</opponent_prediction>"
        "<answer>2</answer>"
    )
    reward = calc.condition_e_reward(env, 2, 3, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_prediction_accuracy_correct():
    calc = _get_calc()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    optimal = calc.solver.optimal_opponent_response(env, 2)
    if optimal >= 0:
        acc = calc._prediction_accuracy(env, 2, optimal)
        assert acc == 1.0


def test_prediction_accuracy_wrong():
    calc = _get_calc()
    env = ConnectFourEnv()
    for col in [3, 3, 4, 2, 5, 1]:
        env.make_move(col)

    optimal = calc.solver.optimal_opponent_response(env, 0)
    if optimal >= 0:
        wrong = (optimal + 1) % 7
        acc = calc._prediction_accuracy(env, 0, wrong)
        assert 0.0 <= acc <= 1.0


def test_format_reward_missing_tag():
    calc = _get_calc()
    assert calc._format_reward("no tags here", "C") == 0.0
    assert calc._format_reward("<reasoning>ok</reasoning>but no answer", "C") == 0.0
