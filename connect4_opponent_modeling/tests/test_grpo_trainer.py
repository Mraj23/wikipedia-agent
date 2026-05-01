"""Tests for GRPO trainer reward computation and integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from spiral.game_playout import play_to_completion
from training.minimax import MinimaxSolver
from training.prompts import format_prompt, parse_response, validate_response
from training.reward import RewardCalculator


_solver = PonsSolver()
_calc = RewardCalculator(_solver)
_minimax = MinimaxSolver(depth=4)


def _make_env_with_moves(*cols):
    env = ConnectFourEnv()
    for col in cols:
        env.make_move(col)
    return env


def test_condition_b_reward_dispatch():
    env = _make_env_with_moves(3, 3, 4, 4, 2, 2)
    move = _minimax.best_move(env)
    result = play_to_completion(env, move, _minimax, env.current_player())
    reward = _calc.condition_b_reward(result)
    assert 0.0 <= reward <= 1.0


def test_condition_c_reward_dispatch():
    env = _make_env_with_moves(3, 3, 4, 4)
    response = "<reasoning>I should play center.</reasoning><answer>3</answer>"
    reward = _calc.condition_c_reward(env, 3, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_condition_d_reward_dispatch():
    env = _make_env_with_moves(3, 3, 4, 4)
    response = (
        "<reasoning>Building.</reasoning>"
        "<future_state>\n. . . . . . .\n. . . . . . .\n. . . . . . .\n"
        ". . . . . . .\n. . . X . . .\n. . . X X . .\n</future_state>"
        "<opponent_prediction>3</opponent_prediction>"
        "<answer>4</answer>"
    )
    reward = _calc.condition_d_reward(env, 4, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_condition_e_reward_dispatch():
    env = _make_env_with_moves(3, 3, 4, 4)
    response = (
        "<reasoning>Opponent will respond center.</reasoning>"
        "<future_state>\n. . . . . . .\n. . . . . . .\n. . . . . . .\n"
        ". . . . . . .\n. . . X . . .\n. . . X X . .\n</future_state>"
        "<opponent_prediction>3</opponent_prediction>"
        "<answer>2</answer>"
    )
    reward = _calc.condition_e_reward(env, 2, 3, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_invalid_response_gets_zero_format_reward():
    assert RewardCalculator._format_reward("I play column 3", "C") == 0.0


def test_valid_response_gets_format_reward():
    response = "<reasoning>test</reasoning><answer>3</answer>"
    assert RewardCalculator._format_reward(response, "C") == 1.0


def test_parse_and_validate_condition_e():
    response = (
        "<reasoning>analysis</reasoning>"
        "<future_state>\n. . . . . . .\n. . . . . . .\n. . . . . . .\n"
        ". . . . . . .\n. . . X . . .\n. . . X X . .\n</future_state>"
        "<opponent_prediction>4</opponent_prediction>"
        "<answer>3</answer>"
    )
    parsed = parse_response(response, "E")
    assert parsed["move"] == 3
    assert parsed["opponent_prediction"] == 4

    env = _make_env_with_moves(3, 3)
    valid, _ = validate_response(parsed, "E", env.legal_moves())
    assert valid


def test_prompt_formatting():
    env = _make_env_with_moves(3, 3, 4)
    for cond in ["B", "C", "D", "E"]:
        prompt = format_prompt(cond, env)
        assert len(prompt) > 50
        assert "Legal columns" in prompt
