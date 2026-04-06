"""Tests for GRPO trainer reward computation and integration.

These tests validate reward dispatch without loading a full model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.minimax import MinimaxSolver
from training.reward import RewardCalculator
from training.prompts import parse_response, validate_response, format_prompt
from spiral.game_playout import play_to_completion


# Shared solver for tests
_solver = PonsSolver()
_calc = RewardCalculator(_solver)
_minimax = MinimaxSolver(depth=4)


def _make_env_with_moves(*cols):
    """Create an env with the given moves played."""
    env = ConnectFourEnv()
    for col in cols:
        env.make_move(col)
    return env


def test_condition_b_reward_dispatch():
    """Condition B reward uses game playout result."""
    env = _make_env_with_moves(3, 3, 4, 4, 2, 2)
    move = _minimax.best_move(env)
    result = play_to_completion(env, move, _minimax, env.current_player())
    reward = _calc.condition_b_reward(result)
    assert 0.0 <= reward <= 1.0


def test_condition_c_reward_dispatch():
    """Condition C reward computes composite score."""
    env = _make_env_with_moves(3, 3, 4, 4)
    response = "<think>I should play center.</think><move>3</move>"
    reward = _calc.condition_c_reward(env, 3, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_condition_d_reward_dispatch():
    """Condition D reward includes future state and opponent prediction tag."""
    env = _make_env_with_moves(3, 3, 4, 4)
    response = (
        "<think>Building.</think>"
        "<opponent_prediction>3</opponent_prediction>"
        "<future_state>\n. . . . . . .\n. . . . . . .\n. . . . . . .\n"
        ". . . . . . .\n. . . X . . .\n. . . X X . .\n</future_state>"
        "<move>4</move>"
    )
    reward = _calc.condition_d_reward(env, 4, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_condition_e_reward_dispatch():
    """Condition E reward includes opponent prediction."""
    env = _make_env_with_moves(3, 3, 4, 4)
    response = (
        "<think>Opponent will respond center.</think>"
        "<opponent_prediction>3</opponent_prediction>"
        "<move>2</move>"
    )
    reward = _calc.condition_e_reward(env, 2, 3, "ongoing", response)
    assert 0.0 <= reward <= 1.0


def test_invalid_response_gets_zero_format_reward():
    """Responses missing required tags should get 0 format reward."""
    response = "I play column 3"
    format_r = RewardCalculator._format_reward(response, "C")
    assert format_r == 0.0


def test_valid_response_gets_format_reward():
    """Valid responses should get 1.0 format reward."""
    response = "<think>test</think><move>3</move>"
    format_r = RewardCalculator._format_reward(response, "C")
    assert format_r == 1.0


def test_parse_and_validate_condition_e():
    """Condition E response parsing extracts opponent prediction."""
    response = (
        "<think>analysis</think>"
        "<opponent_prediction>4</opponent_prediction>"
        "<move>3</move>"
    )
    parsed = parse_response(response, "E")
    assert parsed["move"] == 3
    assert parsed["opponent_prediction"] == 4

    env = _make_env_with_moves(3, 3)
    valid, _ = validate_response(parsed, "E", env.legal_moves())
    assert valid


def test_prompt_formatting():
    """format_prompt should produce a non-empty string with board."""
    env = _make_env_with_moves(3, 3, 4)
    for cond in ["B", "C", "D", "E"]:
        prompt = format_prompt(cond, env)
        assert len(prompt) > 50
        assert "Legal moves" in prompt
