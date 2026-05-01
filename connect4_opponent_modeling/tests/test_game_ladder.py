"""Tests for the canonical game ladder evaluator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyspiel

from eval.game_ladder import (
    PROMPT_STYLE_BASE,
    PROMPT_STYLE_OPPONENT_AWARE,
    make_prompt,
    parse_model_move,
    summarize_game_results,
)


def test_prompt_styles_are_distinct():
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    legal = state.legal_actions()

    base_prompt = make_prompt(state, "connect_four", legal, prompt_style=PROMPT_STYLE_BASE)
    opp_prompt = make_prompt(
        state,
        "connect_four",
        legal,
        prompt_style=PROMPT_STYLE_OPPONENT_AWARE,
    )

    assert "opponent's most likely reply" not in base_prompt
    assert "opponent's most likely reply" in opp_prompt
    assert "<opponent_prediction>" in opp_prompt
    assert "<future_state>" in opp_prompt


def test_parse_model_move_connect_four_number():
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    legal = state.legal_actions()

    move = parse_model_move("<answer>3</answer>", "connect_four", legal, state)
    assert move == 3


def test_parse_model_move_connect_four_text_inside_answer():
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    legal = state.legal_actions()

    move = parse_model_move("<answer>play column 3</answer>", "connect_four", legal, state)
    assert move == 3


def test_parse_model_move_requires_answer_tag():
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    legal = state.legal_actions()

    move = parse_model_move("play column 3", "connect_four", legal, state)
    assert move is None


def test_summarize_game_results_extracts_transfer_score():
    ladder = {
        "games": {
            "breakthrough": {
                "opponents": [
                    {
                        "opponent": "minimax-2",
                        "win_rate": 0.4,
                        "valid_rate": 0.9,
                        "clean_game_win_rate": 0.5,
                        "invalid_as_loss_win_rate": 0.3,
                    },
                    {
                        "opponent": "minimax-4",
                        "win_rate": 0.2,
                        "valid_rate": 0.9,
                        "clean_game_win_rate": 0.3,
                        "invalid_as_loss_win_rate": 0.1,
                    },
                    {
                        "opponent": "mcts-100",
                        "win_rate": 0.1,
                        "valid_rate": 0.9,
                        "clean_game_win_rate": 0.2,
                        "invalid_as_loss_win_rate": 0.0,
                    },
                ]
            }
        }
    }
    summary = summarize_game_results(ladder)
    assert summary["breakthrough_transfer_score"] == (0.4 + 0.2 + 0.1) / 3
    assert summary["breakthrough_clean_transfer_score"] == (0.5 + 0.3 + 0.2) / 3
    assert summary["breakthrough_invalid_as_loss_transfer_score"] == (0.3 + 0.1 + 0.0) / 3
