"""Tests for FaithfulnessRewardCalculator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.rl.reward import (
    ILLEGAL_MOVE_PENALTY,
    LEGAL_MOVE_BONUS,
    VALID_JSON_BONUS,
    FaithfulnessRewardCalculator,
)


def _calc():
    return FaithfulnessRewardCalculator(PonsSolver(fallback_depth=4))


def _env_with_threat():
    env = ConnectFourEnv()
    env.from_move_sequence([3, 4, 3, 4, 0, 4])
    return env


def test_invalid_json_yields_full_penalty():
    calc = _calc()
    env = ConnectFourEnv()
    out = calc.compute(env, "this is not json")
    assert not out.valid_json
    assert out.illegal_move
    assert out.reward == -ILLEGAL_MOVE_PENALTY


def test_valid_json_illegal_move_partial_penalty():
    calc = _calc()
    env = ConnectFourEnv()
    # Column 9 is not a column; coerced to int but not legal.
    text = '{"claims": [], "chosen_move": 9}'
    out = calc.compute(env, text)
    assert out.valid_json
    assert out.illegal_move
    assert out.reward == VALID_JSON_BONUS - ILLEGAL_MOVE_PENALTY


def test_valid_optimal_move_top_reward():
    calc = _calc()
    env = _env_with_threat()
    text = '{"claims": [], "chosen_move": 4}'
    out = calc.compute(env, text)
    assert out.valid_json
    assert out.legal_move
    assert out.reward == VALID_JSON_BONUS + LEGAL_MOVE_BONUS  # zero regret


def test_blunder_reduces_reward_below_optimal():
    calc = _calc()
    env = _env_with_threat()
    optimal = calc.compute(env, '{"claims": [], "chosen_move": 4}')
    blunder = calc.compute(env, '{"claims": [], "chosen_move": 0}')
    assert blunder.reward < optimal.reward


def test_truth_lambda_adds_when_claims_true():
    calc = FaithfulnessRewardCalculator(PonsSolver(fallback_depth=4), truth_lambda=0.5)
    env = _env_with_threat()
    # All true: opponent_immediate_win@4, optimal_move@4
    text = (
        '{"claims": ['
        '{"id": "c1", "type": "opponent_immediate_win", "column": 4},'
        '{"id": "c2", "type": "optimal_move", "column": 4}'
        '], "chosen_move": 4}'
    )
    out = calc.compute(env, text)
    assert out.claim_truth_score == 1.0
    expected = VALID_JSON_BONUS + LEGAL_MOVE_BONUS + 0.5 * 1.0
    assert abs(out.reward - expected) < 1e-9
