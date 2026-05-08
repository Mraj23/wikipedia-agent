"""Tests for solver-regret-based move evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.verifier.move_evaluator import (
    REGRET_CLIP_DEFAULT,
    clipped_regret,
    evaluate_move,
    solver_regret,
)


def _solver():
    return PonsSolver(fallback_depth=4)


def test_best_move_has_zero_regret():
    env = ConnectFourEnv()
    env.from_move_sequence([3, 4, 3, 4, 0, 4])  # opponent threatens col 4
    solver = _solver()
    me = evaluate_move(env, 4, solver)
    assert me.legal
    assert me.is_optimal
    assert me.raw_regret == 0.0
    assert me.clipped_regret == 0.0


def test_blunder_has_positive_regret():
    env = ConnectFourEnv()
    env.from_move_sequence([3, 4, 3, 4, 0, 4])
    solver = _solver()
    me = evaluate_move(env, 0, solver)
    assert me.legal
    assert me.raw_regret > 0
    assert me.clipped_regret > 0


def test_illegal_move_clipped_to_max():
    env = ConnectFourEnv()
    # Fill column 3 to make it illegal.
    env.from_move_sequence([3, 3, 3, 3, 3, 3])
    solver = _solver()
    me = evaluate_move(env, 3, solver)
    assert not me.legal
    assert me.clipped_regret == REGRET_CLIP_DEFAULT


def test_solver_regret_helper_matches_evaluation():
    env = ConnectFourEnv()
    env.from_move_sequence([3, 4, 3, 4, 0, 4])
    solver = _solver()
    raw = solver_regret(env, 4, solver)
    clipped = clipped_regret(env, 4, solver)
    assert raw == 0.0
    assert clipped == 0.0
