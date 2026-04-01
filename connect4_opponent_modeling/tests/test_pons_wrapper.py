"""Tests for the Pons solver wrapper."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver


def _get_solver():
    """Create a PonsSolver with reduced fallback depth for test speed."""
    return PonsSolver(fallback_depth=4)


def test_falls_back_when_binary_absent():
    """No error raised, returns minimax result."""
    solver = PonsSolver(solver_path="/nonexistent/binary")
    env = ConnectFourEnv()
    scores = solver.analyze(env)
    assert isinstance(scores, dict)
    assert len(scores) > 0


def test_is_available_false_without_binary():
    """is_available returns False when binary doesn't exist."""
    solver = PonsSolver(solver_path="/nonexistent/binary")
    assert not solver.is_available()


def test_normalize_reward_in_range():
    """Always [0,1]."""
    solver = _get_solver()
    env = ConnectFourEnv()
    for col in env.legal_moves():
        reward = solver.normalize_reward(env, col)
        assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range for col {col}"


def test_normalize_reward_optimal_is_one():
    """Best move gets 1.0."""
    solver = _get_solver()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    best = solver.best_move(env)
    reward = solver.normalize_reward(env, best)
    assert reward == 1.0


def test_analyze_returns_legal_cols_only():
    """All returned columns are legal moves."""
    solver = _get_solver()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    scores = solver.analyze(env)
    legal = set(env.legal_moves())
    for col in scores:
        assert col in legal, f"Column {col} not in legal moves {legal}"


def test_optimal_opponent_response_is_legal():
    """Optimal opponent response is a legal move."""
    solver = _get_solver()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    for col in env.legal_moves():
        opp = solver.optimal_opponent_response(env, col)
        if opp >= 0:  # -1 means terminal
            next_env = env.copy()
            next_env.make_move(col)
            if not next_env.is_terminal():
                assert opp in next_env.legal_moves()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
