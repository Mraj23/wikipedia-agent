"""Tests for the position buffer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spiral.position_buffer import PositionBuffer, _categorize_depth
from env.connect_four_env import ConnectFourEnv


def test_sampled_positions_are_non_terminal():
    """All sampled positions should be non-terminal."""
    buf = PositionBuffer(pool_size=50, min_moves_remaining=2, seed=42)
    envs = buf.sample(batch_size=10)
    for env in envs:
        assert not env.is_terminal(), "Sampled position should not be terminal"
        assert len(env.legal_moves()) > 0


def test_buffer_has_requested_size():
    """Buffer should contain approximately pool_size positions."""
    buf = PositionBuffer(pool_size=100, min_moves_remaining=2, seed=42)
    assert len(buf) == 100


def test_sampled_envs_are_valid():
    """Sampled environments should have valid game state."""
    buf = PositionBuffer(pool_size=50, min_moves_remaining=2, seed=42)
    envs = buf.sample(batch_size=5)
    for env in envs:
        assert isinstance(env, ConnectFourEnv)
        assert env.current_player() in (1, 2)


def test_phase_categorization():
    """Depth categorization should match expected ranges."""
    assert _categorize_depth(0) == "beginning"
    assert _categorize_depth(8) == "beginning"
    assert _categorize_depth(9) == "middle"
    assert _categorize_depth(20) == "middle"
    assert _categorize_depth(21) == "end"


def test_min_moves_remaining_filter():
    """Positions should have at least min_moves_remaining moves left."""
    buf = PositionBuffer(pool_size=20, min_moves_remaining=6, seed=42)
    from training.minimax import MinimaxSolver
    quick = MinimaxSolver(depth=2)
    envs = buf.sample(batch_size=10)
    for env in envs:
        # Quick playout to estimate remaining moves
        sim = env.copy()
        count = 0
        while not sim.is_terminal() and count < 42:
            move = quick.best_move(sim)
            sim.make_move(move)
            count += 1
        assert count >= 6, f"Position has only {count} moves remaining"


def test_reproducibility():
    """Same seed should produce same positions."""
    buf1 = PositionBuffer(pool_size=20, min_moves_remaining=2, seed=123)
    buf2 = PositionBuffer(pool_size=20, min_moves_remaining=2, seed=123)
    assert buf1._pool == buf2._pool
