"""Tests for the minimax solver."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from training.minimax import MinimaxSolver


def test_returns_legal_move():
    """Move is in legal_moves()."""
    env = ConnectFourEnv()
    solver = MinimaxSolver(depth=4)
    move = solver.best_move(env)
    assert move in env.legal_moves()


def test_takes_winning_move():
    """If win available, minimax takes it."""
    env = ConnectFourEnv()
    # Set up: P1 has 3 in a row at bottom, col 3 open for the win
    # P1: cols 0,1,2  P2: cols 4,5,6 (on rows above to not block)
    env.make_move(0)  # P1 at (5,0)
    env.make_move(4)  # P2 at (5,4)
    env.make_move(1)  # P1 at (5,1)
    env.make_move(5)  # P2 at (5,5)
    env.make_move(2)  # P1 at (5,2)
    env.make_move(6)  # P2 at (5,6)
    # P1 has 3-in-a-row at bottom (0,1,2), col 3 is the winning move
    solver = MinimaxSolver(depth=2)
    move = solver.best_move(env)
    assert move == 3


def test_blocks_opponent_win():
    """If opponent has 3-in-a-row, minimax blocks."""
    env = ConnectFourEnv()
    # P1 plays random, then P2 gets 3-in-a-row
    env.make_move(0)  # P1
    env.make_move(1)  # P2 at (5,1)
    env.make_move(0)  # P1
    env.make_move(2)  # P2 at (5,2)
    env.make_move(6)  # P1
    env.make_move(3)  # P2 at (5,3)
    # P2 has 3-in-a-row at (5,1),(5,2),(5,3). P1 must block at col 4
    # Actually P2 needs one more for the threat — they have 3 consecutive
    # P1 should block at col 4 (right side) or col 0 is already taken
    # Wait — P2 has cols 1,2,3 = 3 in a row. P1 must play col 4 to block.
    solver = MinimaxSolver(depth=4)
    move = solver.best_move(env)
    assert move == 4  # block the 4-in-a-row


def test_analyze_returns_all_columns():
    """Keys match legal_moves()."""
    env = ConnectFourEnv()
    solver = MinimaxSolver(depth=2)
    scores = solver.analyze(env)
    assert set(scores.keys()) == set(env.legal_moves())


def test_depth_1_terminates_fast():
    """Assert time < 0.1s at depth 1."""
    env = ConnectFourEnv()
    solver = MinimaxSolver(depth=1)
    start = time.time()
    solver.analyze(env)
    elapsed = time.time() - start
    assert elapsed < 0.1, f"Depth 1 took {elapsed:.3f}s, expected < 0.1s"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
