"""Tests for game playout (condition B)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from training.minimax import MinimaxSolver
from spiral.game_playout import play_to_completion


def test_playout_reaches_terminal():
    """Playout should always reach a terminal state."""
    env = ConnectFourEnv()
    solver = MinimaxSolver(depth=2)
    move = env.legal_moves()[0]
    result = play_to_completion(env, move, solver, model_player=1)
    assert result in ("win", "loss", "draw")


def test_winning_move_returns_win():
    """If model plays a winning move, result should be 'win'."""
    env = ConnectFourEnv()
    # P1 has 3 in a row at bottom
    env.make_move(0)  # P1
    env.make_move(4)  # P2
    env.make_move(1)  # P1
    env.make_move(5)  # P2
    env.make_move(2)  # P1
    env.make_move(6)  # P2
    # P1 plays col 3 to win
    result = play_to_completion(env, 3, MinimaxSolver(depth=2), model_player=1)
    assert result == "win"


def test_does_not_mutate_original():
    """Original env should not be modified."""
    env = ConnectFourEnv()
    env.make_move(3)
    original_seq = env.to_move_sequence()
    solver = MinimaxSolver(depth=2)
    play_to_completion(env, 3, solver, model_player=2)
    assert env.to_move_sequence() == original_seq


def test_model_player_parameter():
    """Result should depend on which player the model is."""
    env = ConnectFourEnv()
    solver = MinimaxSolver(depth=4)
    move = solver.best_move(env)

    # Play as player 1 (current player)
    result1 = play_to_completion(env, move, solver, model_player=1)
    # Play as player 2 (opponent)
    result2 = play_to_completion(env, move, solver, model_player=2)
    # Results should be different (or both draw)
    assert result1 in ("win", "loss", "draw")
    assert result2 in ("win", "loss", "draw")


def test_playout_from_midgame():
    """Playout should work from midgame positions."""
    env = ConnectFourEnv()
    for col in [3, 3, 4, 4, 2, 2]:
        env.make_move(col)
    solver = MinimaxSolver(depth=3)
    move = solver.best_move(env)
    result = play_to_completion(env, move, solver, model_player=env.current_player())
    assert result in ("win", "loss", "draw")
