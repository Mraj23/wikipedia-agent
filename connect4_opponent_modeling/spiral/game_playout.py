"""Game playout for condition B's sparse win/loss/draw rewards.

After the model picks a single move, this module plays the rest of the game
using minimax to determine the terminal outcome.
"""

from env.connect_four_env import ConnectFourEnv
from training.minimax import MinimaxSolver


def play_to_completion(
    env: ConnectFourEnv,
    model_move: int,
    solver: MinimaxSolver,
    model_player: int,
) -> str:
    """Play a game to completion after the model's move.

    Applies model_move, then alternates minimax moves for both sides
    until a terminal state is reached.

    Args:
        env: Game state BEFORE the model's move (not mutated).
        model_move: Column chosen by the model.
        solver: MinimaxSolver for playout moves.
        model_player: Which player the model is (1 or 2).

    Returns:
        One of 'win', 'loss', 'draw'.
    """
    sim = env.copy()
    sim.make_move(model_move)

    # Play out with minimax for both sides
    while not sim.is_terminal():
        move = solver.best_move(sim)
        sim.make_move(move)

    winner = sim.winner()
    if winner is None:
        return "draw"
    elif winner == model_player:
        return "win"
    else:
        return "loss"
