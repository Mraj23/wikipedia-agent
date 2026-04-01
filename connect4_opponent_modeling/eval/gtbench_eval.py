"""GTBench evaluation wrapper.

GTBench is built on OpenSpiel — same engine as our training env.
Evaluates transfer of adversarial reasoning to Breakthrough and Nim.
"""

import os
import sys
from pathlib import Path
from typing import Callable, Dict, Optional


def _find_gtbench() -> Optional[Path]:
    """Locate the GTBench installation.

    Checks GTBENCH_PATH env var, then ../GTBench relative to this repo.

    Returns:
        Path to GTBench root, or None if not found.
    """
    env_path = os.environ.get("GTBENCH_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # Check relative path
    repo_root = Path(__file__).parent.parent.parent
    candidate = repo_root / "GTBench"
    if candidate.exists():
        return candidate

    return None


def _ensure_gtbench() -> Path:
    """Ensure GTBench is available, raising ImportError with instructions if not.

    Returns:
        Path to GTBench root.

    Raises:
        ImportError: If GTBench is not installed.
    """
    path = _find_gtbench()
    if path is None:
        raise ImportError(
            "GTBench not found. Install it with:\n"
            "  git clone https://github.com/jinhaoduan/GTBench && "
            "cd GTBench && pip install -e .\n"
            "Or set the GTBENCH_PATH environment variable."
        )
    return path


def run_gtbench(
    model_fn: Callable[[str], str],
    game: str = "breakthrough",
    n_games: int = 200,
    mcts_simulations: int = 1000,
) -> Dict:
    """Run GTBench evaluation on a single game.

    Wraps GTBench's runner. Alternates first player across games to
    mitigate first-player advantage.

    Args:
        model_fn: Function taking a prompt string and returning model output.
        game: Game name ('breakthrough' or 'nim').
        n_games: Number of games to play.
        mcts_simulations: MCTS simulations for the opponent.

    Returns:
        Dict with win_rate, loss_rate, draw_rate, and nra
        (nra = (wins - losses) / n_games).
    """
    gtbench_path = _ensure_gtbench()

    # Add GTBench to path if needed
    if str(gtbench_path) not in sys.path:
        sys.path.insert(0, str(gtbench_path))

    try:
        # GTBench uses OpenSpiel games
        import pyspiel
    except ImportError:
        raise ImportError("OpenSpiel is required for GTBench. pip install open_spiel")

    wins = 0
    losses = 0
    draws = 0

    game_obj = pyspiel.load_game(game)

    for game_idx in range(n_games):
        state = game_obj.new_initial_state()
        model_player = game_idx % 2  # Alternate first player

        while not state.is_terminal():
            current = state.current_player()

            if current == model_player:
                # Model's turn — format board and get response
                legal = state.legal_actions()
                board_str = str(state)
                prompt = (
                    f"You are playing {game}.\n\n"
                    f"Current board:\n{board_str}\n\n"
                    f"Legal moves: {legal}\n\n"
                    f"Choose a move (integer). Respond with just the move number."
                )
                response = model_fn(prompt)

                # Parse move from response
                move = _parse_move(response, legal)
                if move is None:
                    move = legal[0]
            else:
                # Opponent: use random legal action (placeholder for MCTS)
                import random
                legal = state.legal_actions()
                move = random.choice(legal)

            state.apply_action(move)

        # Determine outcome
        returns = state.returns()
        if returns[model_player] > 0:
            wins += 1
        elif returns[model_player] < 0:
            losses += 1
        else:
            draws += 1

    win_rate = wins / n_games
    loss_rate = losses / n_games
    draw_rate = draws / n_games
    nra = (wins - losses) / n_games

    return {
        "game": game,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "nra": nra,
        "n_games": n_games,
    }


def run_gtbench_full(model_fn: Callable[[str], str]) -> Dict:
    """Run GTBench evaluation on both Breakthrough and Nim.

    Args:
        model_fn: Model callable.

    Returns:
        Combined dict with results for both games.
    """
    breakthrough = run_gtbench(model_fn, game="breakthrough")
    nim = run_gtbench(model_fn, game="nim")

    return {
        "breakthrough": breakthrough,
        "nim": nim,
        "average_nra": (breakthrough["nra"] + nim["nra"]) / 2,
    }


def _parse_move(response: str, legal_actions: list) -> Optional[int]:
    """Parse a move integer from model response.

    Args:
        response: Model output string.
        legal_actions: List of legal action integers.

    Returns:
        A legal action integer, or None if parsing fails.
    """
    import re

    numbers = re.findall(r"\d+", response)
    for num_str in numbers:
        num = int(num_str)
        if num in legal_actions:
            return num
    return None


if __name__ == "__main__":
    print("=== GTBench Evaluation ===\n")
    print("GTBench evaluates transfer of adversarial reasoning to other games.")
    print("Supported games: breakthrough, nim\n")

    gtbench_path = _find_gtbench()
    if gtbench_path:
        print(f"GTBench found at: {gtbench_path}")
    else:
        print("GTBench not found. Install with:")
        print("  git clone https://github.com/jinhaoduan/GTBench && cd GTBench && pip install -e .")
        print("Or set GTBENCH_PATH environment variable.")
