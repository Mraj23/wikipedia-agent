"""GameBench stub — secondary adversarial evaluation.

GameBench tests adversarial reasoning across diverse board games.
This is a stub structured so the real implementation can be dropped in
without changing the interface.

Install GameBench:
    git clone https://github.com/Costarelli/GameBench
    cd GameBench
    pip install -e .

See: https://github.com/Costarelli/GameBench
"""

from typing import Callable, Dict, Tuple


def run_gamebench(
    model_fn: Callable[[str], str],
    games: Tuple[str, ...] = ("santorini", "hive", "sea_battle"),
    n_games: int = 50,
) -> Dict:
    """Run GameBench evaluation across multiple games.

    TODO: Integrate with the actual GameBench library.
    The interface is stable — swap in the real implementation when ready.

    Install instructions:
        git clone https://github.com/Costarelli/GameBench
        cd GameBench
        pip install -e .

    See: https://github.com/Costarelli/GameBench

    Args:
        model_fn: Function taking a prompt string and returning model output.
        games: Tuple of game names to evaluate.
        n_games: Number of games per game type.

    Returns:
        Dict with per-game results (placeholder values).
    """
    # TODO: Replace with actual GameBench integration
    # The real implementation should:
    # 1. Load each game from GameBench
    # 2. Play n_games per game type with model_fn as the player
    # 3. Return win_rate, loss_rate, draw_rate per game
    # 4. Aggregate into overall adversarial reasoning score

    results = {}
    for game in games:
        results[game] = {
            "win_rate": -1.0,       # placeholder
            "loss_rate": -1.0,      # placeholder
            "draw_rate": -1.0,      # placeholder
            "n_games": 0,           # placeholder
            "status": "stub — GameBench not yet integrated",
        }

    results["overall"] = {
        "average_win_rate": -1.0,   # placeholder
        "status": "stub — install GameBench from https://github.com/Costarelli/GameBench",
    }

    return results


if __name__ == "__main__":
    print("=== GameBench Evaluation Stub ===\n")
    print("This is a stub for the GameBench secondary adversarial eval.")
    print("Install GameBench:")
    print("  git clone https://github.com/Costarelli/GameBench")
    print("  cd GameBench")
    print("  pip install -e .")
    print("\nSee: https://github.com/Costarelli/GameBench")

    # Demo the stub interface
    def dummy_model(prompt: str) -> str:
        return "0"

    results = run_gamebench(dummy_model)
    print(f"\nStub results: {results}")
