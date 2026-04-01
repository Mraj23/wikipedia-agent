"""In-domain Connect Four evaluation using Pons benchmark positions.

Benchmark positions are from blog.gamesolver.org and test whether the model
can identify optimal moves at various game phases.
"""

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.minimax import MinimaxSolver
from training.prompts import format_prompt, parse_response


def _load_benchmark_set(filepath: Path) -> List[Dict]:
    """Load a benchmark set from a CSV/text file.

    Expected format: move_sequence score (space-separated, one per line).

    Args:
        filepath: Path to the benchmark file.

    Returns:
        List of dicts with 'moves' and 'expected_score'.
    """
    positions = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                positions.append({
                    "moves": parts[0],
                    "expected_score": int(parts[1]),
                })
            elif len(parts) == 1:
                positions.append({
                    "moves": parts[0],
                    "expected_score": None,
                })
    return positions


def run_pons_benchmark(
    model_fn: Callable[[str], str],
    benchmark_dir: str = "data/pons_benchmark",
    solver: Optional[PonsSolver] = None,
) -> Dict:
    """Run the in-domain Connect Four benchmark evaluation.

    Loads position files from benchmark_dir, asks the model for each position,
    and compares the chosen move to the Pons-optimal move.

    Args:
        model_fn: Function taking a prompt string and returning model output.
        benchmark_dir: Directory containing benchmark position files.
        solver: PonsSolver instance (created if None).

    Returns:
        Dict with overall_pct_optimal and by_set breakdown, plus win_rate
        against minimax at various depths.
    """
    if solver is None:
        solver = PonsSolver()

    bench_path = Path(benchmark_dir)

    # Expected benchmark sets from blog.gamesolver.org
    set_names = [
        "Test_L3_R1",  # end_easy
        "Test_L2_R1",  # end_hard
        "Test_L2_R2",  # middle_easy
        "Test_L1_R1",  # middle_hard
        "Test_L1_R2",  # beginning_easy
        "Test_L1_R3",  # beginning_hard
    ]
    set_labels = [
        "end_easy", "end_hard",
        "middle_easy", "middle_hard",
        "beginning_easy", "beginning_hard",
    ]

    by_set: Dict[str, float] = {}
    all_correct = 0
    all_total = 0

    for set_name, label in zip(set_names, set_labels):
        # Try various file extensions
        filepath = None
        for ext in [".csv", ".txt", ""]:
            candidate = bench_path / f"{set_name}{ext}"
            if candidate.exists():
                filepath = candidate
                break

        if filepath is None:
            print(f"  Benchmark set {set_name} not found in {benchmark_dir}, skipping.")
            by_set[label] = -1.0  # sentinel for missing
            continue

        positions = _load_benchmark_set(filepath)
        correct = 0
        total = 0

        for pos in positions:
            move_seq = pos["moves"]
            env = ConnectFourEnv()
            try:
                moves = [int(c) for c in move_seq]
                env.from_move_sequence(moves)
            except (ValueError, Exception):
                continue

            if env.is_terminal():
                continue

            # Get optimal move from solver
            optimal = solver.best_move(env)

            # Get model's move
            prompt = format_prompt("A", env)
            response = model_fn(prompt)
            parsed = parse_response(response, "A")
            model_move = parsed.get("move")

            if model_move == optimal:
                correct += 1
            total += 1

        pct = correct / total if total > 0 else 0.0
        by_set[label] = pct
        all_correct += correct
        all_total += total

    overall_pct = all_correct / all_total if all_total > 0 else 0.0

    # Win rate against minimax at depths 2, 4, 6
    win_rates = _evaluate_vs_minimax(model_fn, solver, depths=[2, 4, 6], n_games=100)

    return {
        "overall_pct_optimal": overall_pct,
        "by_set": by_set,
        "total_positions": all_total,
        "win_rate_vs_minimax": win_rates,
    }


def _evaluate_vs_minimax(
    model_fn: Callable[[str], str],
    solver: PonsSolver,
    depths: List[int] = None,
    n_games: int = 100,
) -> Dict[int, float]:
    """Evaluate model win rate against minimax at various depths.

    Args:
        model_fn: Model callable.
        solver: PonsSolver for move parsing validation.
        depths: Minimax depths to test against.
        n_games: Games per depth level.

    Returns:
        Dict mapping depth -> win rate.
    """
    if depths is None:
        depths = [2, 4, 6]

    win_rates = {}
    for depth in depths:
        opponent = MinimaxSolver(depth=depth)
        wins = 0
        for game_idx in range(n_games):
            env = ConnectFourEnv()
            # Alternate who goes first
            model_player = 1 if game_idx % 2 == 0 else 2

            while not env.is_terminal():
                if env.current_player() == model_player:
                    prompt = format_prompt("A", env)
                    response = model_fn(prompt)
                    parsed = parse_response(response, "A")
                    move = parsed.get("move")
                    if move is None or move not in env.legal_moves():
                        move = env.legal_moves()[0]
                else:
                    move = opponent.best_move(env)
                env.make_move(move)

            winner = env.winner()
            if winner == model_player:
                wins += 1

        win_rates[depth] = wins / n_games
    return win_rates


if __name__ == "__main__":
    print("=== Pons Benchmark Demo ===\n")
    print("This module evaluates a model against Pons benchmark positions.")
    print("\nTo download benchmark positions:")
    print("  1. Visit https://blog.gamesolver.org/solving-connect-four/")
    print("  2. Download the test sets (Test_L3_R1 through Test_L1_R3)")
    print("  3. Place them in data/pons_benchmark/")
    print("\nUsage:")
    print("  from eval.pons_benchmark import run_pons_benchmark")
    print("  results = run_pons_benchmark(model_fn)")
