"""In-domain Connect Four evaluation using Pons benchmark positions.

Benchmark positions are from blog.gamesolver.org and test whether the model
can identify optimal moves at various game phases.
"""

import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.minimax import MinimaxSolver
from training.prompts import format_prompt, parse_response

logger = logging.getLogger(__name__)


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
    condition_label: str = "A",
    benchmark_dir: str = "data/pons_benchmark",
    solver: Optional[PonsSolver] = None,
    seed: int = 42,
    minimax_depths: Optional[Sequence[int]] = None,
    minimax_n_games: int = 100,
    max_positions_per_set: Optional[int] = None,
    progress_every: int = 25,
) -> Dict:
    """Run the in-domain Connect Four benchmark evaluation.

    Loads position files from benchmark_dir, asks the model for each position,
    and compares the chosen move to the Pons-optimal move.

    Args:
        model_fn: Function taking a prompt string and returning model output.
        condition_label: Prompt/parse condition used for model evaluation.
        benchmark_dir: Directory containing benchmark position files.
        solver: PonsSolver instance (created if None).
        seed: Evaluation RNG seed.

    Returns:
        Dict with overall_pct_optimal and by_set breakdown, plus win_rate
        against minimax at various depths.
    """
    if solver is None:
        solver = PonsSolver(strict=True)
    if not solver.is_available():
        raise RuntimeError(
            "Pons solver unavailable. Benchmark requires connect4_solver + 7x6.book; "
            "run scripts/bootstrap_gpu.sh before using eval.pons_benchmark."
        )

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
    invalid_outputs = 0

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
        if max_positions_per_set is not None:
            positions = positions[:max_positions_per_set]
        correct = 0
        total = 0

        logger.info("Pons set %s: evaluating %d positions", label, len(positions))
        for idx, pos in enumerate(positions, start=1):
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
            prompt = format_prompt(condition_label, env)
            response = model_fn(prompt)
            parsed = parse_response(response, condition_label)
            model_move = parsed.get("move")
            if model_move is None or model_move not in env.legal_moves():
                invalid_outputs += 1

            if model_move == optimal:
                correct += 1
            total += 1

            if progress_every > 0 and idx % progress_every == 0:
                logger.info(
                    "Pons set %s progress: %d/%d positions",
                    label,
                    idx,
                    len(positions),
                )

        pct = correct / total if total > 0 else 0.0
        by_set[label] = pct
        all_correct += correct
        all_total += total

    overall_pct = all_correct / all_total if all_total > 0 else 0.0

    # Win rate against minimax at depths 2, 4, 6.
    # Skipped entirely when caller passes empty depths or n_games<=0 — periodic
    # mid-training evals use this path; the full ladder is offline-only.
    depths_resolved = list(minimax_depths) if minimax_depths is not None else [2, 4, 6]
    if depths_resolved and minimax_n_games > 0:
        win_rates = _evaluate_vs_minimax(
            model_fn,
            solver,
            condition_label=condition_label,
            depths=depths_resolved,
            n_games=minimax_n_games,
            seed=seed,
            progress_every=progress_every,
        )
    else:
        win_rates = {}

    return {
        "overall_pct_optimal": overall_pct,
        "by_set": by_set,
        "total_positions": all_total,
        "invalid_outputs": invalid_outputs,
        "invalid_output_rate": invalid_outputs / all_total if all_total > 0 else 0.0,
        "seed": seed,
        "win_rate_vs_minimax": win_rates,
    }


def _evaluate_vs_minimax(
    model_fn: Callable[[str], str],
    solver: PonsSolver,
    condition_label: str = "A",
    depths: List[int] = None,
    n_games: int = 100,
    seed: int = 42,
    progress_every: int = 25,
) -> Dict[int, float]:
    """Evaluate model win rate against minimax at various depths.

    Args:
        model_fn: Model callable.
        solver: PonsSolver for move parsing validation.
        condition_label: Prompt/parse condition used for model evaluation.
        depths: Minimax depths to test against.
        n_games: Games per depth level.
        seed: Evaluation RNG seed.

    Returns:
        Dict mapping depth -> win rate.
    """
    if depths is None:
        depths = [2, 4, 6]

    rng = random.Random(seed)
    win_rates = {}
    for depth in depths:
        opponent = MinimaxSolver(depth=depth)
        wins = 0
        invalid_games = 0
        invalid_moves = 0
        logger.info(
            "Minimax depth %d: evaluating %d games",
            depth,
            n_games,
        )
        for game_idx in range(n_games):
            env = ConnectFourEnv()
            # Alternate who goes first
            model_player = 1 if game_idx % 2 == 0 else 2
            had_invalid = False

            while not env.is_terminal():
                if env.current_player() == model_player:
                    prompt = format_prompt(condition_label, env)
                    response = model_fn(prompt)
                    parsed = parse_response(response, condition_label)
                    move = parsed.get("move")
                    if move is None or move not in env.legal_moves():
                        invalid_moves += 1
                        had_invalid = True
                        move = rng.choice(env.legal_moves())
                else:
                    move = opponent.best_move(env)
                env.make_move(move)

            winner = env.winner()
            if winner == model_player:
                wins += 1
            if had_invalid:
                invalid_games += 1

            if progress_every > 0 and (game_idx + 1) % progress_every == 0:
                logger.info(
                    "Minimax depth %d progress: %d/%d games",
                    depth,
                    game_idx + 1,
                    n_games,
                )

        win_rates[depth] = {
            "win_rate": wins / n_games,
            "invalid_games": invalid_games,
            "invalid_game_rate": invalid_games / n_games,
            "invalid_moves": invalid_moves,
        }
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
