"""Mechanistic probe for opponent modeling.

This is the most important eval file. It measures whether the model has
internalized opponent modeling by testing:
1. Opponent response prediction accuracy
2. Board consequence prediction accuracy

Critical invariant: lock_probe_positions must never be called more than once.
The FileExistsError guard is NOT optional.
"""

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver

PROBE_PROMPT = """You are analyzing a Connect Four position.

{board}

The current player is about to play column {col}.
What column will their opponent most likely respond with?
Answer with a single number between 0 and 6."""

CONSEQUENCE_PROBE_PROMPT = """You are analyzing a Connect Four position.

{board}

The current player plays column {col} and their opponent responds optimally.
What does the resulting board look like? Show the board as a text grid using the same format."""


def _categorize_depth(move_count: int) -> str:
    """Categorize a position as beginning/middle/end based on move count.

    Args:
        move_count: Number of moves played so far.

    Returns:
        One of 'beginning', 'middle', 'end'.
    """
    if move_count <= 12:
        return "beginning"
    elif move_count <= 28:
        return "middle"
    else:
        return "end"


def _load_benchmark_positions(benchmark_dir: str) -> List[Dict]:
    """Load positions from Pons benchmark CSV files.

    Args:
        benchmark_dir: Directory containing benchmark CSV/position files.

    Returns:
        List of dicts with 'moves' (str) and 'depth_category' keys.
    """
    bench_path = Path(benchmark_dir)
    positions = []

    if not bench_path.exists():
        # Generate positions by self-play if no benchmark files
        return _generate_diverse_positions()

    for csv_file in sorted(bench_path.glob("*.csv")) + sorted(bench_path.glob("*.txt")):
        with open(csv_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Format: move_sequence score (or just move_sequence)
                parts = line.split()
                if parts:
                    move_seq = parts[0]
                    # Validate: only digits 0-6 (or 1-7 for 1-indexed)
                    if all(c.isdigit() for c in move_seq):
                        positions.append({
                            "moves": move_seq,
                            "depth_category": _categorize_depth(len(move_seq)),
                        })

    if not positions:
        return _generate_diverse_positions()
    return positions


def _generate_diverse_positions() -> List[Dict]:
    """Generate diverse positions via self-play for probing.

    Returns:
        List of position dicts covering beginning/middle/end phases.
    """
    from training.minimax import MinimaxSolver

    positions = []
    solver = MinimaxSolver(depth=4)

    for seed in range(500):
        rng = random.Random(seed)
        env = ConnectFourEnv()
        moves_played = []

        while not env.is_terminal():
            # Mix of random and minimax moves
            if rng.random() < 0.3:
                move = rng.choice(env.legal_moves())
            else:
                move = solver.best_move(env)
            env.make_move(move)
            moves_played.append(str(move))

            if not env.is_terminal() and len(env.legal_moves()) > 1:
                positions.append({
                    "moves": "".join(moves_played),
                    "depth_category": _categorize_depth(len(moves_played)),
                })

    return positions


def lock_probe_positions(
    benchmark_dir: str,
    output_path: str = "data/probe_positions_locked.jsonl",
    n: int = 300,
    seed: int = 42,
) -> None:
    """Lock probe positions for the experiment. MUST only be called once.

    Samples 100 Beginning + 100 Middle + 100 End positions and saves them.
    Raises FileExistsError if output_path already exists — never overwrites.

    Args:
        benchmark_dir: Directory with Pons benchmark files.
        output_path: Output JSONL path.
        n: Total positions (split equally across 3 categories).
        seed: Random seed.

    Raises:
        FileExistsError: If output_path already exists.
    """
    out = Path(output_path)
    if out.exists():
        raise FileExistsError(
            f"Probe positions already locked at {output_path}. "
            "This file must NEVER be regenerated. If you need to start over, "
            "manually delete it and document why."
        )

    out.parent.mkdir(parents=True, exist_ok=True)

    all_positions = _load_benchmark_positions(benchmark_dir)
    rng = random.Random(seed)

    per_category = n // 3
    by_cat = {"beginning": [], "middle": [], "end": []}
    for pos in all_positions:
        cat = pos["depth_category"]
        if cat in by_cat:
            by_cat[cat].append(pos)

    locked = []
    for cat in ["beginning", "middle", "end"]:
        pool = by_cat[cat]
        if len(pool) < per_category:
            print(f"WARNING: Only {len(pool)} {cat} positions available (need {per_category})")
            sample = pool
        else:
            sample = rng.sample(pool, per_category)
        locked.extend(sample)

    with open(out, "w") as f:
        for pos in locked:
            f.write(json.dumps(pos) + "\n")

    print(f"Locked {len(locked)} probe positions to {output_path}")
    print(f"  Beginning: {sum(1 for p in locked if p['depth_category'] == 'beginning')}")
    print(f"  Middle: {sum(1 for p in locked if p['depth_category'] == 'middle')}")
    print(f"  End: {sum(1 for p in locked if p['depth_category'] == 'end')}")


def run_probe(
    model_fn: Callable[[str], str],
    positions_path: str = "data/probe_positions_locked.jsonl",
    solver: Optional[PonsSolver] = None,
) -> Dict:
    """Run the opponent prediction probe.

    For each position, asks the model to predict the opponent's response
    after a specific move, then checks against the solver's optimal response.

    Args:
        model_fn: Function taking a prompt string and returning model output.
        positions_path: Path to locked probe positions JSONL.
        solver: PonsSolver instance (created if None).

    Returns:
        Dict with overall_accuracy and by_depth breakdown.
    """
    if solver is None:
        solver = PonsSolver()

    positions = []
    with open(positions_path) as f:
        for line in f:
            positions.append(json.loads(line))

    results = {"beginning": [], "middle": [], "end": []}
    total_correct = 0
    total = 0

    for pos in positions:
        move_seq = pos["moves"]
        cat = pos["depth_category"]

        env = ConnectFourEnv()
        try:
            # Convert move sequence to list of ints
            moves = [int(c) for c in move_seq]
            env.from_move_sequence(moves)
        except (ValueError, Exception):
            continue

        if env.is_terminal() or not env.legal_moves():
            continue

        # Pick a legal move to probe about
        test_col = solver.best_move(env)
        prompt = PROBE_PROMPT.format(board=env.to_text_grid(), col=test_col)

        response = model_fn(prompt)

        # Parse the response for a single digit
        predicted = _extract_column(response)
        if predicted is None:
            results[cat].append(0)
            total += 1
            continue

        # Get optimal opponent response
        try:
            optimal = solver.optimal_opponent_response(env, test_col)
        except (ValueError, Exception):
            # Skip positions where the solver can't determine opponent response
            continue
        correct = int(predicted == optimal)
        results[cat].append(correct)
        total_correct += correct
        total += 1

    overall_accuracy = total_correct / total if total > 0 else 0.0
    by_depth = {}
    for cat in ["beginning", "middle", "end"]:
        vals = results[cat]
        by_depth[cat] = sum(vals) / len(vals) if vals else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "by_depth": by_depth,
        "total": total,
    }


def run_consequence_probe(
    model_fn: Callable[[str], str],
    positions_path: str = "data/probe_positions_locked.jsonl",
    solver: Optional[PonsSolver] = None,
    n: int = 100,
) -> Dict:
    """Run the board-state prediction probe.

    Tests whether the model can predict what the board looks like after
    a move and the opponent's optimal response.

    Args:
        model_fn: Function taking a prompt string and returning model output.
        positions_path: Path to locked probe positions JSONL.
        solver: PonsSolver instance (created if None).
        n: Number of positions to test (first n).

    Returns:
        Dict with overall cell accuracy and example comparisons.
    """
    if solver is None:
        solver = PonsSolver()

    positions = []
    with open(positions_path) as f:
        for line in f:
            positions.append(json.loads(line))

    accuracies = []

    for pos in positions[:n]:
        move_seq = pos["moves"]

        env = ConnectFourEnv()
        try:
            moves = [int(c) for c in move_seq]
            env.from_move_sequence(moves)
        except (ValueError, Exception):
            continue

        if env.is_terminal() or not env.legal_moves():
            continue

        test_col = solver.best_move(env)
        prompt = CONSEQUENCE_PROBE_PROMPT.format(board=env.to_text_grid(), col=test_col)

        response = model_fn(prompt)

        # Compute actual board after move + optimal opponent response
        actual_env = env.copy()
        actual_env.make_move(test_col)
        if not actual_env.is_terminal():
            opp_response = solver.best_move(actual_env)
            actual_env.make_move(opp_response)

        actual_grid = actual_env.to_text_grid()
        actual_cells = _extract_board_cells(actual_grid)
        predicted_cells = _extract_board_cells(response)

        if actual_cells and predicted_cells:
            matches = sum(1 for a, p in zip(actual_cells, predicted_cells) if a == p)
            acc = matches / 42
        else:
            acc = 0.0
        accuracies.append(acc)

    return {
        "overall_cell_accuracy": np.mean(accuracies).item() if accuracies else 0.0,
        "n_tested": len(accuracies),
    }


def _extract_column(text: str) -> Optional[int]:
    """Extract a column number (0-6) from model output.

    Args:
        text: Model output string.

    Returns:
        Column number or None.
    """
    import re

    # Look for a single digit 0-6
    matches = re.findall(r"\b([0-6])\b", text)
    if matches:
        return int(matches[-1])
    return None


def _extract_board_cells(text: str) -> List[str]:
    """Extract board cells from a text grid.

    Args:
        text: Text grid string.

    Returns:
        List of cell values (should be 42 for a full board).
    """
    cells = []
    for line in text.strip().split("\n")[:6]:
        parts = line.strip().split()
        # Only take lines that look like board rows (contain . X or O)
        if all(p in (".", "X", "O") for p in parts) and len(parts) == 7:
            cells.extend(parts)
    return cells


if __name__ == "__main__":
    print("=== Probe Module Demo ===\n")
    print("PROBE_PROMPT template:")
    print(PROBE_PROMPT.format(board="[board here]", col=3))
    print()
    print("CONSEQUENCE_PROBE_PROMPT template:")
    print(CONSEQUENCE_PROBE_PROMPT.format(board="[board here]", col=3))
    print()

    # Demo: generate positions
    print("Generating sample positions for probing...")
    positions = _generate_diverse_positions()
    cats = {"beginning": 0, "middle": 0, "end": 0}
    for p in positions:
        cats[p["depth_category"]] += 1
    print(f"Generated {len(positions)} positions: {cats}")
