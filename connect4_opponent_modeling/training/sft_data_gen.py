"""Generate SFT warmup positions for Connect Four training.

Produces 50,000 positions by playing minimax vs minimax games at varying
depths. Each position is labeled with the optimal move from the solver.
"""

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.minimax import MinimaxSolver
from training.prompts import format_prompt


def _count_moves_to_end(env: ConnectFourEnv, solver: MinimaxSolver, max_depth: int = 42) -> int:
    """Estimate depth to end by playing out with minimax.

    Args:
        env: Current position.
        solver: Minimax solver for move selection.
        max_depth: Maximum moves to simulate.

    Returns:
        Number of moves until terminal state.
    """
    sim = env.copy()
    depth = 0
    quick_solver = MinimaxSolver(depth=2)
    while not sim.is_terminal() and depth < max_depth:
        move = quick_solver.best_move(sim)
        sim.make_move(move)
        depth += 1
    return depth


def generate_positions(
    n: int = 50000,
    output_path: str = "data/sft_warmup.jsonl",
    seed: int = 42,
    solver: Optional[PonsSolver] = None,
) -> None:
    """Generate n SFT warmup positions.

    Positions are created by playing random-depth minimax vs minimax games.
    Excludes terminal positions and positions where all moves are equivalent.

    Args:
        n: Number of positions to generate.
        output_path: Path to output JSONL file.
        seed: Random seed for reproducibility.
        solver: PonsSolver instance (created if None).
    """
    random.seed(seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if solver is None:
        solver = PonsSolver(fallback_depth=4)

    positions = []
    games_played = 0

    with tqdm(total=n, desc="Generating positions") as pbar:
        while len(positions) < n:
            # Create a game with random-depth minimax players
            depth1 = random.randint(2, 6)
            depth2 = random.randint(2, 6)
            solver1 = MinimaxSolver(depth=depth1)
            solver2 = MinimaxSolver(depth=depth2)

            env = ConnectFourEnv()
            game_positions = []

            while not env.is_terminal():
                # Record this position before the move
                if env.current_player() in (1, 2):
                    snapshot = env.copy()
                    legal = snapshot.legal_moves()

                    # Skip if all moves equivalent
                    scores = solver.analyze(snapshot)
                    values = list(scores.values())
                    if len(set(values)) > 1:
                        best_col = max(scores, key=scores.get)
                        game_positions.append((snapshot, best_col))

                # Play the move
                current_solver = solver1 if env.current_player() == 1 else solver2
                # Add some randomness: 20% chance of random move
                if random.random() < 0.2:
                    move = random.choice(env.legal_moves())
                else:
                    move = current_solver.best_move(env)
                env.make_move(move)

            games_played += 1

            # Add positions from this game
            for snapshot, best_col in game_positions:
                if len(positions) >= n:
                    break

                prompt = format_prompt("A", snapshot)
                completion = f"<think>Optimal play.</think><move>{best_col}</move>"
                depth_to_end = _count_moves_to_end(snapshot, MinimaxSolver(depth=2))

                positions.append({
                    "prompt": prompt,
                    "completion": completion,
                    "position_id": str(uuid.uuid4()),
                    "depth_to_end": depth_to_end,
                })
                pbar.update(1)

    # Write JSONL
    with open(output, "w") as f:
        for pos in positions:
            f.write(json.dumps(pos) + "\n")

    # Split info: first 45000 = train, last 5000 = val
    train_count = min(45000, int(len(positions) * 0.9))
    val_count = len(positions) - train_count

    meta = {
        "total": len(positions),
        "train_count": train_count,
        "val_count": val_count,
        "train_range": [0, train_count],
        "val_range": [train_count, len(positions)],
        "games_played": games_played,
        "seed": seed,
    }

    meta_path = output.parent / "sft_warmup_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nGenerated {len(positions)} positions from {games_played} games")
    print(f"Train: {train_count}, Val: {val_count}")
    print(f"Saved to {output} and {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT warmup positions")
    parser.add_argument("--n", type=int, default=50000, help="Number of positions")
    parser.add_argument("--output", type=str, default="data/sft_warmup.jsonl", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_positions(n=args.n, output_path=args.output, seed=args.seed)
