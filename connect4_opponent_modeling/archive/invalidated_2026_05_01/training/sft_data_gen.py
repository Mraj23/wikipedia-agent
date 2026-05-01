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


def _estimate_depth_to_end(env: ConnectFourEnv) -> int:
    """Estimate depth to end from move count (fast, no playout needed).

    Args:
        env: Current position.

    Returns:
        Estimated number of moves until terminal state.
    """
    moves_played = len(env._move_history)
    # Average Connect Four game is ~36 moves total
    return max(1, 36 - moves_played)


def _generate_think_text(
    env: ConnectFourEnv,
    best_col: int,
    scores: dict,
    rng: random.Random,
) -> str:
    """Generate varied reasoning text for SFT completions.

    Uses the position context to produce diverse think-tag content so the
    model learns the XML format pattern rather than memorising a single
    fixed string like "Optimal play."
    """
    moves_played = len(env._move_history)
    legal = env.legal_moves()
    best_score = scores.get(best_col, 0)

    # Describe why the chosen column is good
    reasons = []

    if best_col == 3:
        reasons.append("Column 3 controls the center.")
    elif best_col in (2, 4):
        reasons.append(f"Column {best_col} is near the center.")
    else:
        reasons.append(f"Column {best_col} is the strongest move here.")

    if best_score > 0:
        reasons.append("This leads to a winning position.")
    elif best_score == 0:
        reasons.append("This maintains a drawn position.")
    else:
        reasons.append("This is the best defensive option.")

    if moves_played < 8:
        reasons.append("Early game — focus on center control.")
    elif moves_played < 20:
        reasons.append("Midgame — building threats.")
    else:
        reasons.append("Late game — precise play required.")

    # Pick 1-2 reasons randomly for variety
    n_reasons = rng.randint(1, min(2, len(reasons)))
    selected = rng.sample(reasons, n_reasons)
    return " ".join(selected)


def generate_positions(
    n: int = 50000,
    output_path: str = "data/sft_warmup.jsonl",
    seed: int = 42,
    solver: Optional[PonsSolver] = None,
    condition: str = "A",
) -> None:
    """Generate n SFT warmup positions.

    Positions are created by playing random-depth minimax vs minimax games.
    Excludes terminal positions and positions where all moves are equivalent.

    Args:
        n: Number of positions to generate.
        output_path: Path to output JSONL file.
        seed: Random seed for reproducibility.
        solver: PonsSolver instance (created if None).
        condition: Prompt condition to use ('A' for standard, 'F' for
            opponent-modeling prompts used by the F-tuned baseline).
    """
    random.seed(seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if solver is None:
        solver = PonsSolver()

    positions = []
    games_played = 0

    with tqdm(total=n, desc="Generating positions") as pbar:
        while len(positions) < n:
            # Create a game with random-depth minimax players
            depth1 = random.randint(2, 4)
            depth2 = random.randint(2, 4)
            solver1 = MinimaxSolver(depth=depth1)
            solver2 = MinimaxSolver(depth=depth2)

            env = ConnectFourEnv()
            snapshots = []

            while not env.is_terminal():
                # Record this position before the move
                if env.current_player() in (1, 2):
                    snapshots.append(env.copy())

                # Play the move
                current_solver = solver1 if env.current_player() == 1 else solver2
                # Add some randomness: 20% chance of random move
                if random.random() < 0.2:
                    move = random.choice(env.legal_moves())
                else:
                    move = current_solver.best_move(env)
                env.make_move(move)

            games_played += 1

            # Batch-analyze all snapshots in one solver call
            game_positions = []
            all_scores = solver.analyze_batch(snapshots)
            for snapshot, scores in zip(snapshots, all_scores):
                values = list(scores.values())
                if len(set(values)) > 1:
                    best_col = max(scores, key=scores.get)
                    game_positions.append((snapshot, best_col))

            # Add positions from this game
            for snapshot, best_col in game_positions:
                if len(positions) >= n:
                    break

                prompt = format_prompt(condition, snapshot)

                # Generate varied reasoning text so the model learns the
                # format pattern, not just a single fixed string.
                think_text = _generate_think_text(
                    snapshot, best_col, scores, random
                )

                if condition in ("E", "F"):
                    # Include opponent prediction for F-tuned baseline
                    opp_response = solver.optimal_opponent_response(snapshot, best_col)
                    opp_col = opp_response if opp_response >= 0 else best_col
                    completion = (
                        f"<think>{think_text}</think>"
                        f"<opponent_prediction>{opp_col}</opponent_prediction>"
                        f"<move>{best_col}</move>"
                    )
                elif condition == "D":
                    # Include opponent prediction and future state
                    opp_response = solver.optimal_opponent_response(snapshot, best_col)
                    opp_col = opp_response if opp_response >= 0 else best_col
                    next_env = snapshot.copy()
                    next_env.make_move(best_col)
                    future_grid = next_env.to_text_grid().split("\n")
                    future_board = "\n".join(future_grid[:6])
                    completion = (
                        f"<think>{think_text}</think>"
                        f"<opponent_prediction>{opp_col}</opponent_prediction>"
                        f"<future_state>\n{future_board}\n</future_state>"
                        f"<move>{best_col}</move>"
                    )
                else:
                    completion = f"<think>{think_text}</think><move>{best_col}</move>"
                depth_to_end = _estimate_depth_to_end(snapshot)

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
    parser.add_argument(
        "--condition", type=str, default="A",
        choices=["A", "D", "E", "F"],
        help="Prompt condition: A (standard), D/E/F (with opponent prediction tags)",
    )
    args = parser.parse_args()

    generate_positions(n=args.n, output_path=args.output, seed=args.seed, condition=args.condition)
