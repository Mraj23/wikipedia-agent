"""Wrapper around the Pons C++ Connect Four solver binary.

Provides automatic fallback to minimax when the binary is not available.
The Pons solver binary protocol: takes a move sequence string on stdin,
returns one line of space-separated integers — one score per column.
"""

import logging
import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional

from env.connect_four_env import ConnectFourEnv
from training.minimax import MinimaxSolver

logger = logging.getLogger(__name__)


class PonsSolver:
    """Wrapper around the Pons C++ solver with minimax fallback.

    The Pons solver provides perfect play analysis for Connect Four.
    When the binary is not available, falls back to alpha-beta minimax
    at configurable depth.
    """

    # Sentinel value for illegal columns in Pons output
    ILLEGAL_SENTINEL = -1000

    def __init__(self, solver_path: str = "./connect4_solver", fallback_depth: int = 8) -> None:
        """Initialize the solver.

        Args:
            solver_path: Path to the Pons solver binary.
            fallback_depth: Minimax depth to use when binary is absent.
        """
        # Resolve relative paths from the project root (parent of env/)
        path = Path(solver_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / path
        self._solver_path = path.resolve()
        self._fallback = MinimaxSolver(depth=fallback_depth)
        self._warned_fallback = False

    def is_available(self) -> bool:
        """Check if the Pons solver binary exists and is executable.

        Returns:
            True if the binary can be run.
        """
        return self._solver_path.is_file() and os.access(self._solver_path, os.X_OK)

    def _warn_fallback(self) -> None:
        """Log a single warning about falling back to minimax."""
        if not self._warned_fallback:
            logger.warning(
                "Pons solver binary not found at '%s'. "
                "Falling back to minimax (depth=%d). "
                "For perfect play, compile the Pons solver: "
                "https://github.com/PascalPons/connect4",
                self._solver_path,
                self._fallback.depth,
            )
            self._warned_fallback = True

    def analyze(self, env: ConnectFourEnv) -> Dict[int, int]:
        """Analyze the position and return scores per legal column.

        If the Pons binary is available, calls it with the move sequence.
        Otherwise falls back to minimax.

        Args:
            env: Current game environment.

        Returns:
            Dict mapping legal column -> integer score.
        """
        if self.is_available():
            return self._analyze_pons(env)
        else:
            self._warn_fallback()
            return self._analyze_minimax(env)

    def _analyze_pons(self, env: ConnectFourEnv) -> Dict[int, int]:
        """Call the Pons binary for analysis.

        The Pons solver returns one score per position. To get per-column scores,
        we batch all legal moves into a single solver call (one line per move).

        Args:
            env: Current game environment.

        Returns:
            Dict mapping legal column -> integer score.
        """
        move_seq = env.to_move_sequence()
        # Convert 0-indexed to 1-indexed for the Pons binary
        base = "".join(str(int(c) + 1) for c in move_seq)
        legal = env.legal_moves()

        # Build batch input: one line per legal column
        lines = []
        for col in legal:
            lines.append(base + str(col + 1))
        batch_input = "\n".join(lines) + "\n"

        try:
            result = subprocess.run(
                [str(self._solver_path)],
                input=batch_input,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning("Pons solver returned error: %s", result.stderr.strip())
                self._warn_fallback()
                return self._analyze_minimax(env)

            return self._parse_pons_batch(result.stdout, legal)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Pons solver failed: %s. Falling back to minimax.", e)
            return self._analyze_minimax(env)

    def _parse_pons_batch(self, output: str, legal: list) -> Dict[int, int]:
        """Parse batched Pons solver output.

        Each output line is: 'move_sequence score'

        Args:
            output: Raw stdout from the Pons binary.
            legal: List of legal columns corresponding to input lines.

        Returns:
            Dict mapping legal column -> integer score (negated, since Pons
            scores the position after the move from the opponent's perspective).
        """
        scores: Dict[int, int] = {}
        out_lines = [l.strip() for l in output.strip().split("\n") if l.strip()]

        for col, line in zip(legal, out_lines):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Pons returns the score from the perspective of the player
                    # who just moved (the opponent of current player). Negate so
                    # higher = better for the current player.
                    scores[col] = -int(parts[-1])
                except ValueError:
                    continue

        if not scores:
            return self._analyze_minimax(ConnectFourEnv())
        return scores

    def _analyze_minimax(self, env: ConnectFourEnv) -> Dict[int, int]:
        """Fall back to minimax analysis.

        Converts float scores to integer scale for compatibility.

        Args:
            env: Current game environment.

        Returns:
            Dict mapping legal column -> integer score (scaled to [-100, 100]).
        """
        float_scores = self._fallback.analyze(env)
        return {col: int(score * 100) for col, score in float_scores.items()}

    def analyze_batch(self, envs: list) -> list:
        """Analyze multiple positions in a single solver call.

        Args:
            envs: List of ConnectFourEnv objects.

        Returns:
            List of score dicts (same order as envs).
        """
        if not self.is_available() or not envs:
            return [self.analyze(env) for env in envs]

        # Build batch: for each env, one line per legal column
        all_lines = []
        env_col_map = []  # (env_idx, col) for each line
        for i, env in enumerate(envs):
            move_seq = env.to_move_sequence()
            base = "".join(str(int(c) + 1) for c in move_seq)
            for col in env.legal_moves():
                all_lines.append(base + str(col + 1))
                env_col_map.append((i, col))

        batch_input = "\n".join(all_lines) + "\n"

        try:
            result = subprocess.run(
                [str(self._solver_path)],
                input=batch_input,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return [self.analyze(env) for env in envs]

            out_lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]

            # Parse into per-env score dicts
            results = [{} for _ in envs]
            for (env_idx, col), line in zip(env_col_map, out_lines):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        results[env_idx][col] = -int(parts[-1])
                    except ValueError:
                        pass

            # Fallback for any env with no scores
            for i, scores in enumerate(results):
                if not scores:
                    results[i] = self._analyze_minimax(envs[i])

            return results
        except (subprocess.TimeoutExpired, OSError):
            return [self.analyze(env) for env in envs]

    def best_move(self, env: ConnectFourEnv) -> int:
        """Return the best column to play.

        Args:
            env: Current game environment.

        Returns:
            Best column index.
        """
        scores = self.analyze(env)
        if not scores:
            legal = env.legal_moves()
            return legal[0] if legal else 0
        return max(scores, key=scores.get)

    def normalize_reward(self, env: ConnectFourEnv, played_col: int) -> float:
        """Compute move quality as a normalized [0, 1] reward.

        Formula: (score(played) - min_score) / (max_score - min_score).
        Returns 1.0 if all moves are equivalent.

        Args:
            env: Game state BEFORE the move was played.
            played_col: The column that was played.

        Returns:
            Float in [0, 1].
        """
        scores = self.analyze(env)
        if played_col not in scores:
            return 0.0

        values = list(scores.values())
        min_s = min(values)
        max_s = max(values)

        if max_s == min_s:
            return 1.0

        return (scores[played_col] - min_s) / (max_s - min_s)

    def optimal_opponent_response(self, env: ConnectFourEnv, played_col: int) -> int:
        """Get the optimal opponent response after a given move.

        Applies played_col to a copy of env, then returns the Pons-optimal
        response for the opponent.

        Args:
            env: Game state BEFORE played_col is applied.
            played_col: The column the current player plays.

        Returns:
            Optimal column for the opponent.
        """
        next_env = env.copy()
        next_env.make_move(played_col)

        if next_env.is_terminal():
            return -1  # No response possible

        return self.best_move(next_env)


if __name__ == "__main__":
    print("=== Pons Solver Wrapper Demo ===\n")

    solver = PonsSolver()
    env = ConnectFourEnv()

    print(f"Solver binary available: {solver.is_available()}")
    print()

    # Make a few moves
    for col in [3, 3, 4, 2]:
        env.make_move(col)

    print(f"Position (Player {env.current_player()} to move):")
    print(env.to_text_grid())
    print()

    scores = solver.analyze(env)
    print("Column scores:")
    for col in sorted(scores):
        print(f"  Column {col}: {scores[col]}")

    best = solver.best_move(env)
    print(f"\nBest move: column {best}")

    # Test normalize_reward
    for col in env.legal_moves():
        reward = solver.normalize_reward(env, col)
        print(f"  Move quality col {col}: {reward:.3f}")

    # Test optimal opponent response
    opp = solver.optimal_opponent_response(env, best)
    print(f"\nOptimal opponent response to col {best}: column {opp}")
