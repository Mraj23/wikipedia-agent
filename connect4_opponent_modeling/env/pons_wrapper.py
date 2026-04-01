"""Wrapper around the Pons C++ Connect Four solver binary.

Provides automatic fallback to minimax when the binary is not available.
The Pons solver binary protocol: takes a move sequence string on stdin,
returns one line of space-separated integers — one score per column.
"""

import logging
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
        self._solver_path = Path(solver_path)
        self._fallback = MinimaxSolver(depth=fallback_depth)
        self._warned_fallback = False

    def is_available(self) -> bool:
        """Check if the Pons solver binary exists and is executable.

        Returns:
            True if the binary can be run.
        """
        return self._solver_path.is_file() and shutil.which(str(self._solver_path)) is not None

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

        Args:
            env: Current game environment.

        Returns:
            Dict mapping legal column -> integer score.
        """
        move_seq = env.to_move_sequence()
        try:
            # Pons solver expects 1-indexed columns, but the standard binary
            # from PascalPons/connect4 uses 1-indexed input.
            # Convert 0-indexed to 1-indexed for the binary.
            pons_input = "".join(str(int(c) + 1) for c in move_seq)
            result = subprocess.run(
                [str(self._solver_path)],
                input=pons_input + "\n",
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning("Pons solver returned error: %s", result.stderr.strip())
                self._warn_fallback()
                return self._analyze_minimax(env)

            return self._parse_pons_output(result.stdout, env)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Pons solver failed: %s. Falling back to minimax.", e)
            return self._analyze_minimax(env)

    def _parse_pons_output(self, output: str, env: ConnectFourEnv) -> Dict[int, int]:
        """Parse the Pons solver output.

        The binary returns one line with space-separated scores per column.

        Args:
            output: Raw stdout from the Pons binary.
            env: Current environment (for legal move validation).

        Returns:
            Dict mapping legal column -> integer score.
        """
        legal = set(env.legal_moves())
        scores: Dict[int, int] = {}
        lines = output.strip().split("\n")
        if not lines:
            return self._analyze_minimax(env)

        # Take the last line (some versions print headers)
        parts = lines[-1].strip().split()
        for col_idx, val_str in enumerate(parts):
            if col_idx >= ConnectFourEnv.COLS:
                break
            try:
                val = int(val_str)
            except ValueError:
                continue
            if col_idx in legal and val > self.ILLEGAL_SENTINEL:
                scores[col_idx] = val

        if not scores:
            return self._analyze_minimax(env)
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
