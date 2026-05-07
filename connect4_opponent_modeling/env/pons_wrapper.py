"""Wrapper around the Pons C++ Connect Four solver binary.

By default the wrapper can fall back to minimax for development convenience.
Active experiment paths should use ``strict=True`` so solver failures surface
immediately instead of silently turning into a much slower evaluation path.
The Pons solver binary protocol: takes a move sequence string on stdin,
returns one line of space-separated integers — one score per column.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

from env.connect_four_env import ConnectFourEnv
from training.minimax import MinimaxSolver

logger = logging.getLogger(__name__)


class PonsSolverError(RuntimeError):
    """Raised when strict Pons solver execution fails."""


class PonsSolver:
    """Wrapper around the Pons C++ solver with minimax fallback.

    The Pons solver provides perfect play analysis for Connect Four.
    When the binary is not available, falls back to alpha-beta minimax
    at configurable depth.
    """

    # Sentinel value for illegal columns in Pons output
    ILLEGAL_SENTINEL = -1000
    TERMINAL_WIN_SCORE = 1000
    TERMINAL_DRAW_SCORE = 0
    TERMINAL_LOSS_SCORE = -1000

    def __init__(
        self,
        solver_path: str = "./connect4_solver",
        book_path: str = "./7x6.book",
        fallback_depth: int = 8,
        strict: bool = False,
    ) -> None:
        """Initialize the solver.

        Args:
            solver_path: Path to the Pons solver binary.
            book_path: Path to the Pascal Pons 7x6 opening book.
            fallback_depth: Minimax depth to use when binary is absent.
            strict: If True, never fall back to minimax silently.
        """
        # Resolve relative paths from the project root (parent of env/)
        project_root = Path(__file__).resolve().parent.parent
        self._project_root = project_root

        path = Path(solver_path)
        if not path.is_absolute():
            path = project_root / path
        self._solver_path = path.resolve()

        opening_book = Path(book_path)
        if not opening_book.is_absolute():
            opening_book = project_root / opening_book
        self._book_path = opening_book.resolve()

        self._fallback = MinimaxSolver(depth=fallback_depth)
        self._warned_fallback = False
        self._strict = strict

    def is_available(self) -> bool:
        """Check if the Pons solver binary exists and is executable.

        Returns:
            True if the binary can be run.
        """
        return (
            self._solver_path.is_file()
            and os.access(self._solver_path, os.X_OK)
            and self._book_path.is_file()
        )

    def _warn_fallback(self) -> None:
        """Log a single warning about falling back to minimax."""
        if not self._warned_fallback:
            missing = []
            if not self._solver_path.is_file():
                missing.append(f"binary '{self._solver_path}'")
            elif not os.access(self._solver_path, os.X_OK):
                missing.append(f"executable bit on '{self._solver_path}'")
            if not self._book_path.is_file():
                missing.append(f"opening book '{self._book_path}'")
            missing_desc = ", ".join(missing) if missing else "required solver assets"
            logger.warning(
                "Pons solver unavailable (%s). "
                "Falling back to minimax (depth=%d). "
                "Run scripts/bootstrap_gpu.sh to install solver assets.",
                missing_desc,
                self._fallback.depth,
            )
            self._warned_fallback = True

    def _strict_error(self, message: str) -> None:
        """Raise a strict-mode solver error."""
        raise PonsSolverError(message)

    def _terminal_score_after_move(self, env_after_move: ConnectFourEnv, player: int) -> int:
        """Score a terminal child without asking Pons to analyze a finished game."""
        outcome = env_after_move.returns()[player - 1]
        if outcome > 0:
            return self.TERMINAL_WIN_SCORE
        if outcome < 0:
            return self.TERMINAL_LOSS_SCORE
        return self.TERMINAL_DRAW_SCORE

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
        if self._strict:
            self._strict_error(
                "Pons solver unavailable. Expected connect4_solver + 7x6.book. "
                "Run scripts/bootstrap_gpu.sh before training or evaluation."
            )
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
        current_player = env.current_player()

        # Build batch input for non-terminal child positions. Pons does not emit
        # ordinary scores for finished games, so terminal children are scored here.
        scores: Dict[int, int] = {}
        lines = []
        pending_cols = []
        for col in legal:
            next_env = env.copy()
            next_env.make_move(col)
            if next_env.is_terminal():
                scores[col] = self._terminal_score_after_move(next_env, current_player)
            else:
                lines.append(base + str(col + 1))
                pending_cols.append(col)

        if not lines:
            return scores

        batch_input = "\n".join(lines) + "\n"

        try:
            result = subprocess.run(
                [str(self._solver_path)],
                input=batch_input,
                capture_output=True,
                text=True,
                cwd=str(self._project_root),
                timeout=30,
            )
            if result.returncode != 0:
                if self._strict:
                    self._strict_error(
                        f"Pons solver returned error code {result.returncode}: "
                        f"{result.stderr.strip() or '<no stderr>'}"
                    )
                logger.warning("Pons solver returned error: %s", result.stderr.strip())
                self._warn_fallback()
                return self._analyze_minimax(env)

            parsed = self._parse_pons_batch(result.stdout, pending_cols)
            scores.update(parsed)
            if set(scores) != set(legal):
                missing = sorted(set(legal) - set(scores))
                if self._strict:
                    self._strict_error(
                        f"Pons solver returned incomplete scores for a legal position. "
                        f"Missing columns: {missing}"
                    )
                logger.warning(
                    "Pons solver returned incomplete scores. Falling back to minimax. "
                    "Missing columns: %s",
                    missing,
                )
                return self._analyze_minimax(env)
            return scores
        except (subprocess.TimeoutExpired, OSError) as e:
            if self._strict:
                self._strict_error(f"Pons solver failed during execution: {e}")
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
        if not envs:
            return []
        if not self.is_available():
            if self._strict:
                self._strict_error(
                    "Pons solver unavailable. Expected connect4_solver + 7x6.book. "
                    "Run scripts/bootstrap_gpu.sh before batched evaluation."
                )
            return [self.analyze(env) for env in envs]

        # Build batch: for each env, one line per non-terminal legal child.
        all_lines = []
        env_col_map = []  # (env_idx, col) for each line
        results = [{} for _ in envs]
        for i, env in enumerate(envs):
            move_seq = env.to_move_sequence()
            base = "".join(str(int(c) + 1) for c in move_seq)
            current_player = env.current_player()
            for col in env.legal_moves():
                next_env = env.copy()
                next_env.make_move(col)
                if next_env.is_terminal():
                    results[i][col] = self._terminal_score_after_move(next_env, current_player)
                else:
                    all_lines.append(base + str(col + 1))
                    env_col_map.append((i, col))

        if not all_lines:
            return results

        batch_input = "\n".join(all_lines) + "\n"

        try:
            result = subprocess.run(
                [str(self._solver_path)],
                input=batch_input,
                capture_output=True,
                text=True,
                cwd=str(self._project_root),
                timeout=60,
            )
            if result.returncode != 0:
                if self._strict:
                    self._strict_error(
                        f"Pons batch solver returned error code {result.returncode}: "
                        f"{result.stderr.strip() or '<no stderr>'}"
                    )
                return [self.analyze(env) for env in envs]

            out_lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]

            if len(out_lines) != len(env_col_map):
                if self._strict:
                    self._strict_error(
                        f"Pons batch solver returned {len(out_lines)} rows for "
                        f"{len(env_col_map)} requested non-terminal child positions."
                    )
                return [self._analyze_minimax(env) for env in envs]

            # Parse into per-env score dicts.
            for (env_idx, col), line in zip(env_col_map, out_lines):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        results[env_idx][col] = -int(parts[-1])
                    except ValueError:
                        pass

            # Fallback for any env with incomplete scores.
            for i, scores in enumerate(results):
                missing = set(envs[i].legal_moves()) - set(scores)
                if missing:
                    if self._strict:
                        self._strict_error(
                            "Pons batch solver returned incomplete scores for at least one "
                            f"position. Missing columns: {sorted(missing)}"
                        )
                    results[i] = self._analyze_minimax(envs[i])

            return results
        except (subprocess.TimeoutExpired, OSError):
            if self._strict:
                self._strict_error("Pons batch solver failed during execution.")
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
