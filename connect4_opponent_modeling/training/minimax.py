"""Alpha-beta minimax solver for Connect Four.

Used as a fallback when the Pons C++ solver binary is not compiled.
Uses OpenSpiel's legal_actions and apply_action internally rather than
reimplementing move generation.
"""

import time
from typing import Dict
from pathlib import Path

from env.connect_four_env import ConnectFourEnv


class MinimaxSolver:
    """Alpha-beta minimax with configurable depth and heuristic evaluation.

    Heuristic: center column preference, threat counting, win/loss detection.
    """

    # Column weights: center > adjacent > edge
    COLUMN_WEIGHTS = [1, 2, 3, 4, 3, 2, 1]

    def __init__(self, depth: int = 6) -> None:
        """Initialize the solver.

        Args:
            depth: Maximum search depth. Default 6 targets < 2s per position.
        """
        self.depth = depth

    def best_move(self, env: ConnectFourEnv) -> int:
        """Return the best column to play.

        Args:
            env: Current game environment.

        Returns:
            Best column index.
        """
        scores = self.analyze(env)
        return max(scores, key=scores.get)

    def analyze(self, env: ConnectFourEnv) -> Dict[int, float]:
        """Score each legal column, normalized to [-1, 1].

        Args:
            env: Current game environment.

        Returns:
            Dict mapping legal column -> score in [-1, 1].
        """
        legal = env.legal_moves()
        if not legal:
            return {}

        scores: Dict[int, float] = {}
        alpha = float("-inf")
        beta = float("inf")

        for col in legal:
            child = env.copy()
            child.make_move(col)
            if child.is_terminal():
                w = child.winner()
                if w is not None:
                    # Current player just won
                    scores[col] = 1.0
                else:
                    scores[col] = 0.0  # draw
            else:
                raw = -self._alphabeta(child, self.depth - 1, -beta, -alpha)
                scores[col] = max(-1.0, min(1.0, raw))
            alpha = max(alpha, scores[col])

        return scores

    def _alphabeta(self, env: ConnectFourEnv, depth: int, alpha: float, beta: float) -> float:
        """Negamax with alpha-beta pruning.

        Args:
            env: Current game state.
            depth: Remaining depth.
            alpha: Alpha bound.
            beta: Beta bound.

        Returns:
            Score from the perspective of the current player.
        """
        if env.is_terminal():
            w = env.winner()
            if w is None:
                return 0.0
            # The player who just moved won — that's the opponent of current_player
            return -1.0

        if depth <= 0:
            return self._evaluate(env)

        legal = env.legal_moves()
        # Order moves: center columns first for better pruning
        legal_sorted = sorted(legal, key=lambda c: -self.COLUMN_WEIGHTS[c])

        value = float("-inf")
        for col in legal_sorted:
            child = env.copy()
            child.make_move(col)
            child_val = -self._alphabeta(child, depth - 1, -beta, -alpha)
            value = max(value, child_val)
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return value

    def _evaluate(self, env: ConnectFourEnv) -> float:
        """Heuristic evaluation of a non-terminal position.

        Considers:
        - Center column preference
        - Threat counting (three-in-a-row with empty fourth)
        - Piece positioning

        Args:
            env: Current game state.

        Returns:
            Score in approximately [-1, 1] from current player's perspective.
        """
        board = env._get_board_array()
        current = env.current_player()
        opponent = 2 if current == 1 else 1

        score = 0.0

        # Center column preference
        for r in range(ConnectFourEnv.ROWS):
            for c in range(ConnectFourEnv.COLS):
                if board[r][c] == current:
                    score += self.COLUMN_WEIGHTS[c] * 0.01
                elif board[r][c] == opponent:
                    score -= self.COLUMN_WEIGHTS[c] * 0.01

        # Threat counting: check all windows of 4
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag-down-right, diag-down-left
        for r in range(ConnectFourEnv.ROWS):
            for c in range(ConnectFourEnv.COLS):
                for dr, dc in directions:
                    window = []
                    for i in range(4):
                        nr, nc = r + dr * i, c + dc * i
                        if 0 <= nr < ConnectFourEnv.ROWS and 0 <= nc < ConnectFourEnv.COLS:
                            window.append(board[nr][nc])
                        else:
                            break
                    if len(window) == 4:
                        score += self._score_window(window, current, opponent)

        # Normalize to roughly [-1, 1]
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _score_window(window: list, current: int, opponent: int) -> float:
        """Score a window of 4 cells for threats.

        Args:
            window: List of 4 cell values.
            current: Current player number.
            opponent: Opponent player number.

        Returns:
            Heuristic score contribution.
        """
        cur_count = window.count(current)
        opp_count = window.count(opponent)
        empty_count = window.count(0)

        if cur_count == 3 and empty_count == 1:
            return 0.15  # strong threat
        elif cur_count == 2 and empty_count == 2:
            return 0.03
        elif opp_count == 3 and empty_count == 1:
            return -0.15  # must block
        elif opp_count == 2 and empty_count == 2:
            return -0.03
        return 0.0


if __name__ == "__main__":
    import random

    print("=== Minimax Solver Demo ===\n")

    env = ConnectFourEnv()
    solver = MinimaxSolver(depth=4)

    # Play a few random moves first
    random.seed(42)
    for _ in range(6):
        if not env.is_terminal():
            env.make_move(random.choice(env.legal_moves()))

    print(f"Position after 6 random moves (Player {env.current_player()} to move):")
    print(env.to_text_grid())
    print()

    start = time.time()
    scores = solver.analyze(env)
    elapsed = time.time() - start

    print(f"Analysis (depth={solver.depth}, {elapsed:.2f}s):")
    for col, score in sorted(scores.items()):
        bar = "+" * int(max(0, score) * 20) or "-" * int(max(0, -score) * 20)
        print(f"  Column {col}: {score:+.3f}  {bar}")

    best = solver.best_move(env)
    print(f"\nBest move: column {best}")

    # Time test at depth 6
    solver6 = MinimaxSolver(depth=6)
    start = time.time()
    _ = solver6.analyze(env)
    elapsed = time.time() - start
    print(f"\nDepth 6 analysis time: {elapsed:.2f}s (target: <2s)")
