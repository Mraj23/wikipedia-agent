"""Thin wrapper around OpenSpiel's Connect Four game.

OpenSpiel is used because GTBench (primary transfer eval) is also built on
OpenSpiel, ensuring identical board representation between training and eval.
"""

import pyspiel
import numpy as np
from typing import List, Optional
from pathlib import Path


class ConnectFourEnv:
    """Thin wrapper around OpenSpiel's Connect Four game.

    Board convention: row 0 = top, row 5 = floor.
    OpenSpiel Connect Four uses this convention natively.
    """

    ROWS = 6
    COLS = 7

    def __init__(self) -> None:
        self._game = pyspiel.load_game("connect_four")
        self._state = self._game.new_initial_state()
        self._move_history: List[int] = []

    def reset(self) -> "ConnectFourEnv":
        """Reset the game to the initial state."""
        self._state = self._game.new_initial_state()
        self._move_history = []
        return self

    def legal_moves(self) -> List[int]:
        """Return list of legal columns (0-6)."""
        if self._state.is_terminal():
            return []
        return self._state.legal_actions()

    def make_move(self, col: int) -> None:
        """Drop a piece in the given column.

        Args:
            col: Column index 0-6.

        Raises:
            ValueError: If the move is illegal.
        """
        if col not in self.legal_moves():
            raise ValueError(f"Illegal move: column {col}. Legal moves: {self.legal_moves()}")
        self._state.apply_action(col)
        self._move_history.append(col)

    def is_terminal(self) -> bool:
        """Return True if the game is over."""
        return self._state.is_terminal()

    def current_player(self) -> int:
        """Return the current player: 1 or 2.

        Returns 0 if the game is terminal.
        """
        if self._state.is_terminal():
            return 0
        # OpenSpiel uses 0-indexed players; we return 1-indexed
        return self._state.current_player() + 1

    def winner(self) -> Optional[int]:
        """Return the winner: 1, 2, or None (draw/ongoing)."""
        if not self._state.is_terminal():
            return None
        returns = self._state.returns()
        if returns[0] > 0:
            return 1
        elif returns[1] > 0:
            return 2
        return None  # draw

    def returns(self) -> List[float]:
        """Return [score_p1, score_p2] from OpenSpiel."""
        return self._state.returns()

    def copy(self) -> "ConnectFourEnv":
        """Return an independent deep copy of this environment."""
        new_env = ConnectFourEnv()
        new_env._state = self._state.clone()
        new_env._move_history = list(self._move_history)
        return new_env

    def _get_board_array(self) -> np.ndarray:
        """Extract a 6x7 board array from the OpenSpiel observation tensor.

        Returns:
            np.ndarray of shape (6, 7) with values 0 (empty), 1 (player 1), 2 (player 2).
        """
        # OpenSpiel connect_four observation tensor layout:
        # 3 planes of 6x7: [current_player_pieces, opponent_pieces, empty]
        # The observation is from the perspective of the current player.
        obs_tensor = self._state.observation_tensor()
        obs = np.array(obs_tensor)
        # Tensor is 3 * 6 * 7 = 126 elements
        planes = obs.reshape(3, self.ROWS, self.COLS)
        # planes[0] = current player's pieces
        # planes[1] = opponent's pieces
        # planes[2] = empty cells

        board = np.zeros((self.ROWS, self.COLS), dtype=int)

        current_osp = self._state.current_player() if not self._state.is_terminal() else 0
        if self._state.is_terminal():
            # For terminal states, rebuild from history
            return self._get_board_from_history()

        # Current player's pieces = 1 if current_osp==0, else 2
        # Opponent's pieces = 2 if current_osp==0, else 1
        cp = current_osp + 1  # 1-indexed current player
        op = 2 if cp == 1 else 1

        for r in range(self.ROWS):
            for c in range(self.COLS):
                if planes[0][r][c] == 1.0:
                    board[r][c] = cp
                elif planes[1][r][c] == 1.0:
                    board[r][c] = op
        return board

    def _get_board_from_history(self) -> np.ndarray:
        """Reconstruct board from move history."""
        board = np.zeros((self.ROWS, self.COLS), dtype=int)
        col_heights = [self.ROWS - 1] * self.COLS  # bottom row index per column
        for i, col in enumerate(self._move_history):
            player = (i % 2) + 1  # player 1 moves first
            row = col_heights[col]
            board[row][col] = player
            col_heights[col] -= 1
        return board

    def to_text_grid(self) -> str:
        """Return a human-readable text grid of the board.

        Current player's pieces are shown as X, opponent's as O.
        Format:
            . . . . . . .
            . . . . . . .
            . . . X . . .
            . . O X . . .
            . O X O . . .
            O X O X O . .
            Columns: 0 1 2 3 4 5 6
            Your pieces: X  Opponent pieces: O
        """
        if self._state.is_terminal():
            board = self._get_board_from_history()
            # For terminal, show player 1 as X, player 2 as O
            cp = 1
        else:
            board = self._get_board_array()
            cp = self.current_player()

        op = 2 if cp == 1 else 1
        lines = []
        for r in range(self.ROWS):
            row_chars = []
            for c in range(self.COLS):
                if board[r][c] == cp:
                    row_chars.append("X")
                elif board[r][c] == op:
                    row_chars.append("O")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        lines.append("Columns: 0 1 2 3 4 5 6")
        lines.append("Your pieces: X  Opponent pieces: O")
        return "\n".join(lines)

    def from_move_sequence(self, moves: List[int]) -> "ConnectFourEnv":
        """Reset and replay the given move sequence.

        Args:
            moves: List of column indices to play in order.

        Returns:
            self for chaining.
        """
        self.reset()
        for col in moves:
            self.make_move(col)
        return self

    def to_move_sequence(self) -> str:
        """Return the move history as a string of column digits.

        This is the format expected by the Pons solver binary.
        E.g., '3344556'
        """
        return "".join(str(m) for m in self._move_history)

    @property
    def state(self) -> pyspiel.State:
        """Access the raw OpenSpiel state."""
        return self._state


if __name__ == "__main__":
    import random

    print("=== Connect Four Random Game Demo ===\n")
    env = ConnectFourEnv()

    while not env.is_terminal():
        print(f"Player {env.current_player()}'s turn")
        print(env.to_text_grid())
        print()

        legal = env.legal_moves()
        move = random.choice(legal)
        print(f"  -> Plays column {move}\n")
        env.make_move(move)

    print("=== GAME OVER ===")
    print(env.to_text_grid())
    winner = env.winner()
    if winner:
        print(f"\nPlayer {winner} wins!")
    else:
        print("\nDraw!")
    print(f"Move sequence: {env.to_move_sequence()}")
    print(f"Returns: {env.returns()}")
