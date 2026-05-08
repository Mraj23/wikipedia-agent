"""Canonical board helpers for faithfulness tactical checks.

The faithfulness experiment needs a stable, physical board convention:
row 0 is the top of the grid and row 5 is the floor.  We derive the board
from the move history so claim verification, prompt rendering, and strategic
rules all agree.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from env.connect_four_env import ConnectFourEnv


def board_array(env: ConnectFourEnv) -> np.ndarray:
    """Return a 6x7 board with row 5 as the physical bottom."""
    return env._get_board_from_history().copy()  # type: ignore[attr-defined]


def landing_row(board: np.ndarray, col: int) -> Optional[int]:
    """Return the row where a piece would land in `col`, or None if full."""
    for row in range(board.shape[0] - 1, -1, -1):
        if board[row][col] == 0:
            return row
    return None


def has_four_through(board: np.ndarray, row: int, col: int, player: int) -> bool:
    """Return True if `player` has four-in-a-row through `(row, col)`."""
    rows, cols = board.shape
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        rr, cc = row + dr, col + dc
        while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == player:
            count += 1
            rr += dr
            cc += dc
        rr, cc = row - dr, col - dc
        while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == player:
            count += 1
            rr -= dr
            cc -= dc
        if count >= 4:
            return True
    return False


def winning_columns(board: np.ndarray, player: int) -> List[int]:
    """Columns where `player` can play immediately and make four-in-a-row."""
    out: List[int] = []
    for col in range(board.shape[1]):
        row = landing_row(board, col)
        if row is None:
            continue
        board[row][col] = player
        try:
            if has_four_through(board, row, col, player):
                out.append(col)
        finally:
            board[row][col] = 0
    return out


def drop_wins_for(board: np.ndarray, col: int, player: int) -> Optional[bool]:
    """Whether dropping `player` in `col` immediately wins."""
    row = landing_row(board, col)
    if row is None:
        return None
    board[row][col] = player
    try:
        return has_four_through(board, row, col, player)
    finally:
        board[row][col] = 0
