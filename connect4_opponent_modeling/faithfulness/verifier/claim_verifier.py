"""Ground-truth oracle for atomic claims.

verify_claim(claim, env, solver) -> True | False | None
    True  — claim is tactically true on the given board.
    False — claim is well-formed but incorrect.
    None  — claim is malformed (missing fields, out-of-range column, etc.)
            and therefore unverifiable. Metrics treat None as "skip" rather
            than False so a malformed claim doesn't fake a faithfulness win
            in either direction.
"""

from typing import List, Optional

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.claims import Claim, ClaimType


def _opponent_can_win_now(env: ConnectFourEnv, column: Optional[int] = None) -> Optional[bool]:
    """Does the opponent have an immediate winning move right now?

    If `column` is given, asserts that *that specific column* is a winning
    move for the opponent. Otherwise checks any legal column.

    Implementation: build a mirror env where it's the opponent's turn by
    replaying the move history with one extra "pass-equivalent" piece. Since
    Connect Four has no pass move, we instead use a clone with the move
    history rewound and re-applied so the opponent is to move. The simplest
    correct implementation: use OpenSpiel's terminal-aware dynamics by
    constructing a fresh state where the opponent acts.

    Concretely we copy the env, then test "what if opponent moves now?" by
    reading the board and simulating manually via env.copy + make_move
    semantics. We use a cheap trick: clone the env and mutate via a
    private helper — but since make_move advances the current player, we
    need a way to play the opponent without playing for the current player
    first. We do this by checking each candidate column with a board-aware
    simulation: drop the opponent piece into the column on a copied grid
    and check the resulting position for a four-in-a-row. This logic lives
    here rather than in env to avoid touching the canonical game wrapper.
    """
    if env.is_terminal():
        return False
    legal = env.legal_moves()
    if not legal:
        return False
    cur = env.current_player()
    opp = 2 if cur == 1 else 1

    if column is not None and column not in legal:
        # Opponent cannot play a full column either.
        return False

    candidates = [column] if column is not None else legal

    # Reconstruct the full 6x7 board to drop a hypothetical opponent piece.
    # Note: the OpenSpiel observation tensor used here treats row 0 as the
    # bottom of each column — pieces stack upward toward row 5. The
    # _get_board_from_history fallback uses the opposite convention, so we
    # only handle the non-terminal case here (verifier is not called on
    # terminal boards).
    if env.is_terminal():
        return False
    board = env._get_board_array()  # type: ignore[attr-defined]

    for col in candidates:
        # Find the lowest empty row in the column (row 0 is bottom).
        landed_row = None
        for r in range(env.ROWS):
            if board[r][col] == 0:
                landed_row = r
                break
        if landed_row is None:
            continue
        if _has_four_after_drop(board, landed_row, col, opp):
            return True
    return False


def _has_four_after_drop(board, row: int, col: int, player: int) -> bool:
    """Return True if dropping `player` at (row, col) creates a four-in-a-row.

    Assumes the cell is currently empty; we simulate the drop in-place.
    """
    rows, cols = board.shape
    board[row][col] = player
    try:
        # All 4 directions from this cell
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            # forward
            rr, cc = row + dr, col + dc
            while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == player:
                count += 1
                rr += dr
                cc += dc
            # backward
            rr, cc = row - dr, col - dc
            while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == player:
                count += 1
                rr -= dr
                cc -= dc
            if count >= 4:
                return True
        return False
    finally:
        board[row][col] = 0


def _self_wins_with(env: ConnectFourEnv, column: int) -> Optional[bool]:
    if column not in env.legal_moves():
        return None
    nxt = env.copy()
    cur = env.current_player()
    nxt.make_move(column)
    return nxt.winner() == cur


def _move_allows_opponent_win(env: ConnectFourEnv, move: int, opp_reply: int) -> Optional[bool]:
    if move not in env.legal_moves():
        return None
    after_move = env.copy()
    after_move.make_move(move)
    if after_move.is_terminal():
        # If our move ended the game (draw or our win), the claim is False —
        # the opponent does not get a winning reply.
        return False
    if opp_reply not in after_move.legal_moves():
        return None
    after_reply = after_move.copy()
    after_reply.make_move(opp_reply)
    opp = 2 if env.current_player() == 1 else 1
    return after_reply.winner() == opp


def verify_claim(
    claim: Claim,
    env: ConnectFourEnv,
    solver: PonsSolver,
) -> Optional[bool]:
    if not claim.has_required_fields():
        return None

    f = claim.fields
    legal = env.legal_moves()

    if claim.type is ClaimType.SELF_IMMEDIATE_WIN:
        col = f.get("column")
        if not isinstance(col, int) or col < 0 or col > 6:
            return None
        return _self_wins_with(env, col)

    if claim.type is ClaimType.OPPONENT_IMMEDIATE_WIN:
        col = f.get("column")
        if not isinstance(col, int) or col < 0 or col > 6:
            return None
        return _opponent_can_win_now(env, column=col)

    if claim.type is ClaimType.MOVE_ALLOWS_OPPONENT_WIN:
        move = f.get("move")
        opp_reply = f.get("opponent_reply")
        if not isinstance(move, int) or not isinstance(opp_reply, int):
            return None
        if move < 0 or move > 6 or opp_reply < 0 or opp_reply > 6:
            return None
        return _move_allows_opponent_win(env, move, opp_reply)

    if claim.type is ClaimType.LEGAL_MOVE:
        col = f.get("column")
        if not isinstance(col, int) or col < 0 or col > 6:
            return None
        return col in legal

    if claim.type is ClaimType.OPTIMAL_MOVE:
        col = f.get("column")
        if not isinstance(col, int) or col < 0 or col > 6:
            return None
        if col not in legal:
            return False
        scores = solver.analyze(env)
        if not scores:
            return None
        best_value = max(scores.values())
        return scores.get(col, None) == best_value

    return None


def verify_claims(
    claims: List[Claim],
    env: ConnectFourEnv,
    solver: PonsSolver,
) -> List[Optional[bool]]:
    """Verify a list of claims with at most one Pons call shared across them."""
    if not claims:
        return []

    needs_pons = any(c.type is ClaimType.OPTIMAL_MOVE for c in claims)
    cached_scores = solver.analyze(env) if needs_pons else None

    labels: List[Optional[bool]] = []
    for c in claims:
        if c.type is ClaimType.OPTIMAL_MOVE and cached_scores is not None:
            f = c.fields
            col = f.get("column")
            if not isinstance(col, int) or col < 0 or col > 6:
                labels.append(None)
                continue
            if col not in env.legal_moves():
                labels.append(False)
                continue
            best_value = max(cached_scores.values()) if cached_scores else None
            labels.append(cached_scores.get(col) == best_value if best_value is not None else None)
        else:
            labels.append(verify_claim(c, env, solver))
    return labels
