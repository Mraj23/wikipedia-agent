"""Deterministic strategic-move rules for Connect Four.

A pure, board-rule-only classifier for the strategic role of a candidate move.
No solver, no minimax, no ML. The output is stable across environments and
fast enough to run on every legal move of every candidate position.

Two layers:
    1. classify_move(env, col) -> StrategicTag
       The single most important strategic role of a move.
    2. classify_position(env) -> PositionTag
       The cheapest deterministic stratum of the position itself, derived
       by aggregating per-move tags. Used by board_generator.py as a pre-
       filter so we only call the expensive Pons/minimax solver on positions
       whose category was not already settled by deterministic rules.

These categorizations also double as ground truth for several of the more
nuanced claim types we may add in future (e.g., "creates_double_threat",
"blocks_opponent_threat") without needing the Pons oracle.

Tag priorities (highest first; the first match for a move wins):
    1. immediate_win                  — move makes 4-in-a-row for mover
    2. block_immediate_threat         — move occupies the cell that would have
                                        let the opponent make 4-in-a-row
                                        on the very next turn
    3. allows_opponent_immediate_win  — after this move, opponent has at least
                                        one immediate-win column
    4. creates_double_threat          — after this move, mover has >=2 distinct
                                        immediate-win columns next turn
    5. creates_threat                 — after this move, mover has exactly one
                                        immediate-win column next turn
    6. blocks_opponent_threat         — opponent had a "creates_threat"-style
                                        winning line through the empty cell
                                        that we just filled (i.e., the move
                                        denies the opponent a future double
                                        without blocking an immediate win)
    7. center_play                    — non-tactical play in the center column
    8. neutral                        — none of the above
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from env.connect_four_env import ConnectFourEnv
from faithfulness.board_utils import (
    board_array,
    has_four_through,
    landing_row,
    winning_columns,
)

CENTER_COL = 3


class StrategicTag(str, Enum):
    IMMEDIATE_WIN = "immediate_win"
    BLOCK_IMMEDIATE_THREAT = "block_immediate_threat"
    ALLOWS_OPPONENT_IMMEDIATE_WIN = "allows_opponent_immediate_win"
    CREATES_DOUBLE_THREAT = "creates_double_threat"
    CREATES_THREAT = "creates_threat"
    BLOCKS_OPPONENT_THREAT = "blocks_opponent_threat"
    CENTER_PLAY = "center_play"
    NEUTRAL = "neutral"


# Priority order matching the docstring above. Earlier = higher priority.
_TAG_PRIORITY: Dict[StrategicTag, int] = {
    tag: i
    for i, tag in enumerate(
        [
            StrategicTag.IMMEDIATE_WIN,
            StrategicTag.BLOCK_IMMEDIATE_THREAT,
            StrategicTag.ALLOWS_OPPONENT_IMMEDIATE_WIN,
            StrategicTag.CREATES_DOUBLE_THREAT,
            StrategicTag.CREATES_THREAT,
            StrategicTag.BLOCKS_OPPONENT_THREAT,
            StrategicTag.CENTER_PLAY,
            StrategicTag.NEUTRAL,
        ]
    )
}


class PositionTag(str, Enum):
    """Position-level summary used for stratified eval-set generation."""
    HAS_IMMEDIATE_WIN = "has_immediate_win"
    MUST_BLOCK_THREAT = "must_block_threat"
    HAS_DOUBLE_THREAT_MOVE = "has_double_threat_move"
    HAS_FORCING_THREAT = "has_forcing_threat"
    QUIET = "quiet"


# ----------------------------------------------------------------------------
# Internal board helpers use row 0 = top and row 5 = floor, matching
# ConnectFourEnv._get_board_from_history.
# ----------------------------------------------------------------------------


_board_array = board_array
_landing_row = landing_row
_has_four_through = has_four_through
_winning_columns = winning_columns


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


@dataclass
class MoveAnalysis:
    column: int
    tag: StrategicTag
    # Auxiliary information for any consumer that wants more than the tag.
    opponent_winning_replies: List[int]
    self_immediate_wins_after: List[int]
    blocked_opponent_winning_columns: List[int]


def analyze_move(env: ConnectFourEnv, col: int) -> Optional[MoveAnalysis]:
    """Return a MoveAnalysis for `col` from `env`'s current player's POV.

    Returns None if the column is illegal or the env is terminal.
    """
    if env.is_terminal():
        return None
    if col not in env.legal_moves():
        return None

    cur = env.current_player()
    opp = 2 if cur == 1 else 1
    board = _board_array(env).copy()

    # 1. Immediate win?
    row = _landing_row(board, col)
    assert row is not None  # legal column implies at least one open row
    board[row][col] = cur
    is_immediate_win = _has_four_through(board, row, col, cur)

    # Determine pre-existing opponent winning columns BEFORE our move (we use
    # the unmodified board for that, so undo our mark temporarily).
    board[row][col] = 0
    opp_wins_before = _winning_columns(board, opp)
    # Now apply our move and check post-state opponent threats.
    board[row][col] = cur
    try:
        opp_wins_after = _winning_columns(board, opp)
        self_wins_after = _winning_columns(board, cur)
    finally:
        # Leave the array in its original state in case the caller reuses it.
        board[row][col] = 0

    # Did our move block one of opponent's pre-existing immediate-win columns?
    blocked_cols = [c for c in opp_wins_before if c not in opp_wins_after]

    # Tag selection (priority order).
    if is_immediate_win:
        tag = StrategicTag.IMMEDIATE_WIN
    elif col in opp_wins_before:
        # We played in the column that the opponent could have used to win
        # next turn. This is the most direct form of blocking.
        tag = StrategicTag.BLOCK_IMMEDIATE_THREAT
    elif opp_wins_after:
        # Opponent has at least one immediate-win column after our move.
        tag = StrategicTag.ALLOWS_OPPONENT_IMMEDIATE_WIN
    elif len(self_wins_after) >= 2:
        tag = StrategicTag.CREATES_DOUBLE_THREAT
    elif len(self_wins_after) == 1:
        tag = StrategicTag.CREATES_THREAT
    elif blocked_cols:
        # We didn't block an *immediate* threat (handled above) but we did
        # remove one of the opponent's later winning columns by occupying
        # the landing cell that they would otherwise have used.
        tag = StrategicTag.BLOCKS_OPPONENT_THREAT
    elif col == CENTER_COL:
        tag = StrategicTag.CENTER_PLAY
    else:
        tag = StrategicTag.NEUTRAL

    return MoveAnalysis(
        column=col,
        tag=tag,
        opponent_winning_replies=opp_wins_after,
        self_immediate_wins_after=self_wins_after,
        blocked_opponent_winning_columns=blocked_cols,
    )


def classify_move(env: ConnectFourEnv, col: int) -> Optional[StrategicTag]:
    """Convenience wrapper returning only the tag."""
    a = analyze_move(env, col)
    return a.tag if a is not None else None


def analyze_position(env: ConnectFourEnv) -> Dict[int, MoveAnalysis]:
    """Run analyze_move on every legal column."""
    out: Dict[int, MoveAnalysis] = {}
    for c in env.legal_moves():
        a = analyze_move(env, c)
        if a is not None:
            out[c] = a
    return out


def classify_position(env: ConnectFourEnv) -> PositionTag:
    """Cheap deterministic stratum for the position as a whole.

    HAS_IMMEDIATE_WIN     : current player can win this turn.
    MUST_BLOCK_THREAT     : opponent has an immediate-win column that the
                            current player has to address.
    HAS_DOUBLE_THREAT_MOVE: at least one move creates a double threat.
    HAS_FORCING_THREAT    : at least one move creates a single threat.
    QUIET                 : none of the above.

    These tags are computed without the solver and are stable regardless of
    Pons availability.
    """
    if env.is_terminal():
        return PositionTag.QUIET
    cur = env.current_player()
    opp = 2 if cur == 1 else 1
    board = _board_array(env)
    if _winning_columns(board, cur):
        return PositionTag.HAS_IMMEDIATE_WIN
    if _winning_columns(board, opp):
        return PositionTag.MUST_BLOCK_THREAT
    analyses = analyze_position(env)
    if any(a.tag is StrategicTag.CREATES_DOUBLE_THREAT for a in analyses.values()):
        return PositionTag.HAS_DOUBLE_THREAT_MOVE
    if any(a.tag is StrategicTag.CREATES_THREAT for a in analyses.values()):
        return PositionTag.HAS_FORCING_THREAT
    return PositionTag.QUIET


# ----------------------------------------------------------------------------
# Rule-based reference agent
# ----------------------------------------------------------------------------


@dataclass
class RuleBasedDecision:
    column: int
    tag: StrategicTag
    rationale: str
    candidates: Dict[int, StrategicTag]


def rule_based_move(
    env: ConnectFourEnv,
    *,
    prefer_center: bool = True,
) -> Optional[RuleBasedDecision]:
    """Pick a move using deterministic Connect Four heuristics only.

    Decision priority:
        1. Take an immediate win if any.
        2. Otherwise, block any immediate opponent threat.
        3. Otherwise, prefer moves that create a double threat.
        4. Otherwise, prefer moves that create a single threat without
           handing the opponent an immediate win.
        5. Otherwise, prefer moves that block an opponent's future winning
           column (BLOCKS_OPPONENT_THREAT).
        6. Otherwise, with prefer_center=True, take the center column if legal.
        7. Otherwise, take the legal column closest to center.
        8. Returns None if no legal moves.

    This is intentionally simple and beats most random opponents while being
    auditable. It is NOT optimal — Pons / minimax can outperform it. The
    point is to have a reproducible non-solver baseline whose decisions are
    fully explained by the rules above.
    """
    if env.is_terminal():
        return None
    legal = env.legal_moves()
    if not legal:
        return None

    analyses = analyze_position(env)
    by_tag: Dict[StrategicTag, List[int]] = {}
    for col, a in analyses.items():
        by_tag.setdefault(a.tag, []).append(col)

    def first(*tags: StrategicTag) -> Optional[int]:
        for t in tags:
            cols = sorted(by_tag.get(t, []), key=lambda c: abs(c - CENTER_COL))
            if cols:
                return cols[0]
        return None

    # 1. immediate win
    win = first(StrategicTag.IMMEDIATE_WIN)
    if win is not None:
        return RuleBasedDecision(
            column=win,
            tag=StrategicTag.IMMEDIATE_WIN,
            rationale="take immediate win",
            candidates={c: a.tag for c, a in analyses.items()},
        )

    # 2. block immediate threat
    block = first(StrategicTag.BLOCK_IMMEDIATE_THREAT)
    if block is not None:
        return RuleBasedDecision(
            column=block,
            tag=StrategicTag.BLOCK_IMMEDIATE_THREAT,
            rationale="block opponent's immediate winning move",
            candidates={c: a.tag for c, a in analyses.items()},
        )

    # 3-5. Avoid moves that allow opponent immediate win when possible.
    safe_cols = [
        c
        for c, a in analyses.items()
        if a.tag is not StrategicTag.ALLOWS_OPPONENT_IMMEDIATE_WIN
    ]

    def first_safe(*tags: StrategicTag) -> Optional[int]:
        for t in tags:
            cols = sorted(
                (c for c in by_tag.get(t, []) if c in safe_cols),
                key=lambda c: abs(c - CENTER_COL),
            )
            if cols:
                return cols[0]
        return None

    double = first_safe(StrategicTag.CREATES_DOUBLE_THREAT)
    if double is not None:
        return RuleBasedDecision(
            column=double,
            tag=StrategicTag.CREATES_DOUBLE_THREAT,
            rationale="create a double threat",
            candidates={c: a.tag for c, a in analyses.items()},
        )

    threat = first_safe(StrategicTag.CREATES_THREAT)
    if threat is not None:
        return RuleBasedDecision(
            column=threat,
            tag=StrategicTag.CREATES_THREAT,
            rationale="create a single threat",
            candidates={c: a.tag for c, a in analyses.items()},
        )

    deny = first_safe(StrategicTag.BLOCKS_OPPONENT_THREAT)
    if deny is not None:
        return RuleBasedDecision(
            column=deny,
            tag=StrategicTag.BLOCKS_OPPONENT_THREAT,
            rationale="deny opponent a future winning cell",
            candidates={c: a.tag for c, a in analyses.items()},
        )

    # 6-7. Quiet moves: prefer center, then closest-to-center safe column.
    if prefer_center and CENTER_COL in safe_cols:
        return RuleBasedDecision(
            column=CENTER_COL,
            tag=analyses[CENTER_COL].tag,
            rationale="quiet center play",
            candidates={c: a.tag for c, a in analyses.items()},
        )
    if safe_cols:
        col = sorted(safe_cols, key=lambda c: abs(c - CENTER_COL))[0]
        return RuleBasedDecision(
            column=col,
            tag=analyses[col].tag,
            rationale="closest-to-center safe move",
            candidates={c: a.tag for c, a in analyses.items()},
        )
    # Forced: every legal move allows opponent immediate win.
    col = sorted(legal, key=lambda c: abs(c - CENTER_COL))[0]
    return RuleBasedDecision(
        column=col,
        tag=StrategicTag.ALLOWS_OPPONENT_IMMEDIATE_WIN,
        rationale="forced loss: every move allows opponent immediate win",
        candidates={c: a.tag for c, a in analyses.items()},
    )
