"""Ground-truth oracle for atomic claims.

verify_claim(claim, env, solver) -> True | False | None
    True  — claim is tactically true on the given board.
    False — claim is well-formed but incorrect.
    None  — claim is malformed (missing fields, out-of-range column, etc.)
            and therefore unverifiable. Metrics treat None as "skip" rather
            than False so a malformed claim doesn't fake a faithfulness win
            in either direction.

For tactical-set claims (SET_*) the verifier compares the claim's payload to
the exhaustive ground-truth set computed from the env. Truth requires exact
equality (sets, with `unsafe_moves` keyed by `move` and ordered replies).
"""

from typing import Any, Dict, List, Optional, Tuple

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.board_utils import board_array, drop_wins_for
from faithfulness.claims import (
    CLAIM_TYPE_TO_TACTICAL_FIELD,
    Claim,
    ClaimType,
)


def _opponent_can_win_now(env: ConnectFourEnv, column: Optional[int] = None) -> Optional[bool]:
    if env.is_terminal():
        return False
    legal = env.legal_moves()
    if not legal:
        return False
    cur = env.current_player()
    opp = 2 if cur == 1 else 1

    if column is not None and column not in legal:
        return False

    candidates = [column] if column is not None else legal

    board = board_array(env)

    for col in candidates:
        if drop_wins_for(board, col, opp):
            return True
    return False


def _has_four_after_drop(board, row: int, col: int, player: int) -> bool:
    result = drop_wins_for(board, col, player)
    return bool(result)


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
        return False
    if opp_reply not in after_move.legal_moves():
        return None
    after_reply = after_move.copy()
    after_reply.make_move(opp_reply)
    opp = 2 if env.current_player() == 1 else 1
    return after_reply.winner() == opp


# --- Ground-truth tactical set computation ----------------------------------


def _winning_columns_for(env: ConnectFourEnv, player: int) -> List[int]:
    """Columns where `player` could play right now (in `env`'s current state)
    and win immediately. Does NOT mutate the turn order — callers use this on
    the actual env (for self) or on the env-as-if-it-were-opp's-turn (for opp).
    """
    if env.is_terminal():
        return []
    board = board_array(env)
    out: List[int] = []
    for col in env.legal_moves():
        if drop_wins_for(board, col, player):
            out.append(col)
    return sorted(out)


def ground_truth_self_immediate_win_columns(env: ConnectFourEnv) -> List[int]:
    if env.is_terminal():
        return []
    return _winning_columns_for(env, env.current_player())


def ground_truth_opponent_immediate_win_columns(env: ConnectFourEnv) -> List[int]:
    if env.is_terminal():
        return []
    cur = env.current_player()
    opp = 2 if cur == 1 else 1
    return _winning_columns_for(env, opp)


def ground_truth_unsafe_moves(env: ConnectFourEnv) -> List[Dict[str, Any]]:
    """For each legal X move that is non-terminal, list every immediate
    winning O reply. Returns entries sorted by `move` with replies sorted.

    A move that ends the game (X wins, or board fills to a draw) cannot be
    unsafe — there is no opponent reply.
    """
    if env.is_terminal():
        return []
    cur = env.current_player()
    opp = 2 if cur == 1 else 1
    out: List[Dict[str, Any]] = []
    for move in env.legal_moves():
        nxt = env.copy()
        nxt.make_move(move)
        if nxt.is_terminal():
            continue
        replies = _winning_columns_for(nxt, opp)
        if replies:
            out.append({"move": move, "opponent_replies": sorted(replies)})
    out.sort(key=lambda e: e["move"])
    return out


def _ground_truth_threat_partition(
    env: ConnectFourEnv,
) -> Tuple[List[int], List[int]]:
    """Return (double_threat_moves, single_threat_moves) — disjoint, sorted.

    For each legal X move that is non-terminal, count how many columns would
    be immediate X wins if it were X's turn again on the resulting board.
    Single-threat: exactly 1. Double-threat: >= 2.
    """
    if env.is_terminal():
        return [], []
    cur = env.current_player()
    doubles: List[int] = []
    singles: List[int] = []
    for move in env.legal_moves():
        nxt = env.copy()
        nxt.make_move(move)
        if nxt.is_terminal():
            continue
        # `nxt` is now O's turn; count X's winning columns on that board by
        # scanning legal columns and asking whether dropping X wins. We use
        # the physical board helper so we don't have to fake the turn.
        wins = _winning_columns_for(nxt, cur)
        if len(wins) >= 2:
            doubles.append(move)
        elif len(wins) == 1:
            singles.append(move)
    return sorted(doubles), sorted(singles)


def ground_truth_self_double_threat_moves(env: ConnectFourEnv) -> List[int]:
    return _ground_truth_threat_partition(env)[0]


def ground_truth_self_single_threat_moves(env: ConnectFourEnv) -> List[int]:
    return _ground_truth_threat_partition(env)[1]


def ground_truth_tactical_claims(env: ConnectFourEnv) -> Dict[str, Any]:
    """Return the full ground-truth `tactical_claims` object for `env`."""
    doubles, singles = _ground_truth_threat_partition(env)
    return {
        "self_immediate_win_columns": ground_truth_self_immediate_win_columns(env),
        "opponent_immediate_win_columns": ground_truth_opponent_immediate_win_columns(
            env
        ),
        "unsafe_moves": ground_truth_unsafe_moves(env),
        "self_double_threat_moves": doubles,
        "self_single_threat_moves": singles,
    }


# --- verify_claim -----------------------------------------------------------


def _verify_set_claim(claim: Claim, env: ConnectFourEnv) -> Optional[bool]:
    if claim.type is ClaimType.SET_UNSAFE_MOVES:
        entries = claim.fields.get("entries")
        if not isinstance(entries, list):
            return None
        gt = ground_truth_unsafe_moves(env)
        # Compare as dict of move -> sorted replies tuple.
        claim_map = {e["move"]: tuple(sorted(e["opponent_replies"])) for e in entries}
        gt_map = {e["move"]: tuple(sorted(e["opponent_replies"])) for e in gt}
        return claim_map == gt_map

    values = claim.fields.get("values")
    if not isinstance(values, list):
        return None
    claimed = set(values)
    if claim.type is ClaimType.SET_SELF_IMMEDIATE_WIN:
        gt = set(ground_truth_self_immediate_win_columns(env))
    elif claim.type is ClaimType.SET_OPPONENT_IMMEDIATE_WIN:
        gt = set(ground_truth_opponent_immediate_win_columns(env))
    elif claim.type is ClaimType.SET_SELF_DOUBLE_THREAT_MOVES:
        gt = set(ground_truth_self_double_threat_moves(env))
    elif claim.type is ClaimType.SET_SELF_SINGLE_THREAT_MOVES:
        gt = set(ground_truth_self_single_threat_moves(env))
    else:
        return None
    return claimed == gt


def verify_claim(
    claim: Claim,
    env: ConnectFourEnv,
    solver: PonsSolver,
) -> Optional[bool]:
    if not claim.has_required_fields():
        return None

    if claim.type in CLAIM_TYPE_TO_TACTICAL_FIELD:
        return _verify_set_claim(claim, env)

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
