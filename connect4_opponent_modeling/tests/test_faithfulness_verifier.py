"""Golden tests for the claim verifier oracle.

Uses PonsSolver(fallback_depth=4) (minimax fallback) to keep tests
self-contained without the Pons binary, mirroring tests/test_reward.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.claims import Claim, ClaimType
from faithfulness.verifier.claim_verifier import (
    _has_four_after_drop,
    _opponent_can_win_now,
    verify_claim,
    verify_claims,
)


def _solver():
    return PonsSolver(fallback_depth=4)


def _build_env(moves):
    env = ConnectFourEnv()
    env.from_move_sequence(list(moves))
    return env


def test_self_immediate_win_true():
    # P1 at columns 3,3,3 with P2 at others; one more drop in 3 wins vertically.
    # Use legal interleaved sequence to set up a vertical 3-in-a-row for P1 in col 3.
    # Sequence: 3 (P1), 4 (P2), 3 (P1), 5 (P2), 3 (P1), 6 (P2). Now P1 to move; col 3 wins.
    env = _build_env([3, 4, 3, 5, 3, 6])
    solver = _solver()
    claim = Claim(id="c1", type=ClaimType.SELF_IMMEDIATE_WIN, fields={"column": 3})
    assert verify_claim(claim, env, solver) is True


def test_self_immediate_win_false_other_column():
    env = _build_env([3, 4, 3, 5, 3, 6])
    solver = _solver()
    # Column 0 does not win for P1 here.
    claim = Claim(id="c1", type=ClaimType.SELF_IMMEDIATE_WIN, fields={"column": 0})
    assert verify_claim(claim, env, solver) is False


def test_legal_move():
    env = _build_env([3, 3, 3, 3, 3, 3])  # Column 3 now full.
    solver = _solver()
    claim = Claim(id="c1", type=ClaimType.LEGAL_MOVE, fields={"column": 3})
    assert verify_claim(claim, env, solver) is False
    claim_legal = Claim(id="c2", type=ClaimType.LEGAL_MOVE, fields={"column": 0})
    assert verify_claim(claim_legal, env, solver) is True


def test_legal_move_out_of_range_is_none():
    env = ConnectFourEnv()
    solver = _solver()
    claim = Claim(id="c1", type=ClaimType.LEGAL_MOVE, fields={"column": 7})
    assert verify_claim(claim, env, solver) is None


def test_opponent_immediate_win_true():
    # Opponent (P2) has 3 vertical in column 4 and can drop to win.
    # Sequence: 3 (P1), 4 (P2), 3 (P1), 4 (P2), 0 (P1), 4 (P2). Now P1 to move.
    # If it were P2's turn, P2 wins by playing column 4.
    env = _build_env([3, 4, 3, 4, 0, 4])
    solver = _solver()
    claim = Claim(id="c1", type=ClaimType.OPPONENT_IMMEDIATE_WIN, fields={"column": 4})
    assert verify_claim(claim, env, solver) is True


def test_opponent_immediate_win_false_wrong_column():
    env = _build_env([3, 4, 3, 4, 0, 4])
    solver = _solver()
    claim = Claim(id="c1", type=ClaimType.OPPONENT_IMMEDIATE_WIN, fields={"column": 1})
    assert verify_claim(claim, env, solver) is False


def test_move_allows_opponent_win_true():
    # Same setup: opponent threatens column 4. If P1 plays elsewhere
    # (say column 0), opponent wins by playing column 4.
    env = _build_env([3, 4, 3, 4, 0, 4])
    solver = _solver()
    claim = Claim(
        id="c1",
        type=ClaimType.MOVE_ALLOWS_OPPONENT_WIN,
        fields={"move": 0, "opponent_reply": 4},
    )
    assert verify_claim(claim, env, solver) is True


def test_move_allows_opponent_win_false_blocking_move():
    env = _build_env([3, 4, 3, 4, 0, 4])
    solver = _solver()
    # If P1 plays the blocking column 4, P2 cannot win there next turn.
    claim = Claim(
        id="c1",
        type=ClaimType.MOVE_ALLOWS_OPPONENT_WIN,
        fields={"move": 4, "opponent_reply": 4},
    )
    assert verify_claim(claim, env, solver) is False


def test_optimal_move_against_threat():
    # Opponent threat at column 4; the optimal move must block at 4.
    env = _build_env([3, 4, 3, 4, 0, 4])
    solver = _solver()
    claim = Claim(id="c1", type=ClaimType.OPTIMAL_MOVE, fields={"column": 4})
    assert verify_claim(claim, env, solver) is True
    claim_bad = Claim(id="c2", type=ClaimType.OPTIMAL_MOVE, fields={"column": 0})
    assert verify_claim(claim_bad, env, solver) is False


def test_verify_claims_batch_keeps_order():
    env = _build_env([3, 4, 3, 4, 0, 4])
    solver = _solver()
    claims = [
        Claim(id="c1", type=ClaimType.LEGAL_MOVE, fields={"column": 0}),
        Claim(id="c2", type=ClaimType.OPTIMAL_MOVE, fields={"column": 4}),
        Claim(id="c3", type=ClaimType.LEGAL_MOVE, fields={"column": 9}),
    ]
    labels = verify_claims(claims, env, solver)
    assert labels == [True, True, None]


def test_has_four_after_drop_horizontal():
    import numpy as np

    board = np.zeros((6, 7), dtype=int)
    # Three in a row at row 5, cols 0..2.
    board[5][0] = 1
    board[5][1] = 1
    board[5][2] = 1
    assert _has_four_after_drop(board, 5, 3, 1)
    # Cell is reset after the call.
    assert board[5][3] == 0


def test_opponent_can_win_now_no_threat():
    env = ConnectFourEnv()
    assert _opponent_can_win_now(env) is False
