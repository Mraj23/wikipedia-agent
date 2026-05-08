"""Tests for the claim-list interventions used by the causal pipeline."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.causal.interventions import (
    apply_intervention,
    change_column,
    delete_claim,
    replace_with_false_claim,
)
from faithfulness.claims import Claim, ClaimType
from faithfulness.verifier.claim_verifier import verify_claim


def _env_with_threat():
    env = ConnectFourEnv()
    env.from_move_sequence([3, 4, 3, 4, 0, 4])
    return env


def _claims_sample():
    return [
        Claim(id="c1", type=ClaimType.OPPONENT_IMMEDIATE_WIN, fields={"column": 4}),
        Claim(id="c2", type=ClaimType.OPTIMAL_MOVE, fields={"column": 4}),
        Claim(id="c3", type=ClaimType.LEGAL_MOVE, fields={"column": 0}),
    ]


def test_delete_shrinks_by_one():
    claims = _claims_sample()
    new, meta = delete_claim(claims, 1)
    assert len(new) == len(claims) - 1
    assert meta.kind == "delete"
    assert meta.original is claims[1]


def test_change_column_modifies_one_claim():
    env = _env_with_threat()
    claims = _claims_sample()
    rng = random.Random(0)
    new, meta = change_column(claims, 0, env.legal_moves(), rng=rng)
    # Only target index changed; others stay equal (deepcopy by value).
    assert new[1].fields == claims[1].fields
    assert new[2].fields == claims[2].fields
    assert new[0].fields["column"] != claims[0].fields["column"]
    assert meta.kind == "change_column"
    assert meta.succeeded


def test_replace_with_false_claim_yields_false_verdict():
    env = _env_with_threat()
    solver = PonsSolver(fallback_depth=4)
    claims = _claims_sample()  # c2 OPTIMAL_MOVE column=4 is True; we'll falsify it
    rng = random.Random(7)
    new, meta = replace_with_false_claim(claims, 1, env, solver, rng=rng)
    assert meta.kind == "replace_with_false"
    if meta.succeeded:
        assert verify_claim(new[1], env, solver) is False


def test_apply_intervention_dispatch():
    env = _env_with_threat()
    solver = PonsSolver(fallback_depth=4)
    claims = _claims_sample()
    new_del, _ = apply_intervention("delete", claims, 0, env, solver, rng=random.Random(0))
    assert len(new_del) == len(claims) - 1
    new_cc, _ = apply_intervention("change_column", claims, 0, env, solver, rng=random.Random(0))
    assert len(new_cc) == len(claims)


def test_apply_intervention_unknown_raises():
    env = _env_with_threat()
    solver = PonsSolver(fallback_depth=4)
    try:
        apply_intervention("nope", _claims_sample(), 0, env, solver)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown intervention kind")
