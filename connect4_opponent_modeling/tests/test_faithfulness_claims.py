"""Tests for the Claim/ClaimType data layer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from faithfulness.claims import COLUMN_FIELDS, REQUIRED_FIELDS, Claim, ClaimType


def test_claim_type_values_are_strings():
    assert ClaimType.SELF_IMMEDIATE_WIN.value == "self_immediate_win"
    assert ClaimType.OPTIMAL_MOVE.value == "optimal_move"


def test_required_fields_cover_all_types():
    for t in ClaimType:
        assert t in REQUIRED_FIELDS, f"missing required fields for {t}"


def test_column_fields_cover_all_types():
    for t in ClaimType:
        assert t in COLUMN_FIELDS, f"missing column fields for {t}"


def test_has_required_fields_positive():
    c = Claim(id="c1", type=ClaimType.MOVE_ALLOWS_OPPONENT_WIN, fields={"move": 3, "opponent_reply": 4})
    assert c.has_required_fields()


def test_has_required_fields_negative():
    c = Claim(id="c1", type=ClaimType.MOVE_ALLOWS_OPPONENT_WIN, fields={"move": 3})
    assert not c.has_required_fields()


def test_to_dict_round_trip_shape():
    c = Claim(id="c2", type=ClaimType.OPTIMAL_MOVE, fields={"column": 5})
    d = c.to_dict()
    assert d == {"id": "c2", "type": "optimal_move", "column": 5}
