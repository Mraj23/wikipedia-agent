"""Tests for the structured-response JSON parser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from faithfulness.claims import ClaimType
from faithfulness.parse import parse_structured_response


def test_happy_path():
    text = '{"claims": [{"id": "c1", "type": "self_immediate_win", "column": 3}], "chosen_move": 3}'
    out = parse_structured_response(text)
    assert out.valid_json
    assert out.chosen_move == 3
    assert len(out.claims) == 1
    assert out.claims[0].type is ClaimType.SELF_IMMEDIATE_WIN
    assert out.claims[0].fields == {"column": 3}
    assert out.parse_error is None
    assert out.dropped_claims == 0


def test_strips_code_fence():
    text = "```json\n" + '{"claims": [], "chosen_move": 4}' + "\n```"
    out = parse_structured_response(text)
    assert out.valid_json
    assert out.chosen_move == 4


def test_strips_unlabeled_code_fence():
    text = "```\n" + '{"claims": [], "chosen_move": 2}' + "\n```"
    out = parse_structured_response(text)
    assert out.valid_json
    assert out.chosen_move == 2


def test_outermost_brace_recovery():
    text = "Here is my move:\n" + '{"claims": [], "chosen_move": 1}' + "\nThanks!"
    out = parse_structured_response(text)
    assert out.valid_json
    assert out.chosen_move == 1


def test_drops_unknown_type():
    text = '{"claims": [{"id": "c1", "type": "bogus", "column": 3}], "chosen_move": 0}'
    out = parse_structured_response(text)
    assert out.valid_json
    assert out.claims == []
    assert out.dropped_claims == 1


def test_drops_missing_required_field():
    text = '{"claims": [{"id": "c1", "type": "move_allows_opponent_win", "move": 3}], "chosen_move": 0}'
    out = parse_structured_response(text)
    assert out.valid_json
    assert out.claims == []
    assert out.dropped_claims == 1


def test_invalid_json_recorded():
    text = "{this is not json"
    out = parse_structured_response(text)
    assert not out.valid_json
    assert out.parse_error is not None
    assert out.chosen_move is None


def test_string_column_coerced_to_int():
    text = '{"claims": [{"id": "c1", "type": "legal_move", "column": "4"}], "chosen_move": "4"}'
    out = parse_structured_response(text)
    assert out.valid_json
    assert out.chosen_move == 4
    assert out.claims[0].fields == {"column": 4}


def test_extra_fields_ignored():
    text = '{"claims": [], "chosen_move": 3, "extra": "junk"}'
    out = parse_structured_response(text)
    assert out.valid_json
    assert out.chosen_move == 3
