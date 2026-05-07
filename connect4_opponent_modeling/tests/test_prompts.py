"""Tests for prompt templates and parsers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from training.prompts import CONDITION_PROMPTS, format_prompt, parse_response, validate_response


def test_all_conditions_have_prompts():
    """All active conditions should have prompt templates."""
    for cond in [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "BaseSimple",
        "BaseScaffold",
        "Value",
        "OpponentNextMove",
    ]:
        assert cond in CONDITION_PROMPTS, f"Missing prompt for condition {cond}"


def test_format_prompt_fills_placeholders():
    """Board and legal-move placeholders are rendered."""
    env = ConnectFourEnv()
    for cond in ["A", "B", "C", "D", "E", "F", "G"]:
        prompt = format_prompt(cond, env)
        assert "{board}" not in prompt
        assert "{legal_moves}" not in prompt
        assert "." in prompt
        assert "0" in prompt


def test_parse_response_condition_a():
    """Extracts reasoning and move from the current schema."""
    resp = "<reasoning>I should play center.</reasoning><answer>3</answer>"
    parsed = parse_response(resp, "A")
    assert parsed["move"] == 3
    assert parsed["reasoning"] == "I should play center."


def test_parse_response_condition_a_allows_text_inside_answer_tag():
    """Parser should recover the first legal digit inside answer text."""
    resp = "<reasoning>Block now.</reasoning><answer>column 3</answer>"
    parsed = parse_response(resp, "A")
    assert parsed["move"] == 3


def test_parse_response_condition_d():
    """Extracts both auxiliary fields and move for condition D."""
    resp = (
        "<reasoning>Playing center.</reasoning>"
        "<future_state>. . . . . . .\n. . . . . . .</future_state>"
        "<opponent_prediction>4</opponent_prediction>"
        "<answer>3</answer>"
    )
    parsed = parse_response(resp, "D")
    assert parsed["move"] == 3
    assert parsed["future_state"] is not None
    assert parsed["opponent_prediction"] == 4
    assert "." in parsed["future_state"]


def test_parse_response_condition_e():
    """Extracts both auxiliary fields for condition E."""
    resp = (
        "<reasoning>If I play 3, opponent plays 4.</reasoning>"
        "<future_state>. . . . . . .\n. . . . . . .</future_state>"
        "<opponent_prediction>4</opponent_prediction>"
        "<answer>3</answer>"
    )
    parsed = parse_response(resp, "E")
    assert parsed["move"] == 3
    assert parsed["future_state"] is not None
    assert parsed["opponent_prediction"] == 4


def test_parse_response_opponent_next_move():
    """Extracts the narrow-experiment scaffold fields."""
    resp = (
        "<reasoning>If I play 3, opponent's best next move is 4.</reasoning>"
        "<opponent_prediction>4</opponent_prediction>"
        "<answer>3</answer>"
    )
    parsed = parse_response(resp, "OpponentNextMove")
    assert parsed["move"] == 3
    assert parsed["opponent_prediction"] == 4


def test_conditions_d_and_e_share_output_contract():
    """D and E should require the same structured fields."""
    env = ConnectFourEnv()
    prompt_d = format_prompt("D", env)
    prompt_e = format_prompt("E", env)
    for required in ["<future_state>", "<opponent_prediction>", "<answer>"]:
        assert required in prompt_d
        assert required in prompt_e


def test_validate_rejects_missing_tags():
    """Missing required tags are rejected."""
    parsed = parse_response("I play column 3", "A")
    valid, reason = validate_response(parsed, "A", [0, 1, 2, 3, 4, 5, 6])
    assert not valid
    assert "Missing" in reason


def test_validate_rejects_illegal_move():
    """Illegal moves are rejected."""
    parsed = parse_response("<reasoning>test</reasoning><answer>3</answer>", "A")
    valid, reason = validate_response(parsed, "A", [0, 1, 2, 4, 5, 6])
    assert not valid
    assert "not in legal moves" in reason


def test_validate_accepts_valid_response():
    """Valid responses pass validation."""
    parsed = parse_response("<reasoning>Center is strong.</reasoning><answer>3</answer>", "A")
    valid, reason = validate_response(parsed, "A", [0, 1, 2, 3, 4, 5, 6])
    assert valid
    assert reason == ""
