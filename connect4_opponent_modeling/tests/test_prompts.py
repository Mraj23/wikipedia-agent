"""Tests for prompt templates and parsers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from training.prompts import (
    CONDITION_PROMPTS,
    format_prompt,
    parse_response,
    validate_response,
)


def test_all_conditions_have_prompts():
    """A-F all present in CONDITION_PROMPTS."""
    for cond in ["A", "B", "C", "D", "E", "F"]:
        assert cond in CONDITION_PROMPTS, f"Missing prompt for condition {cond}"


def test_format_prompt_fills_placeholders():
    """Placeholders {board} and {legal_moves} are filled."""
    env = ConnectFourEnv()
    for cond in ["A", "B", "C", "D", "E", "F"]:
        prompt = format_prompt(cond, env)
        assert "{board}" not in prompt
        assert "{legal_moves}" not in prompt
        assert "." in prompt  # board has dots
        assert "0" in prompt  # legal moves include 0


def test_parse_response_condition_a():
    """Extracts move from <move> tag."""
    resp = "<think>I should play center.</think><move>3</move>"
    parsed = parse_response(resp, "A")
    assert parsed["move"] == 3
    assert parsed["think"] == "I should play center."


def test_parse_response_condition_d():
    """Extracts future_state and opponent_prediction."""
    resp = (
        "<think>Playing center.</think>"
        "<opponent_prediction>4</opponent_prediction>"
        "<future_state>. . . . . . .\n. . . . . . .</future_state>"
        "<move>3</move>"
    )
    parsed = parse_response(resp, "D")
    assert parsed["move"] == 3
    assert parsed["future_state"] is not None
    assert "." in parsed["future_state"]
    assert parsed["opponent_prediction"] == 4


def test_parse_response_condition_e():
    """Extracts opponent_prediction."""
    resp = (
        "<think>If I play 3, opponent plays 4.</think>"
        "<opponent_prediction>4</opponent_prediction>"
        "<move>3</move>"
    )
    parsed = parse_response(resp, "E")
    assert parsed["move"] == 3
    assert parsed["opponent_prediction"] == 4


def test_validate_rejects_missing_tags():
    """Missing required tags are rejected."""
    resp = "I play column 3"
    parsed = parse_response(resp, "A")
    valid, reason = validate_response(parsed, "A", [0, 1, 2, 3, 4, 5, 6])
    assert not valid
    assert "Missing" in reason


def test_validate_rejects_illegal_move():
    """Illegal moves are rejected."""
    resp = "<think>test</think><move>3</move>"
    parsed = parse_response(resp, "A")
    valid, reason = validate_response(parsed, "A", [0, 1, 2, 4, 5, 6])  # 3 not legal
    assert not valid
    assert "not in legal moves" in reason


def test_validate_accepts_valid_response():
    """Valid responses pass validation."""
    resp = "<think>I should play center.</think><move>3</move>"
    parsed = parse_response(resp, "A")
    valid, reason = validate_response(parsed, "A", [0, 1, 2, 3, 4, 5, 6])
    assert valid
    assert reason == ""


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
