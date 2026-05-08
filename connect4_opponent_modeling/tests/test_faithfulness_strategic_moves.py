"""Tests for the deterministic strategic-move classifier.

Pure board logic, no solver. Mirrors the verifier golden-position style.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from faithfulness.strategic_moves import (
    PositionTag,
    StrategicTag,
    analyze_move,
    analyze_position,
    classify_move,
    classify_position,
    rule_based_move,
)


def _env(moves):
    env = ConnectFourEnv()
    env.from_move_sequence(list(moves))
    return env


def test_immediate_win_classification():
    # P1 has 3-in-a-row vertically in col 3; col 3 wins.
    env = _env([3, 4, 3, 5, 3, 6])
    a = analyze_move(env, 3)
    assert a is not None
    assert a.tag is StrategicTag.IMMEDIATE_WIN


def test_block_immediate_threat_classification():
    # Opponent threatens col 4 vertically; blocking move is col 4.
    env = _env([3, 4, 3, 4, 0, 4])
    a = analyze_move(env, 4)
    assert a is not None
    assert a.tag is StrategicTag.BLOCK_IMMEDIATE_THREAT


def test_allows_opponent_immediate_win():
    # Same threat. If P1 plays col 0, P2 wins next turn at col 4.
    env = _env([3, 4, 3, 4, 0, 4])
    a = analyze_move(env, 0)
    assert a is not None
    assert a.tag is StrategicTag.ALLOWS_OPPONENT_IMMEDIATE_WIN
    assert 4 in a.opponent_winning_replies


def test_creates_threat_classification():
    # P1 builds a horizontal pair on the bottom row that creates a single
    # threat after another build. We use a constructed position where
    # playing col 2 gives P1 three in a row with one open cell that
    # completes a 4-in-a-row next turn.
    # Sequence: 3 (P1), 6 (P2), 4 (P1), 6 (P2). Now P1 to move; play col 2
    # → P1 has pieces at 2,3,4 in row 0; col 1 or col 5 would complete.
    env = _env([3, 6, 4, 6])
    a = analyze_move(env, 2)
    assert a is not None
    # Can be CREATES_THREAT or CREATES_DOUBLE_THREAT depending on whether
    # both flanks are open. With cols 1 and 5 both open, this is a double.
    assert a.tag in {StrategicTag.CREATES_THREAT, StrategicTag.CREATES_DOUBLE_THREAT}
    # Ensure we recorded at least one immediate-win column for P1 next turn.
    assert len(a.self_immediate_wins_after) >= 1


def test_creates_double_threat_classification():
    # Same as above: 2-3-4 horizontal with both flanks open → double threat.
    env = _env([3, 6, 4, 6])
    a = analyze_move(env, 2)
    assert a is not None
    assert a.tag is StrategicTag.CREATES_DOUBLE_THREAT
    assert set(a.self_immediate_wins_after) >= {1, 5}


def test_neutral_or_center_classification_on_empty_board():
    env = ConnectFourEnv()
    a = analyze_move(env, 3)
    assert a is not None
    # Center column with no other content — center play.
    assert a.tag is StrategicTag.CENTER_PLAY
    a_neutral = analyze_move(env, 0)
    assert a_neutral is not None
    assert a_neutral.tag is StrategicTag.NEUTRAL


def test_classify_position_immediate_win():
    env = _env([3, 4, 3, 5, 3, 6])
    assert classify_position(env) is PositionTag.HAS_IMMEDIATE_WIN


def test_classify_position_must_block():
    env = _env([3, 4, 3, 4, 0, 4])
    assert classify_position(env) is PositionTag.MUST_BLOCK_THREAT


def test_classify_position_quiet_on_empty_board():
    env = ConnectFourEnv()
    assert classify_position(env) is PositionTag.QUIET


def test_analyze_position_returns_one_per_legal_move():
    env = _env([3, 4, 3, 4, 0, 4])
    analyses = analyze_position(env)
    assert set(analyses.keys()) == set(env.legal_moves())


def test_classify_move_illegal_returns_none():
    env = ConnectFourEnv()
    env.from_move_sequence([3] * 6)  # column 3 full
    assert classify_move(env, 3) is None


def test_rule_based_takes_immediate_win():
    env = _env([3, 4, 3, 5, 3, 6])
    decision = rule_based_move(env)
    assert decision is not None
    assert decision.column == 3
    assert decision.tag is StrategicTag.IMMEDIATE_WIN


def test_rule_based_blocks_threat():
    env = _env([3, 4, 3, 4, 0, 4])
    decision = rule_based_move(env)
    assert decision is not None
    assert decision.column == 4
    assert decision.tag is StrategicTag.BLOCK_IMMEDIATE_THREAT


def test_rule_based_prefers_center_when_quiet():
    env = ConnectFourEnv()
    decision = rule_based_move(env)
    assert decision is not None
    assert decision.column == 3  # center column


def test_rule_based_avoids_giving_opponent_immediate_win():
    """Construct a position where one move loses, another is safe — the
    rule-based agent must not pick the losing move."""
    # P2 threatens col 4 vertically. The only move that doesn't let P2 win
    # next turn is col 4 itself. Block must be chosen.
    env = _env([3, 4, 3, 4, 0, 4])
    decision = rule_based_move(env)
    assert decision is not None
    assert decision.column == 4


def test_rule_based_returns_none_for_terminal():
    env = ConnectFourEnv()
    # Play P1 vertical win in col 3.
    env.from_move_sequence([3, 4, 3, 4, 3, 4, 3])
    assert env.is_terminal()
    assert rule_based_move(env) is None
