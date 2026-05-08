"""Smoke test for the stratified eval-set generator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.eval.board_generator import (
    CATEGORIES,
    _categorize,
    generate_stratified_eval_set,
)


def test_categorize_immediate_win():
    env = ConnectFourEnv()
    env.from_move_sequence([3, 4, 3, 5, 3, 6])  # P1 to move; col 3 wins
    cat = _categorize(env, PonsSolver(fallback_depth=4))
    assert cat == "immediate_win_available"


def test_categorize_opponent_threat_takes_precedence():
    env = ConnectFourEnv()
    env.from_move_sequence([3, 4, 3, 4, 0, 4])  # P1 to move; opponent threatens col 4
    cat = _categorize(env, PonsSolver(fallback_depth=4))
    # Could be opponent_immediate_threat or blunder_state. We only require
    # that we recognize the position as tactically charged.
    assert cat in {"opponent_immediate_threat", "blunder_state"}


def test_generate_stratified_eval_set_smoke():
    records = generate_stratified_eval_set(
        n_per_category=2,
        seed=0,
        solver=PonsSolver(fallback_depth=4),
        candidate_games=80,
    )
    cats = {r["category"] for r in records}
    # At least three of five categories should be reachable in a small pool.
    assert cats.issubset(set(CATEGORIES))
    assert len(cats) >= 3
    for r in records:
        assert "moves" in r and "verifier_meta" in r
        assert isinstance(r["verifier_meta"]["best_moves"], list)
