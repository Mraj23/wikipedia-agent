"""Tests for the training-pool generator (no Pons calls)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from faithfulness.eval.training_pool import (
    env_from_training_record,
    generate_training_pool,
    stratify_summary,
)


def test_generate_training_pool_basic():
    records = generate_training_pool(n_positions=100, seed=0, max_games=200)
    assert len(records) == 100
    for rec in records:
        assert isinstance(rec["moves"], str)
        assert all(ch in "0123456" for ch in rec["moves"])
        assert rec["current_player"] in (1, 2)
        assert isinstance(rec["legal_moves"], list)
        assert isinstance(rec["move_tags"], dict)
        assert rec["ply"] == len(rec["moves"])
        assert rec["position_tag"] in {
            "has_immediate_win",
            "must_block_threat",
            "has_double_threat_move",
            "has_forcing_threat",
            "quiet",
        }


def test_dedup_drops_duplicates():
    records = generate_training_pool(n_positions=200, seed=1, max_games=50, dedup=True)
    sequences = [r["moves"] for r in records]
    assert len(sequences) == len(set(sequences))


def test_env_round_trip():
    records = generate_training_pool(n_positions=10, seed=2, max_games=20)
    for rec in records:
        env = env_from_training_record(rec)
        assert sorted(env.legal_moves()) == sorted(rec["legal_moves"])
        assert env.current_player() == rec["current_player"]


def test_target_per_tag_caps():
    cap = 5
    records = generate_training_pool(
        n_positions=500,
        seed=3,
        max_games=2000,
        target_per_tag={
            "has_immediate_win": cap,
            "must_block_threat": cap,
            "has_double_threat_move": cap,
            "has_forcing_threat": cap,
            "quiet": cap,
        },
    )
    summary = stratify_summary(records)
    for tag, n in summary.items():
        assert n <= cap, f"{tag} exceeded cap: {n} > {cap}"
