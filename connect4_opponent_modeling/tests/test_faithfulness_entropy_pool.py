"""Tests for entropy-filtered training pool logic (no real model calls)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from faithfulness.eval.entropy_pool import (
    generate_entropy_pool,
    passes_entropy_filter,
    score_entropy_candidate,
    shannon_entropy,
)


class FakeSolver:
    def analyze(self, env):
        return {c: (10 if c == 3 else 0) for c in env.legal_moves()}


def _json_move(col):
    return (
        '{"claims":[{"id":"c1","type":"optimal_move","column":%d}],'
        '"rationale":"test","chosen_move":%d}'
    ) % (col, col)


def test_entropy_filter_keeps_uncertain_high_spread_candidate():
    env = ConnectFourEnv()
    completions = [_json_move(c) for c in [2, 3, 2, 3, 4, 3, 2, 4]]
    record = score_entropy_candidate(env, "", completions, FakeSolver())

    assert record is not None
    meta = record["entropy_filter"]
    assert meta["valid_json_rate"] == 1.0
    assert meta["legal_move_rate"] == 1.0
    assert meta["most_common_move_pct"] == 0.375
    assert meta["unique_legal_moves_sampled"] == 3
    assert meta["solver_score_spread"] >= 0.5
    assert passes_entropy_filter(record)


def test_entropy_filter_rejects_confident_or_flat_candidates():
    env = ConnectFourEnv()
    confident = score_entropy_candidate(
        env,
        "",
        [_json_move(3) for _ in range(8)],
        FakeSolver(),
    )
    assert confident is not None
    assert not passes_entropy_filter(confident)

    class FlatSolver:
        def analyze(self, env):
            return {c: 0 for c in env.legal_moves()}

    flat = score_entropy_candidate(
        env,
        "",
        [_json_move(c) for c in [1, 2, 1, 2, 3, 4, 3, 4]],
        FlatSolver(),
    )
    assert flat is not None
    assert not passes_entropy_filter(flat)


def test_generate_entropy_pool_uses_injected_sampler():
    def fake_sample_fn(messages, n):
        return [_json_move(c) for c in [2, 3, 2, 3, 4, 3, 2, 4]][:n]

    records = generate_entropy_pool(
        sample_fn=fake_sample_fn,
        solver=FakeSolver(),
        n_positions=3,
        seed=0,
        candidate_games=10,
        samples_per_board=8,
        max_candidates=30,
    )

    assert len(records) == 3
    assert all("entropy_filter" in rec for rec in records)


def test_shannon_entropy_zero_for_empty_distribution():
    assert shannon_entropy({}) == 0.0
