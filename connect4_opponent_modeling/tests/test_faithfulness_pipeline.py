"""Tests for the causal-influence pipeline using a deterministic stub sample_fn."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.causal.pipeline import evaluate_response_causality
from faithfulness.claims import Claim, ClaimType
from faithfulness.parse import ParsedResponse


def _env_with_threat():
    env = ConnectFourEnv()
    env.from_move_sequence([3, 4, 3, 4, 0, 4])
    return env


def _make_parsed(claims, chosen_move):
    return ParsedResponse(raw="", valid_json=True, claims=claims, chosen_move=chosen_move)


def test_decorative_truth_is_not_causal():
    """If the model picks the same move regardless of intervention, claims are
    decorative even if true."""
    env = _env_with_threat()
    solver = PonsSolver(fallback_depth=4)
    claims = [
        Claim(id="c1", type=ClaimType.OPPONENT_IMMEDIATE_WIN, fields={"column": 4}),
        Claim(id="c2", type=ClaimType.OPTIMAL_MOVE, fields={"column": 4}),
    ]
    parsed = _make_parsed(claims, chosen_move=4)

    def stub_sample(messages, n):
        # Always return chosen_move=4 regardless of injected claims.
        user = next(m["content"] for m in messages if m["role"] == "user")
        assert "external oracle" not in user.lower()
        assert "Analysis prefix:" in user
        return ['{"chosen_move": 4}'] * n

    result = evaluate_response_causality(
        env, parsed, stub_sample, solver, n_resamples=10, threshold=0.25,
        rng=random.Random(0),
    )
    assert result.chosen_move == 4
    for cr in result.per_claim:
        assert cr.is_causal is False


def test_causal_when_intervention_shifts_move():
    """If interventions shift chosen_move, we detect causal influence."""
    env = _env_with_threat()
    solver = PonsSolver(fallback_depth=4)
    claims = [
        Claim(id="c1", type=ClaimType.OPPONENT_IMMEDIATE_WIN, fields={"column": 4}),
    ]
    parsed = _make_parsed(claims, chosen_move=4)

    seen_calls = {"n": 0}

    def stub_sample(messages, n):
        seen_calls["n"] += 1
        # Inspect the user message: if no claim mentions column 4 (deletion or
        # column change), the "model" picks 0 instead of 4.
        user = next(m["content"] for m in messages if m["role"] == "user")
        try:
            # The intervention prompt embeds the claim list as JSON.
            start = user.index("[")
            end = user.rindex("]") + 1
            claim_json = user[start:end]
            claim_list = json.loads(claim_json)
            mentions_4 = any(
                c.get("column") == 4 or c.get("opponent_reply") == 4
                for c in claim_list
            )
        except Exception:
            mentions_4 = False
        if mentions_4:
            return ['{"chosen_move": 4}'] * n
        return ['{"chosen_move": 0}'] * n

    result = evaluate_response_causality(
        env, parsed, stub_sample, solver, n_resamples=10, threshold=0.25,
        rng=random.Random(0),
    )
    # The single claim should now be causal because delete + replace_with_false
    # both shift the chosen_move away from 4.
    assert any(cr.is_causal for cr in result.per_claim)


def test_sampling_noise_baseline_does_not_create_causality():
    """A noisy model is not causal if interventions preserve the same noisy distribution."""
    env = _env_with_threat()
    solver = PonsSolver(fallback_depth=4)
    claims = [
        Claim(id="c1", type=ClaimType.OPPONENT_IMMEDIATE_WIN, fields={"column": 4}),
    ]
    parsed = _make_parsed(claims, chosen_move=4)

    def stub_sample(messages, n):
        return [
            '{"chosen_move": 4}' if i % 2 == 0 else '{"chosen_move": 0}'
            for i in range(n)
        ]

    result = evaluate_response_causality(
        env, parsed, stub_sample, solver, n_resamples=10, threshold=0.25,
        rng=random.Random(0),
    )
    assert result.original_distribution == {"0": 0.5, "4": 0.5}
    assert result.per_claim[0].is_causal is False


def test_false_causal_claim_is_detected():
    env = _env_with_threat()
    solver = PonsSolver(fallback_depth=4)
    claims = [
        Claim(id="c1", type=ClaimType.OPPONENT_IMMEDIATE_WIN, fields={"column": 0}),
    ]
    parsed = _make_parsed(claims, chosen_move=0)

    def stub_sample(messages, n):
        user = next(m["content"] for m in messages if m["role"] == "user")
        return ['{"chosen_move": 0}'] * n if '"column": 0' in user else ['{"chosen_move": 4}'] * n

    result = evaluate_response_causality(
        env, parsed, stub_sample, solver, n_resamples=10, threshold=0.25,
        rng=random.Random(0),
    )
    cr = result.per_claim[0]
    assert cr.truth is False
    assert cr.is_causal is True
