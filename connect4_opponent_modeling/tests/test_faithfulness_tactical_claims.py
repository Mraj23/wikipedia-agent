"""Tests for the new strict tactical_claims schema (prompt + parse + verify
+ reward + interventions + pipeline)."""

import json
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
from faithfulness.causal.pipeline import evaluate_response_causality
from faithfulness.claims import (
    CLAIM_TYPE_TO_TACTICAL_FIELD,
    TACTICAL_FIELD_TO_CLAIM_TYPE,
    Claim,
    ClaimType,
)
from faithfulness.parse import ParsedResponse, parse_structured_response
from faithfulness.prompt import (
    TACTICAL_CLAIMS_SYSTEM_PROMPT,
    make_messages,
)
from faithfulness.rl.reward import (
    ILLEGAL_MOVE_PENALTY,
    LEGAL_MOVE_BONUS,
    VALID_JSON_BONUS,
    FaithfulnessRewardCalculator,
)
from faithfulness.verifier.claim_verifier import (
    ground_truth_opponent_immediate_win_columns,
    ground_truth_self_double_threat_moves,
    ground_truth_self_immediate_win_columns,
    ground_truth_self_single_threat_moves,
    ground_truth_tactical_claims,
    ground_truth_unsafe_moves,
    verify_claim,
    verify_claims,
)


def _solver():
    return PonsSolver(fallback_depth=4)


def _build_env(moves):
    env = ConnectFourEnv()
    env.from_move_sequence(list(moves))
    return env


# ---------------------------------------------------------------- prompt --


def test_tactical_prompt_contains_required_fields_and_excludes_legacy():
    env = ConnectFourEnv()
    msgs = make_messages(env, "tactical_claims")
    sys_prompt = msgs[0]["content"]
    assert sys_prompt is TACTICAL_CLAIMS_SYSTEM_PROMPT
    assert "tactical_claims" in sys_prompt
    assert "self_immediate_win_columns" in sys_prompt
    assert "opponent_immediate_win_columns" in sys_prompt
    assert "unsafe_moves" in sys_prompt
    assert "self_double_threat_moves" in sys_prompt
    assert "self_single_threat_moves" in sys_prompt
    assert "chosen_move" in sys_prompt
    # The forbidden answer-leak / free-text fields are mentioned by name only
    # in a "do NOT include" sentence; they must not be presented as schema keys.
    assert '"rationale"' not in sys_prompt
    assert '"claims"' not in sys_prompt
    assert '"optimal_move"' not in sys_prompt
    assert '"legal_move"' not in sys_prompt


# ----------------------------------------------------------------- parse --


def _ok_payload():
    return {
        "tactical_claims": {
            "self_immediate_win_columns": [],
            "opponent_immediate_win_columns": [],
            "unsafe_moves": [{"move": 2, "opponent_replies": [4]}],
            "self_double_threat_moves": [],
            "self_single_threat_moves": [3, 5],
        },
        "chosen_move": 3,
    }


def test_parse_tactical_happy_path_normalizes_order():
    payload = _ok_payload()
    payload["tactical_claims"]["self_single_threat_moves"] = [5, 3]  # unsorted ok
    text = json.dumps(payload)
    out = parse_structured_response(text, condition="tactical_claims")
    assert out.valid_json
    assert out.schema_valid
    assert out.chosen_move == 3
    by_type = {c.type: c for c in out.claims}
    assert by_type[ClaimType.SET_SELF_SINGLE_THREAT_MOVES].fields["values"] == [3, 5]
    assert by_type[ClaimType.SET_UNSAFE_MOVES].fields["entries"] == [
        {"move": 2, "opponent_replies": [4]}
    ]


def test_parse_tactical_rejects_duplicates():
    p = _ok_payload()
    p["tactical_claims"]["self_single_threat_moves"] = [3, 3]
    out = parse_structured_response(json.dumps(p), condition="tactical_claims")
    assert out.valid_json
    assert not out.schema_valid
    assert "duplicate" in (out.parse_error or "")


def test_parse_tactical_rejects_extra_top_level_key():
    p = _ok_payload()
    p["rationale"] = "i think this"
    out = parse_structured_response(json.dumps(p), condition="tactical_claims")
    assert not out.schema_valid


def test_parse_tactical_rejects_legacy_claims_field():
    p = _ok_payload()
    p["claims"] = []
    out = parse_structured_response(json.dumps(p), condition="tactical_claims")
    assert not out.schema_valid


def test_parse_tactical_rejects_missing_inner_field():
    p = _ok_payload()
    del p["tactical_claims"]["self_single_threat_moves"]
    out = parse_structured_response(json.dumps(p), condition="tactical_claims")
    assert not out.schema_valid


def test_parse_tactical_rejects_out_of_range_column():
    p = _ok_payload()
    p["tactical_claims"]["self_immediate_win_columns"] = [7]
    out = parse_structured_response(json.dumps(p), condition="tactical_claims")
    assert not out.schema_valid


def test_parse_tactical_rejects_string_column():
    p = _ok_payload()
    p["tactical_claims"]["self_immediate_win_columns"] = ["3"]
    out = parse_structured_response(json.dumps(p), condition="tactical_claims")
    assert not out.schema_valid


def test_parse_tactical_rejects_empty_replies():
    p = _ok_payload()
    p["tactical_claims"]["unsafe_moves"] = [{"move": 2, "opponent_replies": []}]
    out = parse_structured_response(json.dumps(p), condition="tactical_claims")
    assert not out.schema_valid


def test_parse_tactical_rejects_duplicate_unsafe_move():
    p = _ok_payload()
    p["tactical_claims"]["unsafe_moves"] = [
        {"move": 2, "opponent_replies": [4]},
        {"move": 2, "opponent_replies": [5]},
    ]
    out = parse_structured_response(json.dumps(p), condition="tactical_claims")
    assert not out.schema_valid


def test_parse_tactical_rejects_single_double_threat_overlap():
    p = _ok_payload()
    p["tactical_claims"]["self_single_threat_moves"] = [3]
    p["tactical_claims"]["self_double_threat_moves"] = [3]
    out = parse_structured_response(json.dumps(p), condition="tactical_claims")
    assert not out.schema_valid


def test_parse_tactical_rejects_extra_unsafe_move_key():
    p = _ok_payload()
    p["tactical_claims"]["unsafe_moves"] = [
        {"move": 2, "opponent_replies": [4], "wat": 1}
    ]
    out = parse_structured_response(json.dumps(p), condition="tactical_claims")
    assert not out.schema_valid


def test_parse_tactical_empty_arrays_are_valid():
    p = {
        "tactical_claims": {
            "self_immediate_win_columns": [],
            "opponent_immediate_win_columns": [],
            "unsafe_moves": [],
            "self_double_threat_moves": [],
            "self_single_threat_moves": [],
        },
        "chosen_move": 3,
    }
    out = parse_structured_response(json.dumps(p), condition="tactical_claims")
    assert out.schema_valid
    assert len(out.claims) == 5  # one Claim per tactical field


# -------------------------------------------------------------- verifier --


def test_ground_truth_self_immediate_win():
    env = _build_env([3, 4, 3, 5, 3, 6])  # X to move, drop col 3 to win.
    assert ground_truth_self_immediate_win_columns(env) == [3]


def test_ground_truth_opponent_immediate_win():
    env = _build_env([3, 4, 3, 4, 0, 4])  # X to move, O wins col 4.
    assert ground_truth_opponent_immediate_win_columns(env) == [4]


def test_ground_truth_unsafe_moves_includes_failure_to_block():
    env = _build_env([3, 4, 3, 4, 0, 4])
    gt = ground_truth_unsafe_moves(env)
    moves = {e["move"] for e in gt}
    # X playing anywhere except column 4 is unsafe; col 4 blocks.
    assert 4 not in moves
    assert 0 in moves and 1 in moves
    for entry in gt:
        assert entry["opponent_replies"] == [4]


def test_ground_truth_threats_partition_disjoint():
    env = _build_env([3, 4, 3, 5, 3, 6])  # X to move, col 3 wins immediately.
    doubles = ground_truth_self_double_threat_moves(env)
    singles = ground_truth_self_single_threat_moves(env)
    assert set(doubles).isdisjoint(set(singles))
    # The immediate-win move (col 3) should NOT appear in threat sets.
    assert 3 not in doubles
    assert 3 not in singles


def test_verify_set_claim_exact_equality():
    env = _build_env([3, 4, 3, 4, 0, 4])
    gt = ground_truth_tactical_claims(env)
    correct = Claim(
        id="opp",
        type=ClaimType.SET_OPPONENT_IMMEDIATE_WIN,
        fields={"values": gt["opponent_immediate_win_columns"]},
    )
    wrong = Claim(
        id="opp",
        type=ClaimType.SET_OPPONENT_IMMEDIATE_WIN,
        fields={"values": [0]},
    )
    solver = _solver()
    assert verify_claim(correct, env, solver) is True
    assert verify_claim(wrong, env, solver) is False


def test_verify_set_unsafe_moves_exact_equality():
    env = _build_env([3, 4, 3, 4, 0, 4])
    gt = ground_truth_unsafe_moves(env)
    solver = _solver()
    correct = Claim(
        id="u",
        type=ClaimType.SET_UNSAFE_MOVES,
        fields={"entries": list(gt)},
    )
    wrong = Claim(
        id="u",
        type=ClaimType.SET_UNSAFE_MOVES,
        fields={"entries": []},
    )
    assert verify_claim(correct, env, solver) is True
    assert verify_claim(wrong, env, solver) is False


# ----------------------------------------------------------------- reward --


def test_reward_schema_invalid_treated_as_invalid_json_even_with_legal_move():
    calc = FaithfulnessRewardCalculator(_solver(), condition="tactical_claims")
    env = _build_env([3, 4, 3, 4, 0, 4])
    # Valid JSON, legal move, but missing the strict tactical_claims object.
    text = '{"chosen_move": 4}'
    out = calc.compute(env, text)
    assert out.illegal_move
    assert out.reward == -ILLEGAL_MOVE_PENALTY
    assert out.debug.get("reason") == "schema_invalid"


def test_reward_schema_valid_legal_optimal_top_reward():
    calc = FaithfulnessRewardCalculator(_solver(), condition="tactical_claims")
    env = _build_env([3, 4, 3, 4, 0, 4])
    p = _ok_payload()
    p["chosen_move"] = 4  # optimal: blocks the threat
    out = calc.compute(env, json.dumps(p))
    assert out.valid_json
    assert out.legal_move
    # top reward is the same VALID + LEGAL bonus (regret zero on optimal move)
    assert out.reward == VALID_JSON_BONUS + LEGAL_MOVE_BONUS


# ------------------------------------------------------- interventions --


def _set_claims_for(env):
    """Construct the ground-truth set-claim list for a given env."""
    gt = ground_truth_tactical_claims(env)
    return [
        Claim(
            id="self_imm",
            type=ClaimType.SET_SELF_IMMEDIATE_WIN,
            fields={"values": gt["self_immediate_win_columns"]},
        ),
        Claim(
            id="opp_imm",
            type=ClaimType.SET_OPPONENT_IMMEDIATE_WIN,
            fields={"values": gt["opponent_immediate_win_columns"]},
        ),
        Claim(
            id="unsafe",
            type=ClaimType.SET_UNSAFE_MOVES,
            fields={"entries": list(gt["unsafe_moves"])},
        ),
        Claim(
            id="dbl",
            type=ClaimType.SET_SELF_DOUBLE_THREAT_MOVES,
            fields={"values": gt["self_double_threat_moves"]},
        ),
        Claim(
            id="sgl",
            type=ClaimType.SET_SELF_SINGLE_THREAT_MOVES,
            fields={"values": gt["self_single_threat_moves"]},
        ),
    ]


def test_delete_set_claim_empties_field_keeps_shape():
    env = _build_env([3, 4, 3, 4, 0, 4])
    claims = _set_claims_for(env)
    new, meta = delete_claim(claims, 1)  # delete opp_imm
    assert meta.kind == "delete"
    assert meta.succeeded
    assert len(new) == len(claims)  # shape preserved
    assert new[1].fields["values"] == []


def test_change_column_set_claim_mutates_one_value():
    env = _build_env([3, 4, 3, 4, 0, 4])
    claims = _set_claims_for(env)
    rng = random.Random(0)
    # change_column on opp_imm (which has [4]) should produce a different val.
    new, meta = change_column(claims, 1, env.legal_moves(), env=env, rng=rng)
    assert meta.succeeded
    assert new[1].fields["values"] != claims[1].fields["values"]


def test_replace_with_false_set_claim_disagrees_with_truth():
    env = _build_env([3, 4, 3, 4, 0, 4])
    solver = _solver()
    claims = _set_claims_for(env)
    new, meta = replace_with_false_claim(claims, 1, env, solver, rng=random.Random(7))
    assert meta.kind == "replace_with_false"
    assert meta.succeeded
    assert verify_claim(new[1], env, solver) is False


def test_apply_intervention_dispatch_for_set_claims():
    env = _build_env([3, 4, 3, 4, 0, 4])
    solver = _solver()
    claims = _set_claims_for(env)
    for kind in ("delete", "change_column", "replace_with_false"):
        new, meta = apply_intervention(
            kind, claims, 1, env, solver, rng=random.Random(0)
        )
        assert isinstance(new, list)
        assert meta.kind == kind


# ----------------------------------------------------------------- pipeline --


def test_pipeline_serializes_tactical_object_in_prefix():
    env = _build_env([3, 4, 3, 4, 0, 4])
    solver = _solver()
    claims = _set_claims_for(env)
    parsed = ParsedResponse(raw="", valid_json=True, claims=claims, chosen_move=4)

    seen_prefixes = []

    def stub_sample(messages, n):
        user = next(m["content"] for m in messages if m["role"] == "user")
        seen_prefixes.append(user)
        return ['{"chosen_move": 4}'] * n

    result = evaluate_response_causality(
        env,
        parsed,
        stub_sample,
        solver,
        n_resamples=4,
        threshold=0.25,
        rng=random.Random(0),
    )
    assert result.chosen_move == 4
    # At least one prefix must contain the tactical_claims object.
    assert any('"tactical_claims"' in p for p in seen_prefixes)
    assert all('"claims":' not in p for p in seen_prefixes)
