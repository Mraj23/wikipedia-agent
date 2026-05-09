"""Claim-list interventions for the causal-influence pipeline.

Three interventions per claim:
    delete                  — drop the claim entirely (legacy) or empty the
                              targeted tactical field (set claims).
    change_column           — swap a column field with a different legal column.
                              For set claims, alters one element of the set.
    replace_with_false_claim — replace the claim with a same-typed claim whose
                              verifier returns False. For set claims, mutates
                              the set so it no longer matches ground truth.

All interventions are pure: they take a Claim list, an index, and an env,
and return a NEW list. Originals are never mutated.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.claims import (
    CLAIM_TYPE_TO_TACTICAL_FIELD,
    COLUMN_FIELDS,
    Claim,
    ClaimType,
)
from faithfulness.verifier.claim_verifier import (
    ground_truth_opponent_immediate_win_columns,
    ground_truth_self_double_threat_moves,
    ground_truth_self_immediate_win_columns,
    ground_truth_self_single_threat_moves,
    ground_truth_unsafe_moves,
    verify_claim,
)

INTERVENTION_KINDS = ("delete", "change_column", "replace_with_false")
MAX_REPLACE_RETRIES = 32


@dataclass
class InterventionMeta:
    kind: str
    target_index: int
    original: Optional[Claim]
    replacement: Optional[Claim]
    succeeded: bool


# --- helpers for set-valued claims -----------------------------------------


def _is_set_claim(claim: Claim) -> bool:
    return claim.type in CLAIM_TYPE_TO_TACTICAL_FIELD


def _empty_set_claim(claim: Claim) -> Claim:
    new = copy.deepcopy(claim)
    if claim.type is ClaimType.SET_UNSAFE_MOVES:
        new.fields["entries"] = []
    else:
        new.fields["values"] = []
    return new


def _ground_truth_set(claim_type: ClaimType, env: ConnectFourEnv):
    if claim_type is ClaimType.SET_SELF_IMMEDIATE_WIN:
        return set(ground_truth_self_immediate_win_columns(env))
    if claim_type is ClaimType.SET_OPPONENT_IMMEDIATE_WIN:
        return set(ground_truth_opponent_immediate_win_columns(env))
    if claim_type is ClaimType.SET_SELF_DOUBLE_THREAT_MOVES:
        return set(ground_truth_self_double_threat_moves(env))
    if claim_type is ClaimType.SET_SELF_SINGLE_THREAT_MOVES:
        return set(ground_truth_self_single_threat_moves(env))
    return None


def _change_column_set(
    claim: Claim, env: ConnectFourEnv, rng: random.Random
) -> Optional[Claim]:
    """Alter one existing column/reply to a different legal column.

    Deterministic preference (per plan): pick the existing element whose
    swap produces the largest |new - orig|; break ties by smaller original
    column index. If there is no element to mutate or no alternative
    column available, returns None (caller marks skipped).
    """
    new = copy.deepcopy(claim)
    legal = sorted(env.legal_moves())
    if claim.type is ClaimType.SET_UNSAFE_MOVES:
        entries = new.fields.get("entries", [])
        if not entries:
            return None
        # Try mutating the first entry's first reply, deterministically.
        for entry in entries:
            replies = entry.get("opponent_replies", [])
            if not replies:
                continue
            orig = replies[0]
            alternatives = [c for c in range(7) if c != orig]
            if not alternatives:
                continue
            alternatives.sort(key=lambda c: (-abs(c - orig), c))
            entry["opponent_replies"] = sorted({alternatives[0]} | set(replies[1:]))
            if len(entry["opponent_replies"]) == 0:
                continue
            return new
        return None

    values = new.fields.get("values", [])
    if not values:
        # No element to swap; intervention skipped.
        return None
    orig = values[0]
    pool = [c for c in range(7) if c != orig]
    if not pool:
        return None
    pool.sort(key=lambda c: (-abs(c - orig), c))
    new_values = sorted({pool[0]} | set(values[1:]))
    new.fields["values"] = new_values
    return new


def _replace_with_false_set(
    claim: Claim, env: ConnectFourEnv
) -> Optional[Claim]:
    """Deterministically alter the claim's set so it disagrees with truth.

    Strategy: compute ground truth; if claim equals it, flip one element.
    If claim already disagrees, just return claim itself (already false). If
    we cannot construct a disagreeing set (e.g. ground truth is the full
    universe and claim already matches it), return None.
    """
    new = copy.deepcopy(claim)
    if claim.type is ClaimType.SET_UNSAFE_MOVES:
        gt = ground_truth_unsafe_moves(env)
        gt_map = {e["move"]: tuple(sorted(e["opponent_replies"])) for e in gt}
        entries = new.fields.get("entries", [])
        cur_map = {
            e["move"]: tuple(sorted(e["opponent_replies"])) for e in entries
        }
        if cur_map != gt_map:
            return new  # already false
        # Mutate: if there is a true entry, drop one of its replies (or the
        # whole entry); if the gt set is empty, we cannot fabricate a False
        # one without knowing legality nuances — invent a fake unsafe entry
        # using a legal move with a non-winning opponent reply, then verify.
        if entries:
            # Drop the first reply of the first entry; if that empties the
            # replies, drop the entry entirely.
            entry = copy.deepcopy(entries[0])
            replies = list(entry["opponent_replies"])
            replies = replies[1:]
            if replies:
                entry["opponent_replies"] = replies
                new_entries = [entry] + [copy.deepcopy(e) for e in entries[1:]]
            else:
                new_entries = [copy.deepcopy(e) for e in entries[1:]]
            new.fields["entries"] = new_entries
            return new
        # Empty set case: we can't manufacture an unsafe entry without solver
        # interference; report skipped.
        return None

    gt = _ground_truth_set(claim.type, env)
    if gt is None:
        return None
    values = set(claim.fields.get("values", []))
    if values != gt:
        return new  # already false
    universe = set(range(7))
    # Try removing an element first, else add a non-member.
    removable = sorted(values)
    addable = sorted(universe - gt)
    if removable:
        new_values = sorted(values - {removable[0]})
    elif addable:
        new_values = sorted(values | {addable[0]})
    else:
        return None
    new.fields["values"] = new_values
    return new


# --- interventions ----------------------------------------------------------


def delete_claim(
    claims: List[Claim], idx: int
) -> Tuple[List[Claim], InterventionMeta]:
    target = claims[idx]
    if _is_set_claim(target):
        new_claims = [copy.deepcopy(c) for c in claims]
        new_claims[idx] = _empty_set_claim(target)
        return new_claims, InterventionMeta(
            kind="delete",
            target_index=idx,
            original=target,
            replacement=copy.deepcopy(new_claims[idx]),
            succeeded=True,
        )
    new_claims = [c for i, c in enumerate(claims) if i != idx]
    return new_claims, InterventionMeta(
        kind="delete",
        target_index=idx,
        original=claims[idx],
        replacement=None,
        succeeded=True,
    )


def change_column(
    claims: List[Claim],
    idx: int,
    legal_cols: List[int],
    *,
    env: Optional[ConnectFourEnv] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Claim], InterventionMeta]:
    rng = rng or random.Random()
    new_claims = [copy.deepcopy(c) for c in claims]
    claim = new_claims[idx]

    if _is_set_claim(claim):
        if env is None:
            return new_claims, InterventionMeta(
                kind="change_column",
                target_index=idx,
                original=claims[idx],
                replacement=None,
                succeeded=False,
            )
        replaced = _change_column_set(claim, env, rng)
        if replaced is None:
            return new_claims, InterventionMeta(
                kind="change_column",
                target_index=idx,
                original=claims[idx],
                replacement=None,
                succeeded=False,
            )
        new_claims[idx] = replaced
        return new_claims, InterventionMeta(
            kind="change_column",
            target_index=idx,
            original=claims[idx],
            replacement=copy.deepcopy(replaced),
            succeeded=True,
        )

    field_names = COLUMN_FIELDS.get(claim.type, ())
    if not field_names:
        return new_claims, InterventionMeta(
            kind="change_column",
            target_index=idx,
            original=claims[idx],
            replacement=None,
            succeeded=False,
        )
    fname = field_names[0]
    orig_val = claim.fields.get(fname)
    other = [c for c in legal_cols if c != orig_val]
    if not other:
        return new_claims, InterventionMeta(
            kind="change_column",
            target_index=idx,
            original=claims[idx],
            replacement=None,
            succeeded=False,
        )
    new_val = rng.choice(other)
    claim.fields[fname] = new_val
    return new_claims, InterventionMeta(
        kind="change_column",
        target_index=idx,
        original=claims[idx],
        replacement=copy.deepcopy(claim),
        succeeded=True,
    )


def replace_with_false_claim(
    claims: List[Claim],
    idx: int,
    env: ConnectFourEnv,
    solver: PonsSolver,
    *,
    rng: Optional[random.Random] = None,
    max_retries: int = MAX_REPLACE_RETRIES,
) -> Tuple[List[Claim], InterventionMeta]:
    rng = rng or random.Random()
    new_claims = [copy.deepcopy(c) for c in claims]
    target = new_claims[idx]

    if _is_set_claim(target):
        replaced = _replace_with_false_set(target, env)
        if replaced is None:
            return new_claims, InterventionMeta(
                kind="replace_with_false",
                target_index=idx,
                original=claims[idx],
                replacement=None,
                succeeded=False,
            )
        new_claims[idx] = replaced
        return new_claims, InterventionMeta(
            kind="replace_with_false",
            target_index=idx,
            original=claims[idx],
            replacement=copy.deepcopy(replaced),
            succeeded=True,
        )

    legal = env.legal_moves()
    field_names = COLUMN_FIELDS.get(target.type, ())
    if not field_names:
        return new_claims, InterventionMeta(
            kind="replace_with_false",
            target_index=idx,
            original=claims[idx],
            replacement=None,
            succeeded=False,
        )

    for _ in range(max_retries):
        candidate = copy.deepcopy(target)
        for fname in field_names:
            candidate.fields[fname] = rng.randrange(0, 7)
        if candidate.type is ClaimType.MOVE_ALLOWS_OPPONENT_WIN:
            candidate.fields["move"] = rng.choice(legal)
        if candidate.type in (
            ClaimType.SELF_IMMEDIATE_WIN,
            ClaimType.LEGAL_MOVE,
            ClaimType.OPTIMAL_MOVE,
            ClaimType.OPPONENT_IMMEDIATE_WIN,
        ):
            if candidate.type is ClaimType.LEGAL_MOVE:
                illegal = [c for c in range(7) if c not in legal]
                if illegal:
                    candidate.fields["column"] = rng.choice(illegal)
        verdict = verify_claim(candidate, env, solver)
        if verdict is False:
            new_claims[idx] = candidate
            return new_claims, InterventionMeta(
                kind="replace_with_false",
                target_index=idx,
                original=claims[idx],
                replacement=copy.deepcopy(candidate),
                succeeded=True,
            )

    return new_claims, InterventionMeta(
        kind="replace_with_false",
        target_index=idx,
        original=claims[idx],
        replacement=None,
        succeeded=False,
    )


def apply_intervention(
    kind: str,
    claims: List[Claim],
    idx: int,
    env: ConnectFourEnv,
    solver: PonsSolver,
    *,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Claim], InterventionMeta]:
    legal = env.legal_moves()
    if kind == "delete":
        return delete_claim(claims, idx)
    if kind == "change_column":
        return change_column(claims, idx, legal, env=env, rng=rng)
    if kind == "replace_with_false":
        return replace_with_false_claim(claims, idx, env, solver, rng=rng)
    raise ValueError(f"Unknown intervention kind: {kind}")
