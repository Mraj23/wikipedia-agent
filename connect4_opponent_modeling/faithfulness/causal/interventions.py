"""Claim-list interventions for the causal-influence pipeline.

Three interventions per claim:
    delete                  — drop the claim entirely.
    change_column           — swap the column field with a different legal column.
    replace_with_false_claim — replace the claim with a same-typed claim whose
                               verifier returns False (sample-and-check).

All interventions are pure: they take a Claim list, an index, and an env,
and return a NEW list. Originals are never mutated.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.claims import COLUMN_FIELDS, Claim, ClaimType
from faithfulness.verifier.claim_verifier import verify_claim

INTERVENTION_KINDS = ("delete", "change_column", "replace_with_false")
MAX_REPLACE_RETRIES = 32


@dataclass
class InterventionMeta:
    kind: str
    target_index: int
    original: Optional[Claim]
    replacement: Optional[Claim]
    succeeded: bool


def delete_claim(
    claims: List[Claim], idx: int
) -> Tuple[List[Claim], InterventionMeta]:
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
    rng: Optional[random.Random] = None,
) -> Tuple[List[Claim], InterventionMeta]:
    rng = rng or random.Random()
    new_claims = [copy.deepcopy(c) for c in claims]
    claim = new_claims[idx]
    field_names = COLUMN_FIELDS.get(claim.type, ())
    if not field_names:
        return new_claims, InterventionMeta(
            kind="change_column",
            target_index=idx,
            original=claims[idx],
            replacement=None,
            succeeded=False,
        )
    # Pick the first column-bearing field.
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

    # Sample-and-check: jitter all column-bearing fields uniformly until the
    # verifier returns False. For OPTIMAL_MOVE we can be smarter (any non-best
    # legal column is False), but the generic loop keeps the code uniform.
    for _ in range(max_retries):
        candidate = copy.deepcopy(target)
        for fname in field_names:
            candidate.fields[fname] = rng.randrange(0, 7)
        # For MOVE_ALLOWS_OPPONENT_WIN, the move must be legal for the verifier
        # to return False rather than None; nudge into legal range.
        if candidate.type is ClaimType.MOVE_ALLOWS_OPPONENT_WIN:
            candidate.fields["move"] = rng.choice(legal)
        if candidate.type in (
            ClaimType.SELF_IMMEDIATE_WIN,
            ClaimType.LEGAL_MOVE,
            ClaimType.OPTIMAL_MOVE,
            ClaimType.OPPONENT_IMMEDIATE_WIN,
        ):
            # For LEGAL_MOVE we need a column that is NOT legal to make it
            # False; bias toward illegal columns when possible.
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

    # Couldn't find a False replacement (rare; can happen if every column
    # makes the claim true or unverifiable). Return unchanged with succeeded=False.
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
        return change_column(claims, idx, legal, rng=rng)
    if kind == "replace_with_false":
        return replace_with_false_claim(claims, idx, env, solver, rng=rng)
    raise ValueError(f"Unknown intervention kind: {kind}")
