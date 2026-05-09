"""Atomic, machine-checkable claim types for the faithfulness experiment.

Each claim asserts a single tactical fact that the verifier oracle can label
True / False / None (None = malformed or unverifiable). Restricting the
output schema to a small enum of typed claims is what makes the truth axis
of the 2x2 well-defined.

Two families of claim types coexist:

* The legacy `claims_rationale` family — one claim per tactical observation
  (SELF_IMMEDIATE_WIN, OPPONENT_IMMEDIATE_WIN, MOVE_ALLOWS_OPPONENT_WIN,
  LEGAL_MOVE, OPTIMAL_MOVE). LEGAL_MOVE and OPTIMAL_MOVE are kept here only
  for backwards-compatibility with archived eval artifacts; the
  `tactical_claims` schema does not allow them, because they leak the answer
  into the trace.

* The `tactical_claims` family (SET_*) — one claim per tactical FIELD; each
  claim carries an entire set (or list of move/replies objects). The
  verifier scores a SET_* claim by exact set equality with the ground-truth
  set. This is what the new training schema uses.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ClaimType(str, Enum):
    # Legacy, single-fact claims used by `claims_rationale`.
    SELF_IMMEDIATE_WIN = "self_immediate_win"
    OPPONENT_IMMEDIATE_WIN = "opponent_immediate_win"
    MOVE_ALLOWS_OPPONENT_WIN = "move_allows_opponent_win"
    LEGAL_MOVE = "legal_move"  # legacy-only; rejected under tactical_claims
    OPTIMAL_MOVE = "optimal_move"  # legacy-only; rejected under tactical_claims

    # Tactical-set claims used by `tactical_claims`. Each carries a list
    # payload under `fields["values"]` (or `fields["entries"]` for
    # SET_UNSAFE_MOVES, whose entries are objects with their own structure).
    SET_SELF_IMMEDIATE_WIN = "set_self_immediate_win_columns"
    SET_OPPONENT_IMMEDIATE_WIN = "set_opponent_immediate_win_columns"
    SET_UNSAFE_MOVES = "set_unsafe_moves"
    SET_SELF_DOUBLE_THREAT_MOVES = "set_self_double_threat_moves"
    SET_SELF_SINGLE_THREAT_MOVES = "set_self_single_threat_moves"


# Required field names per claim type. The parser drops claims whose fields
# don't match this contract; the verifier assumes the contract holds.
REQUIRED_FIELDS: Dict[ClaimType, frozenset] = {
    ClaimType.SELF_IMMEDIATE_WIN: frozenset({"column"}),
    ClaimType.OPPONENT_IMMEDIATE_WIN: frozenset({"column"}),
    ClaimType.MOVE_ALLOWS_OPPONENT_WIN: frozenset({"move", "opponent_reply"}),
    ClaimType.LEGAL_MOVE: frozenset({"column"}),
    ClaimType.OPTIMAL_MOVE: frozenset({"column"}),
    ClaimType.SET_SELF_IMMEDIATE_WIN: frozenset({"values"}),
    ClaimType.SET_OPPONENT_IMMEDIATE_WIN: frozenset({"values"}),
    ClaimType.SET_UNSAFE_MOVES: frozenset({"entries"}),
    ClaimType.SET_SELF_DOUBLE_THREAT_MOVES: frozenset({"values"}),
    ClaimType.SET_SELF_SINGLE_THREAT_MOVES: frozenset({"values"}),
}

# Field names that carry a column number for the legacy interventions.
COLUMN_FIELDS: Dict[ClaimType, tuple] = {
    ClaimType.SELF_IMMEDIATE_WIN: ("column",),
    ClaimType.OPPONENT_IMMEDIATE_WIN: ("column",),
    ClaimType.MOVE_ALLOWS_OPPONENT_WIN: ("move", "opponent_reply"),
    ClaimType.LEGAL_MOVE: ("column",),
    ClaimType.OPTIMAL_MOVE: ("column",),
    # SET_* claim columns live inside their list payload — interventions
    # handle them via dedicated set-mutation logic, not COLUMN_FIELDS.
    ClaimType.SET_SELF_IMMEDIATE_WIN: (),
    ClaimType.SET_OPPONENT_IMMEDIATE_WIN: (),
    ClaimType.SET_UNSAFE_MOVES: (),
    ClaimType.SET_SELF_DOUBLE_THREAT_MOVES: (),
    ClaimType.SET_SELF_SINGLE_THREAT_MOVES: (),
}

# Tactical-claims schema — fixed JSON-key layout. The parser enforces this
# exactly when condition == "tactical_claims".
TACTICAL_FIELD_TO_CLAIM_TYPE: Dict[str, ClaimType] = {
    "self_immediate_win_columns": ClaimType.SET_SELF_IMMEDIATE_WIN,
    "opponent_immediate_win_columns": ClaimType.SET_OPPONENT_IMMEDIATE_WIN,
    "unsafe_moves": ClaimType.SET_UNSAFE_MOVES,
    "self_double_threat_moves": ClaimType.SET_SELF_DOUBLE_THREAT_MOVES,
    "self_single_threat_moves": ClaimType.SET_SELF_SINGLE_THREAT_MOVES,
}

CLAIM_TYPE_TO_TACTICAL_FIELD: Dict[ClaimType, str] = {
    v: k for k, v in TACTICAL_FIELD_TO_CLAIM_TYPE.items()
}

TACTICAL_CLAIM_TYPES = frozenset(TACTICAL_FIELD_TO_CLAIM_TYPE.values())
LEGACY_ANSWER_LEAK_TYPES = frozenset(
    {ClaimType.LEGAL_MOVE, ClaimType.OPTIMAL_MOVE}
)


@dataclass
class Claim:
    id: str
    type: ClaimType
    fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {"id": self.id, "type": self.type.value}
        out.update(self.fields)
        return out

    def has_required_fields(self) -> bool:
        required = REQUIRED_FIELDS[self.type]
        return required.issubset(self.fields.keys())
