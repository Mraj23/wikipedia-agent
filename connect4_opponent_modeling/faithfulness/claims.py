"""Atomic, machine-checkable claim types for the faithfulness experiment.

Each claim asserts a single tactical fact that the verifier oracle can label
True / False / None (None = malformed or unverifiable). Restricting the
output schema to a small enum of typed claims is what makes the truth axis
of the 2x2 well-defined.

OPPONENT_IMMEDIATE_WIN semantics are locked here so the verifier and the
prompt agree: the claim asserts that *right now*, before the model's move,
if it were the opponent's turn the opponent could win by playing `column`.
This is the "threat the model is reading" interpretation. It does NOT
condition on any move the model is considering.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ClaimType(str, Enum):
    SELF_IMMEDIATE_WIN = "self_immediate_win"
    OPPONENT_IMMEDIATE_WIN = "opponent_immediate_win"
    MOVE_ALLOWS_OPPONENT_WIN = "move_allows_opponent_win"
    LEGAL_MOVE = "legal_move"
    OPTIMAL_MOVE = "optimal_move"


# Required field names per claim type. The parser drops claims whose fields
# don't match this contract; the verifier assumes the contract holds.
REQUIRED_FIELDS: Dict[ClaimType, frozenset] = {
    ClaimType.SELF_IMMEDIATE_WIN: frozenset({"column"}),
    ClaimType.OPPONENT_IMMEDIATE_WIN: frozenset({"column"}),
    ClaimType.MOVE_ALLOWS_OPPONENT_WIN: frozenset({"move", "opponent_reply"}),
    ClaimType.LEGAL_MOVE: frozenset({"column"}),
    ClaimType.OPTIMAL_MOVE: frozenset({"column"}),
}

# Field names that carry a column number. Used by interventions to know what
# to swap when applying change_column / replace_with_false_claim.
COLUMN_FIELDS: Dict[ClaimType, tuple] = {
    ClaimType.SELF_IMMEDIATE_WIN: ("column",),
    ClaimType.OPPONENT_IMMEDIATE_WIN: ("column",),
    ClaimType.MOVE_ALLOWS_OPPONENT_WIN: ("move", "opponent_reply"),
    ClaimType.LEGAL_MOVE: ("column",),
    ClaimType.OPTIMAL_MOVE: ("column",),
}


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
