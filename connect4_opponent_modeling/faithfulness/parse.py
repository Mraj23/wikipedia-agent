"""Parse model output (JSON) into structured form.

Two parsing modes:

* `claims_rationale` (default, legacy):
    {"claims": [...], "rationale": "...", "chosen_move": 5}

* `tactical_claims` (strict, new):
    {"tactical_claims": {"self_immediate_win_columns": [...],
                         "opponent_immediate_win_columns": [...],
                         "unsafe_moves": [{"move": M, "opponent_replies": [...]}, ...],
                         "self_double_threat_moves": [...],
                         "self_single_threat_moves": [...]},
     "chosen_move": <int 0-6>}

For tactical_claims, the parser is strict: any extra/missing keys, duplicates,
out-of-range columns, empty opponent_replies, or legacy answer-leak fields
flip `schema_valid` to False. The reward calculator treats schema-invalid
output the same as invalid JSON for the purposes of training reward.

Tolerates (both modes): markdown code fences, surrounding prose with
outermost-brace recovery, and string-encoded ints (only in legacy mode —
strict mode insists on integer JSON values).
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from faithfulness.claims import (
    CLAIM_TYPE_TO_TACTICAL_FIELD,
    LEGACY_ANSWER_LEAK_TYPES,
    REQUIRED_FIELDS,
    TACTICAL_FIELD_TO_CLAIM_TYPE,
    Claim,
    ClaimType,
)

# Legacy parsing accepts the historical free-form claim list; we deliberately
# include the answer-leak types here so old artifacts round-trip cleanly.
_VALID_LEGACY_TYPES = {
    ClaimType.SELF_IMMEDIATE_WIN.value,
    ClaimType.OPPONENT_IMMEDIATE_WIN.value,
    ClaimType.MOVE_ALLOWS_OPPONENT_WIN.value,
    ClaimType.LEGAL_MOVE.value,
    ClaimType.OPTIMAL_MOVE.value,
}


@dataclass
class ParsedResponse:
    raw: str
    valid_json: bool = False
    claims: List[Claim] = field(default_factory=list)
    rationale: str = ""
    chosen_move: Optional[int] = None
    parse_error: Optional[str] = None
    dropped_claims: int = 0
    # `schema_valid` is True iff the response satisfies the strict schema for
    # the requested condition. For legacy claims_rationale this defaults to
    # `valid_json` (no strict-schema policy is enforced). For tactical_claims
    # it requires the exact 5-field tactical_claims object plus chosen_move
    # and nothing else; see `_parse_tactical_claims` below.
    schema_valid: bool = False


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        if s.endswith("```"):
            s = s[: -len("```")]
        return s.strip()
    return s


def _extract_outermost_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _coerce_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _strict_int_in_range(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if not isinstance(value, int):
        return None
    if value < 0 or value > 6:
        return None
    return value


def parse_structured_response(
    text: str,
    condition: str = "claims_rationale",
) -> ParsedResponse:
    """Parse a model response under the given training condition.

    `condition` mirrors the prompt-side `PromptCondition` literal. Only
    `tactical_claims` triggers the strict tactical schema; everything else
    behaves like the legacy free-form claim list (which is also used for
    `move_only`, where claims are simply absent).
    """
    out = ParsedResponse(raw=text)
    cleaned = _strip_code_fence(text)
    candidate = _extract_outermost_object(cleaned) or cleaned

    try:
        data = json.loads(candidate)
    except Exception as exc:  # noqa: BLE001 — record any parse error
        out.parse_error = f"json_decode: {exc}"
        return out

    if not isinstance(data, dict):
        out.parse_error = "top_level_not_object"
        return out

    out.valid_json = True

    if condition == "tactical_claims":
        return _parse_tactical_claims(data, out)
    return _parse_legacy(data, out)


def _parse_legacy(data: dict, out: ParsedResponse) -> ParsedResponse:
    rationale = data.get("rationale", "")
    if isinstance(rationale, str):
        out.rationale = rationale
    elif rationale is not None:
        out.rationale = str(rationale)

    out.chosen_move = _coerce_int(data.get("chosen_move"))

    raw_claims = data.get("claims", [])
    if not isinstance(raw_claims, list):
        out.parse_error = "claims_not_list"
        out.schema_valid = False
        return out

    for idx, entry in enumerate(raw_claims):
        if not isinstance(entry, dict):
            out.dropped_claims += 1
            continue
        ctype = entry.get("type")
        if ctype not in _VALID_LEGACY_TYPES:
            out.dropped_claims += 1
            continue
        ctype_enum = ClaimType(ctype)

        cid = entry.get("id") or f"c{idx + 1}"
        if not isinstance(cid, str):
            cid = str(cid)

        fields = {k: v for k, v in entry.items() if k not in ("id", "type")}

        coerced = {}
        coerce_ok = True
        for k, v in fields.items():
            if k in ("column", "move", "opponent_reply"):
                cv = _coerce_int(v)
                if cv is None:
                    coerce_ok = False
                    break
                coerced[k] = cv
            else:
                coerced[k] = v
        if not coerce_ok:
            out.dropped_claims += 1
            continue

        claim = Claim(id=cid, type=ctype_enum, fields=coerced)
        if not claim.has_required_fields():
            out.dropped_claims += 1
            continue
        out.claims.append(claim)

    # Legacy mode: schema_valid mirrors valid_json (no strict policy).
    out.schema_valid = True
    return out


_TACTICAL_TOPLEVEL_KEYS = frozenset({"tactical_claims", "chosen_move"})
_TACTICAL_INNER_KEYS = frozenset(TACTICAL_FIELD_TO_CLAIM_TYPE.keys())
_UNSAFE_MOVE_KEYS = frozenset({"move", "opponent_replies"})


def _parse_tactical_claims(data: dict, out: ParsedResponse) -> ParsedResponse:
    """Strictly parse the tactical_claims schema.

    On any deviation from the exact schema we record a parse_error reason and
    leave `schema_valid=False`. We still try to populate `chosen_move` when
    it's present and well-formed so downstream code can show what the model
    answered, but the reward calculator treats schema_invalid the same as
    invalid_json regardless of chosen_move.
    """
    keys = set(data.keys())
    if keys != _TACTICAL_TOPLEVEL_KEYS:
        out.parse_error = "tactical_top_level_keys"
        # Try to surface chosen_move for diagnostics even if extras present.
        cm = data.get("chosen_move")
        if isinstance(cm, int) and not isinstance(cm, bool):
            out.chosen_move = cm
        return out

    chosen_move = data.get("chosen_move")
    if not isinstance(chosen_move, int) or isinstance(chosen_move, bool):
        out.parse_error = "tactical_chosen_move_not_int"
        return out
    out.chosen_move = chosen_move

    tc = data.get("tactical_claims")
    if not isinstance(tc, dict):
        out.parse_error = "tactical_claims_not_object"
        return out
    if set(tc.keys()) != _TACTICAL_INNER_KEYS:
        out.parse_error = "tactical_inner_keys"
        return out

    claims: List[Claim] = []

    # Plain int-list fields.
    for fname in (
        "self_immediate_win_columns",
        "opponent_immediate_win_columns",
        "self_double_threat_moves",
        "self_single_threat_moves",
    ):
        values = tc[fname]
        if not isinstance(values, list):
            out.parse_error = f"{fname}_not_list"
            return out
        normalized: List[int] = []
        seen = set()
        for v in values:
            iv = _strict_int_in_range(v)
            if iv is None:
                out.parse_error = f"{fname}_bad_value"
                return out
            if iv in seen:
                out.parse_error = f"{fname}_duplicate"
                return out
            seen.add(iv)
            normalized.append(iv)
        normalized.sort()
        ctype = TACTICAL_FIELD_TO_CLAIM_TYPE[fname]
        claims.append(Claim(id=fname, type=ctype, fields={"values": normalized}))

    # Disjointness of single/double threat sets is part of schema validity.
    double = next(
        c for c in claims if c.type is ClaimType.SET_SELF_DOUBLE_THREAT_MOVES
    )
    single = next(
        c for c in claims if c.type is ClaimType.SET_SELF_SINGLE_THREAT_MOVES
    )
    if set(double.fields["values"]) & set(single.fields["values"]):
        out.parse_error = "single_double_threat_overlap"
        return out

    # unsafe_moves: list of objects.
    unsafe_raw = tc["unsafe_moves"]
    if not isinstance(unsafe_raw, list):
        out.parse_error = "unsafe_moves_not_list"
        return out
    entries: List[dict] = []
    moves_seen = set()
    for entry in unsafe_raw:
        if not isinstance(entry, dict) or set(entry.keys()) != _UNSAFE_MOVE_KEYS:
            out.parse_error = "unsafe_moves_bad_entry"
            return out
        mv = _strict_int_in_range(entry["move"])
        if mv is None:
            out.parse_error = "unsafe_moves_bad_move"
            return out
        if mv in moves_seen:
            out.parse_error = "unsafe_moves_duplicate_move"
            return out
        moves_seen.add(mv)
        replies_raw = entry["opponent_replies"]
        if not isinstance(replies_raw, list) or len(replies_raw) == 0:
            out.parse_error = "unsafe_moves_empty_replies"
            return out
        replies: List[int] = []
        seen = set()
        for r in replies_raw:
            iv = _strict_int_in_range(r)
            if iv is None:
                out.parse_error = "unsafe_moves_bad_reply"
                return out
            if iv in seen:
                out.parse_error = "unsafe_moves_duplicate_reply"
                return out
            seen.add(iv)
            replies.append(iv)
        replies.sort()
        entries.append({"move": mv, "opponent_replies": replies})
    entries.sort(key=lambda e: e["move"])
    claims.append(
        Claim(
            id="unsafe_moves",
            type=ClaimType.SET_UNSAFE_MOVES,
            fields={"entries": entries},
        )
    )

    out.claims = claims
    out.schema_valid = True
    return out
