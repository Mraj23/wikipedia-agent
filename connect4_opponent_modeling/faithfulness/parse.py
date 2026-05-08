"""Parse model output (JSON with atomic claims) into structured form.

Accepts:
    {"claims": [{"id": "c1", "type": "self_immediate_win", "column": 3}, ...],
     "rationale": "free-text reasoning",
     "chosen_move": 5}

Tolerates:
- Markdown code fences (```json ... ``` or ``` ... ```)
- Extra fields (ignored)
- Whitespace / surrounding prose (best-effort outermost-brace recovery)

Drops malformed claim entries (missing required fields, unknown type) but
keeps the rest. Records counts so reward shaping can penalize parse errors.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from faithfulness.claims import REQUIRED_FIELDS, Claim, ClaimType

_VALID_TYPES = {t.value for t in ClaimType}


@dataclass
class ParsedResponse:
    raw: str
    valid_json: bool = False
    claims: List[Claim] = field(default_factory=list)
    rationale: str = ""
    chosen_move: Optional[int] = None
    parse_error: Optional[str] = None
    dropped_claims: int = 0


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        # Remove leading ```lang and trailing ```
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        if s.endswith("```"):
            s = s[: -len("```")]
        return s.strip()
    return s


def _extract_outermost_object(text: str) -> Optional[str]:
    """Return the substring from the first '{' to the matching closing '}'.

    Tolerates leading/trailing prose. Naive brace matching (does not parse
    string literals); good enough for well-formed JSON dumped by the model.
    """
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


def parse_structured_response(text: str) -> ParsedResponse:
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

    rationale = data.get("rationale", "")
    if isinstance(rationale, str):
        out.rationale = rationale
    elif rationale is not None:
        out.rationale = str(rationale)

    # chosen_move
    out.chosen_move = _coerce_int(data.get("chosen_move"))

    # claims
    raw_claims = data.get("claims", [])
    if not isinstance(raw_claims, list):
        out.parse_error = "claims_not_list"
        return out

    for idx, entry in enumerate(raw_claims):
        if not isinstance(entry, dict):
            out.dropped_claims += 1
            continue
        ctype = entry.get("type")
        if ctype not in _VALID_TYPES:
            out.dropped_claims += 1
            continue
        ctype_enum = ClaimType(ctype)

        cid = entry.get("id") or f"c{idx + 1}"
        if not isinstance(cid, str):
            cid = str(cid)

        fields = {k: v for k, v in entry.items() if k not in ("id", "type")}

        # Coerce columns/moves to ints; drop on failure
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

    return out
