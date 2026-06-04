"""Parse + grade model outputs.

Handles two answer types, auto-detected from the gold string:
  - multiple-choice letter (BBH): gold looks like "(C)" or a bare "C"
  - numeric (GSM8K):              gold is a number, e.g. "18"

Robustness notes (these matter because the whole experiment measures terse
"caveman" outputs, which are exactly the outputs a naive grader mishandles):
  - The reasoning/answer split is reported via `parse_success`. When the
    model drops the "Answer:" label we fall back conservatively and flag it,
    so downstream analysis can exclude unparsed rows from the *reasoning*-token
    metric. The primary length axis is `total_output_tokens` (see eval.py),
    which never depends on parsing and is the quantity Caveman actually
    targets ("make mouth smaller").
  - Letter extraction prefers explicit "(X)" / "answer is X" cues and only
    falls back to a bare standalone letter as a last resort, and never to a
    lowercase article like "a" unless nothing else is present.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REASONING_RE = re.compile(
    r"^\s*reasoning\s*:\s*(.*?)(?=^\s*answer\s*:|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
ANSWER_LINE_RE = re.compile(
    r"^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

PAREN_LETTER_RE = re.compile(r"\(([a-zA-Z])\)")
ANSWER_IS_LETTER_RE = re.compile(
    r"\b(?:answer|option|choice)\s*(?:is|=|:)?\s*\(?([a-zA-Z])\)?\b",
    re.IGNORECASE,
)
LEADING_LETTER_RE = re.compile(r"^\s*\(?([a-zA-Z])\)?\b")
UPPER_LETTER_RE = re.compile(r"\b([A-Z])\b")
ANY_LETTER_RE = re.compile(r"\b([a-zA-Z])\b")

NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
GOLD_LETTER_RE = re.compile(r"^\s*\(?([a-zA-Z])\)?\s*$")


def parse_output(raw: str):
    """Return (reasoning_text, answer_text, parse_success).

    parse_success is True only when an explicit "Answer:" (or "Final
    answer:") line is found; callers should treat reasoning-token counts
    from unparsed rows as unreliable.
    """
    raw = raw or ""
    answer_matches = ANSWER_LINE_RE.findall(raw)
    if answer_matches:
        answer = answer_matches[-1].strip()
        reasoning_match = REASONING_RE.search(raw)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # Everything before the (last) answer line is reasoning.
            idx = raw.lower().rfind("answer:")
            reasoning = raw[:idx].strip() if idx > 0 else ""
            reasoning = re.sub(r"^\s*reasoning\s*:\s*", "", reasoning, flags=re.IGNORECASE)
        return reasoning, answer, True

    lines = [l for l in raw.strip().splitlines() if l.strip()]
    if not lines:
        return "", "", False
    return "\n".join(lines[:-1]).strip(), lines[-1].strip(), False


def extract_choice(text: str) -> str:
    """Normalize a multiple-choice answer to a single uppercase letter ("" if none)."""
    if not text:
        return ""
    text = text.strip()

    m = ANSWER_IS_LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = PAREN_LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = LEADING_LETTER_RE.match(text)
    if m:
        return m.group(1).upper()
    # Prefer a standalone UPPERCASE letter (options are uppercase) before
    # risking a lowercase article like "a".
    upper = UPPER_LETTER_RE.findall(text)
    if upper:
        return upper[-1].upper()
    any_letters = ANY_LETTER_RE.findall(text)
    if any_letters:
        return any_letters[-1].upper()
    return ""


def extract_number(text: str) -> str:
    """Return the last number in `text` as a normalized string ("" if none)."""
    if not text:
        return ""
    matches = NUMBER_RE.findall(text)
    if not matches:
        return ""
    raw = matches[-1].replace(",", "").replace("$", "")
    try:
        f = float(raw)
    except ValueError:
        return ""
    return str(int(f)) if f.is_integer() else repr(f)


def _is_letter_gold(gold: str) -> bool:
    return bool(GOLD_LETTER_RE.match((gold or "").strip()))


def grade_one(raw_output: str, gold: str, task: str | None = None):
    reasoning, answer, parse_ok = parse_output(raw_output)
    letter_mode = _is_letter_gold(gold)
    if letter_mode:
        gold_norm = extract_choice(gold)
        ans_norm = extract_choice(answer)
    else:
        gold_norm = extract_number(gold)
        ans_norm = extract_number(answer)
    return {
        "reasoning_text": reasoning,
        "answer_text": answer,
        "parse_success": parse_ok,
        "answer_type": "letter" if letter_mode else "numeric",
        "normalized_gold": gold_norm,
        "normalized_answer": ans_norm,
        "correct": gold_norm == ans_norm and bool(ans_norm),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            graded = grade_one(row["raw_output"], row["gold"], row.get("task"))
            row.update(graded)
            fout.write(json.dumps(row) + "\n")
            n += 1
    print(f"graded {n} -> {out_path}")


if __name__ == "__main__":
    main()
