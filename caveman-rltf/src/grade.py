"""Parse + grade model outputs from a thinking model (Qwen3.x hybrid).

Output shape (the prompt already opened "<think>"):
    <thinking ...>\n</think>\n\n<answer region, ending in "Answer: X">

We split on "</think>" into:
    thinking_text   -- the reasoning "brain" tokens (primary compression target)
    post_think_text -- everything after </think> (the visible "mouth")
and extract the final answer from post_think_text.

Answer type is auto-detected from the gold string:
  - multiple-choice letter (BBH): gold looks like "(C)" or a bare "C"
  - numeric (GSM8K):              gold is a number, e.g. "18"

Robustness: letter extraction prefers explicit "(X)" / "answer is X" cues and
only falls back to a bare standalone letter as a last resort (uppercase first),
so terse outputs are not mis-graded by grabbing a stray lowercase article.
`parse_success` is True only when an explicit "Answer:" line is found.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


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


def split_thinking(raw: str):
    """Return (thinking_text, post_think_text)."""
    raw = raw or ""
    if "</think>" in raw:
        thinking, rest = raw.split("</think>", 1)
        thinking = re.sub(r"^\s*<think>\s*", "", thinking, flags=re.IGNORECASE)
        return thinking.strip(), rest.strip()
    # No think block (e.g. thinking disabled) — treat all as post-think.
    return "", raw.strip()


def extract_answer_text(post_think: str):
    """Return (answer_text, parse_success) from the post-</think> region."""
    matches = ANSWER_LINE_RE.findall(post_think)
    if matches:
        return matches[-1].strip(), True
    lines = [l for l in post_think.strip().splitlines() if l.strip()]
    if not lines:
        return "", False
    return lines[-1].strip(), False


def extract_choice(text: str) -> str:
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
    upper = UPPER_LETTER_RE.findall(text)
    if upper:
        return upper[-1].upper()
    any_letters = ANY_LETTER_RE.findall(text)
    if any_letters:
        return any_letters[-1].upper()
    return ""


def extract_number(text: str) -> str:
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
    thinking_text, post_think_text = split_thinking(raw_output)
    answer_text, parse_ok = extract_answer_text(post_think_text)
    letter_mode = _is_letter_gold(gold)
    if letter_mode:
        gold_norm = extract_choice(gold)
        ans_norm = extract_choice(answer_text)
    else:
        gold_norm = extract_number(gold)
        ans_norm = extract_number(answer_text)
    return {
        "thinking_text": thinking_text,
        "post_think_text": post_think_text,
        "answer_text": answer_text,
        "parse_success": parse_ok,
        "has_thinking": bool(thinking_text),
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
