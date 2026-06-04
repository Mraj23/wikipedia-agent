"""Exact-match grading.

Auto-detects the answer type from the gold string:
  - multiple-choice letter (BBH): gold looks like "(C)" or a bare "C"
  - numeric (GSM8K):              gold is a number, e.g. "18"

Letter extraction prefers explicit "(X)" / "answer is X" cues and only
falls back to a bare standalone letter as a last resort (uppercase first),
so terse outputs are not mis-graded by grabbing a stray lowercase article.
"""

import argparse
import json
import re
from pathlib import Path


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


def grade_row(row):
    gold = row["gold"]
    if _is_letter_gold(gold):
        gold_norm = extract_choice(gold)
        ans_norm = extract_choice(row["answer_text"])
    else:
        gold_norm = extract_number(gold)
        ans_norm = extract_number(row["answer_text"])
    return (gold_norm == ans_norm and bool(ans_norm)), gold_norm, ans_norm


def process_file(in_path: Path, fout):
    n = 0
    with in_path.open() as f:
        for line in f:
            row = json.loads(line)
            correct, gold_norm, ans_norm = grade_row(row)
            row["correct"] = correct
            row["normalized_gold"] = gold_norm
            row["normalized_answer"] = ans_norm
            fout.write(json.dumps(row) + "\n")
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = [in_path] if in_path.is_file() else sorted(in_path.rglob("*.jsonl"))

    n_rows = 0
    with out_path.open("w") as fout:
        for f in files:
            n_rows += process_file(f, fout)
    print(f"graded {n_rows} rows from {len(files)} files -> {out_path}")


if __name__ == "__main__":
    main()
