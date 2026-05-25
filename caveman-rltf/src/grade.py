"""Parse + grade model outputs. Handles BBH multiple-choice answers."""

import argparse
import json
import re
from pathlib import Path


REASONING_RE = re.compile(
    r"^\s*reasoning\s*:\s*(.*?)(?=^\s*answer\s*:|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
ANSWER_LINE_RE = re.compile(
    r"^\s*answer\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

PAREN_LETTER_RE = re.compile(r"\(([a-zA-Z])\)")
LEADING_LETTER_RE = re.compile(r"^\s*([a-zA-Z])\b")
ANY_LETTER_RE = re.compile(r"\b([a-zA-Z])\b")


def parse_output(raw: str):
    """Return (reasoning_text, answer_text, parse_success)."""
    answer_matches = ANSWER_LINE_RE.findall(raw)
    if answer_matches:
        answer = answer_matches[-1].strip()
        reasoning_match = REASONING_RE.search(raw)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        return reasoning, answer, True

    lines = [l for l in raw.strip().splitlines() if l.strip()]
    if not lines:
        return "", "", False
    return "\n".join(lines[:-1]).strip(), lines[-1].strip(), False


def extract_choice(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    m = PAREN_LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = LEADING_LETTER_RE.match(text)
    if m:
        return m.group(1).upper()
    matches = ANY_LETTER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    return text.upper()


def grade_one(raw_output: str, gold: str):
    reasoning, answer, parse_ok = parse_output(raw_output)
    gold_norm = extract_choice(gold)
    ans_norm = extract_choice(answer)
    return {
        "reasoning_text": reasoning,
        "answer_text": answer,
        "parse_success": parse_ok,
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
            graded = grade_one(row["raw_output"], row["gold"])
            row.update(graded)
            fout.write(json.dumps(row) + "\n")
            n += 1
    print(f"graded {n} -> {out_path}")


if __name__ == "__main__":
    main()
