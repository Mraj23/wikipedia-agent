"""Exact-match grading with multiple-choice letter normalization."""

import argparse
import json
import re
from pathlib import Path


PAREN_LETTER_RE = re.compile(r"\(([a-zA-Z])\)")
LEADING_LETTER_RE = re.compile(r"^\s*([a-zA-Z])\b")
ANY_LETTER_RE = re.compile(r"\b([a-zA-Z])\b")


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


def grade_row(row):
    gold_norm = extract_choice(row["gold"])
    ans_norm = extract_choice(row["answer_text"])
    return gold_norm == ans_norm, gold_norm, ans_norm


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
