"""Split raw_output into reasoning_text and answer_text."""

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


def parse_one(condition: str, raw: str):
    if condition == "answer_only":
        return "", raw.strip(), True

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


def process_file(in_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            reasoning, answer, ok = parse_one(row["condition"], row["raw_output"])
            out = {
                "id": row["id"],
                "task": row["task"],
                "condition": row["condition"],
                "model": row["model"],
                "gold": row["gold"],
                "question": row["question"],
                "raw_output": row["raw_output"],
                "reasoning_text": reasoning,
                "answer_text": answer,
                "parse_success": ok,
            }
            fout.write(json.dumps(out) + "\n")
            n += 1
    print(f"parsed {n} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_file():
        process_file(in_path, out_path)
        return

    for f in sorted(in_path.rglob("*.jsonl")):
        rel = f.relative_to(in_path)
        process_file(f, out_path / rel)


if __name__ == "__main__":
    main()
