"""Count reasoning, answer, and total output tokens using the model's tokenizer."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="override tokenizer name; defaults to row['model']",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {}

    def get_tok(name: str):
        if name not in cache:
            cache[name] = AutoTokenizer.from_pretrained(name)
        return cache[name]

    n = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            tok = get_tok(args.tokenizer or row["model"])
            reasoning = row.get("reasoning_text", "") or ""
            answer = row.get("answer_text", "") or ""
            raw = row.get("raw_output", "") or ""

            if row["condition"] == "answer_only":
                row["reasoning_tokens"] = 0
            else:
                row["reasoning_tokens"] = len(
                    tok.encode(reasoning, add_special_tokens=False)
                )
            row["answer_tokens"] = len(tok.encode(answer, add_special_tokens=False))
            row["total_output_tokens"] = len(tok.encode(raw, add_special_tokens=False))

            fout.write(json.dumps(row) + "\n")
            n += 1
    print(f"tokenized {n} rows -> {out_path}")


if __name__ == "__main__":
    main()
