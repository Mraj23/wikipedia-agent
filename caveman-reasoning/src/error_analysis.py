"""Sample errors per (task, condition) and apply heuristic labels.

Plan section 16 label categories (manual review fills in the rest):
- state tracking mistake
- constraint missed
- answer formatting issue
- over-compression lost necessary relation
- hallucinated transition
- correct reasoning but wrong final answer
- unlabeled

Default cells: normal_cot, chain_of_draft, caveman_full,
chain_of_draft_matched_budget.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def heuristic_label(row, cell_mean_reasoning_tokens: float) -> str:
    parse_ok = row.get("parse_success", True)
    ans = (row.get("normalized_answer") or "").strip()
    gold = (row.get("normalized_gold") or "").strip()
    reasoning = (row.get("reasoning_text") or "").strip()
    reasoning_tokens = row.get("reasoning_tokens", 0) or 0

    if not parse_ok or not ans:
        return "answer formatting issue"

    if gold and gold.lower() in reasoning.lower() and ans != gold:
        return "correct reasoning but wrong final answer"

    if (
        cell_mean_reasoning_tokens
        and reasoning_tokens < 0.4 * cell_mean_reasoning_tokens
    ):
        return "over-compression lost necessary relation"

    return "unlabeled"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/metrics/tokenized.jsonl")
    parser.add_argument("--output-dir", default="outputs/error_analysis")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "normal_cot",
            "chain_of_draft",
            "caveman_full",
            "chain_of_draft_matched_budget",
        ],
    )
    parser.add_argument("--n-per-cell", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            rows.append(json.loads(line))

    cell_tokens = defaultdict(list)
    for r in rows:
        cell_tokens[(r["task"], r["condition"])].append(
            r.get("reasoning_tokens", 0) or 0
        )
    cell_means = {
        k: (sum(v) / len(v) if v else 0) for k, v in cell_tokens.items()
    }

    by_cell = defaultdict(list)
    for r in rows:
        if r["condition"] not in args.conditions:
            continue
        if r.get("correct"):
            continue
        by_cell[(r["task"], r["condition"])].append(r)

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for (task, condition), cell_rows in sorted(by_cell.items()):
        rng.shuffle(cell_rows)
        sampled = cell_rows[: args.n_per_cell]
        cell_mean = cell_means.get((task, condition), 0.0)

        labels = defaultdict(int)
        out_path = out_dir / f"{task}__{condition}.jsonl"
        with out_path.open("w") as f:
            for r in sampled:
                label = heuristic_label(r, cell_mean)
                labels[label] += 1
                f.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "task": task,
                            "condition": condition,
                            "question": r["question"],
                            "gold": r["gold"],
                            "reasoning_text": r.get("reasoning_text", ""),
                            "answer_text": r.get("answer_text", ""),
                            "normalized_gold": r.get("normalized_gold"),
                            "normalized_answer": r.get("normalized_answer"),
                            "parse_success": r.get("parse_success"),
                            "reasoning_tokens": r.get("reasoning_tokens"),
                            "heuristic_label": label,
                            "manual_label": "",
                        }
                    )
                    + "\n"
                )
        summary_rows.append(
            {
                "task": task,
                "condition": condition,
                "n_sampled": len(sampled),
                "labels": dict(labels),
            }
        )
        print(f"wrote {len(sampled)} -> {out_path}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
