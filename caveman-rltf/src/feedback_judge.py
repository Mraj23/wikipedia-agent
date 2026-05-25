"""Structured caveman-compression critique from a stronger judge model.

Default judge is Anthropic Claude (`claude-sonnet-4-6`); override with
`--judge-model`. Uses ephemeral prompt caching on the judge system
prompt — it is identical across all rows, so caching cuts judge cost
substantially.

Requires ANTHROPIC_API_KEY.
"""

import argparse
import asyncio
import json
from pathlib import Path

from anthropic import AsyncAnthropic

from prompts import build_judge_messages


async def critique_one(client, judge_model, max_tokens, question, gold, y0):
    system, user = build_judge_messages(question, gold, y0)
    msg = await client.messages.create(
        model=judge_model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


async def amain(args):
    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.max_rows:
        rows = rows[: args.max_rows]
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    total = len(rows)

    async def run_one(row):
        nonlocal done
        async with sem:
            try:
                c0 = await critique_one(
                    client,
                    args.judge_model,
                    args.max_tokens,
                    row["question"],
                    row["gold"],
                    row["raw_output"],
                )
                err = None
            except Exception as e:
                c0 = None
                err = f"{type(e).__name__}: {e}"
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}")
        return row, c0, err

    results = await asyncio.gather(*(run_one(r) for r in rows))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with out_path.open("w") as f:
        for row, c0, err in results:
            obj = dict(row)
            obj["c0"] = c0
            obj["judge_model"] = args.judge_model
            obj["judge_error"] = err
            f.write(json.dumps(obj) + "\n")
            if c0:
                n_ok += 1
    print(
        f"wrote {len(results)} ({n_ok} ok, {len(results) - n_ok} errored) -> {out_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-model", default="claude-sonnet-4-6")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
