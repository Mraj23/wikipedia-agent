"""Generate first-turn answers y0 ~ pi(. | x0) for each train row."""

import argparse
import asyncio
import json
from pathlib import Path

from prompts import build_x0
from _tinker import build_sampling_client, sample_many


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


async def amain(args):
    import tinker  # type: ignore[import-not-found]

    rows = load_jsonl(args.input)
    sc, renderer, tokenizer = await build_sampling_client(
        args.model, args.renderer, lora_rank=1
    )
    sp = tinker.SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=renderer.get_stop_sequences(),
    )

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    total = len(rows)

    async def run_one(row):
        nonlocal done
        x0 = build_x0(row["question"])
        async with sem:
            texts = await sample_many(
                sc, renderer, tokenizer, x0, sp, n_samples=args.n_samples
            )
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}")
        return row, x0, texts

    results = await asyncio.gather(*(run_one(r) for r in rows))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row, x0, texts in results:
            for i, text in enumerate(texts):
                obj = {
                    "id": row["id"],
                    "sample_idx": i,
                    "task": row["task"],
                    "question": row["question"],
                    "gold": row["gold"],
                    "x0": x0,
                    "raw_output": text,
                    "model": args.model,
                    "temperature": args.temperature,
                }
                f.write(json.dumps(obj) + "\n")
    print(f"wrote {len(results) * args.n_samples} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--renderer", default="qwen2_5_instruct")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
