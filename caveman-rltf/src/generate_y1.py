"""Generate revised answers y1 ~ pi(. | x1)."""

import argparse
import asyncio
import json
from pathlib import Path

from _tinker import build_sampling_client, sample_many


async def amain(args):
    import tinker  # type: ignore[import-not-found]

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
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
        async with sem:
            samples = await sample_many(
                sc, renderer, tokenizer, row["x1"], sp, n_samples=args.n_samples
            )
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}")
        return row, samples

    results = await asyncio.gather(*(run_one(r) for r in rows))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row, samples in results:
            for i, s in enumerate(samples):
                obj = dict(row)
                obj["y1_sample_idx"] = i
                obj["raw_output"] = s["text"]  # so grade.py picks it up
                obj["n_gen_tokens"] = s["n_tokens"]
                f.write(json.dumps(obj) + "\n")
    print(f"wrote {len(results) * args.n_samples} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--renderer", default="qwen3")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
