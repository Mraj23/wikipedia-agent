"""Evaluate a sampler (base or trained) on held-out tasks.

If --sampler-path is provided (tinker:// URI from a training manifest),
the eval samples from that checkpoint. Otherwise it evaluates the base
model. Inference is greedy (temp=0) and feedback-free (just x0) per
plan §13.
"""

import argparse
import asyncio
import json
from pathlib import Path

from transformers import AutoTokenizer

from grade import grade_one
from prompts import build_x0
from _tinker import (
    build_sampling_client,
    build_sampling_client_from_path,
    sample_many,
)


async def amain(args):
    import tinker  # type: ignore[import-not-found]

    rows = [json.loads(l) for l in open(args.eval_data) if l.strip()]
    if args.max_rows:
        rows = rows[: args.max_rows]

    if args.sampler_path:
        sc, renderer, tokenizer = await build_sampling_client_from_path(
            args.sampler_path, args.model, args.renderer
        )
        sampler_label = args.sampler_path
    else:
        sc, renderer, tokenizer = await build_sampling_client(
            args.model, args.renderer, lora_rank=1
        )
        sampler_label = f"base:{args.model}"

    sp = tinker.SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=renderer.get_stop_sequences(),
    )

    sem = asyncio.Semaphore(args.concurrency)
    tok = AutoTokenizer.from_pretrained(args.model)

    done = 0
    total = len(rows)

    async def run_one(row):
        nonlocal done
        x0 = build_x0(row["question"])
        async with sem:
            texts = await sample_many(
                sc, renderer, tokenizer, x0, sp, n_samples=1
            )
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}")
        return row, x0, texts[0]

    results = await asyncio.gather(*(run_one(r) for r in rows))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row, x0, text in results:
            g = grade_one(text, row["gold"])
            reasoning_tokens = len(
                tok.encode(g["reasoning_text"], add_special_tokens=False)
            )
            answer_tokens = len(
                tok.encode(g["answer_text"], add_special_tokens=False)
            )
            total_tokens = len(tok.encode(text, add_special_tokens=False))
            obj = {
                "id": row["id"],
                "task": row["task"],
                "condition": args.condition,
                "question": row["question"],
                "gold": row["gold"],
                "prompt": x0,
                "raw_output": text,
                "model": args.model,
                "sampler": sampler_label,
                "reasoning_tokens": reasoning_tokens,
                "answer_tokens": answer_tokens,
                "total_output_tokens": total_tokens,
                **g,
            }
            f.write(json.dumps(obj) + "\n")
    print(f"wrote {len(results)} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--renderer", default="qwen2_5_instruct")
    parser.add_argument(
        "--sampler-path",
        default=None,
        help="tinker:// path from a training manifest; omit to eval the base",
    )
    parser.add_argument("--manifest", default=None,
                        help="read --sampler-path from this manifest.json")
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        help="label for this eval, e.g. base / sft_caveman / rltf_sd",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    if args.manifest and not args.sampler_path:
        m = json.loads(open(args.manifest).read())
        args.sampler_path = m["final_sampler_path"]
        print(f"using sampler from manifest: {args.sampler_path}")

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
