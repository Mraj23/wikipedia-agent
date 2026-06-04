"""Evaluate a sampler (base or trained) on held-out tasks.

If --sampler-path is provided (tinker:// URI from a training manifest),
the eval samples from that checkpoint. Otherwise it evaluates the base
model. Inference is greedy (temp=0) and feedback-free.

--prompt-condition selects the inference prompt (plain / concise /
chain_of_draft / caveman). The key internalization test is evaluating a
TRAINED checkpoint under `--prompt-condition plain`: if it stays terse
without the caveman prompt, the compression was internalized rather than
merely prompted.
"""

import argparse
import asyncio
import json
from pathlib import Path

from grade import grade_one
from prompts import build_prompt
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
    # Use the Tinker tokenizer (works for Tinker-hosted ids that may not be on
    # the HF Hub) for counting think/answer sub-spans.
    tok = tokenizer

    def n_tok(text):
        return len(tok.encode(text or "", add_special_tokens=False))

    done = 0
    total = len(rows)

    async def run_one(row):
        nonlocal done
        prompt = build_prompt(args.prompt_condition, row["question"])
        async with sem:
            samples = await sample_many(
                sc, renderer, tokenizer, prompt, sp, n_samples=1
            )
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}")
        return row, prompt, samples[0]

    results = await asyncio.gather(*(run_one(r) for r in rows))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row, prompt, s in results:
            text = s["text"]
            g = grade_one(text, row["gold"], row.get("task"))
            # Separate "brain" (thinking) from "mouth" (post-</think> output).
            # thinking_tokens is the PRIMARY compression target for a thinking
            # model; answer/output tokens are reported alongside it.
            thinking_tokens = n_tok(g["thinking_text"])
            output_tokens = n_tok(g["post_think_text"])  # visible answer region
            answer_tokens = n_tok(g["answer_text"])
            total_tokens = s["n_tokens"]  # exact generated token count
            obj = {
                "id": row["id"],
                "task": row["task"],
                "condition": args.condition,
                "prompt_condition": args.prompt_condition,
                "question": row["question"],
                "gold": row["gold"],
                "prompt": prompt,
                "raw_output": text,
                "model": args.model,
                "sampler": sampler_label,
                "thinking_tokens": thinking_tokens,
                "output_tokens": output_tokens,
                "answer_tokens": answer_tokens,
                "total_output_tokens": total_tokens,
                **g,
            }
            f.write(json.dumps(obj) + "\n")
    print(f"wrote {len(results)} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--renderer", default="qwen3")
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
        help="label for the model/training arm, e.g. base / rltf_sft / grpo_length",
    )
    parser.add_argument(
        "--prompt-condition",
        default="caveman",
        choices=["plain", "concise", "chain_of_draft", "caveman"],
        help="inference prompt; use 'plain' on a trained model to test internalization",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
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
