"""Tinker batched inference for one (model, task, condition) cell.

Follows the base-model sampling pattern used by connect4_opponent_modeling:
create a rank-1 LoRA training client, save weights for the sampler, then
create a sampling client from that path. No LoRA adapter is trained, so
this is equivalent to base-model sampling.

Set TINKER_API_KEY before running.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from prompts import build_prompt


def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_tinker_value(value):
    result = getattr(value, "result", None)
    if callable(result):
        return result()
    return value


async def _resolve_async(value):
    if asyncio.iscoroutine(value):
        return await value
    return _resolve_tinker_value(value)


async def _build_sampling_client(base_model: str, renderer_name: str, lora_rank: int):
    import tinker  # type: ignore[import-not-found]
    from tinker_cookbook.renderers import get_renderer

    service_client = tinker.ServiceClient()
    training_client = await _resolve_async(
        service_client.create_lora_training_client_async(
            base_model=base_model, rank=lora_rank
        )
    )
    save_result = await _resolve_async(
        training_client.save_weights_for_sampler_async(
            name="caveman-inference", ttl_seconds=3600
        )
    )
    sampling_client = await _resolve_async(
        service_client.create_sampling_client_async(model_path=save_result.path)
    )
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer, model_name=base_model)
    return sampling_client, renderer, tokenizer


async def _sample_one(sampling_client, renderer, tokenizer, user_msg, sampling_params):
    from tinker_cookbook.renderers import get_text_content

    messages = [{"role": "user", "content": user_msg}]
    prompt = renderer.build_generation_prompt(messages)
    result = await _resolve_async(
        sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        )
    )
    seq = result.sequences[0]
    tokens = list(seq.tokens)
    try:
        msg, ok = renderer.parse_response(tokens)
        if ok:
            return str(get_text_content(msg))
    except Exception:
        pass
    return tokenizer.decode(tokens, skip_special_tokens=True)


async def _amain(args):
    import tinker  # type: ignore[import-not-found]

    rows = load_jsonl(args.input)
    sampling_client, renderer, tokenizer = await _build_sampling_client(
        args.model, args.renderer, args.lora_rank
    )
    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=renderer.get_stop_sequences(),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)

    async def run_one(row):
        user_msg = build_prompt(args.condition, row["question"], budget=args.budget)
        async with sem:
            text = await _sample_one(
                sampling_client, renderer, tokenizer, user_msg, sampling_params
            )
        return row, user_msg, text

    completed = 0
    total = len(rows)
    results = []

    async def run_and_log(r):
        nonlocal completed
        out = await run_one(r)
        completed += 1
        if completed % 25 == 0 or completed == total:
            print(f"  {completed}/{total} done")
        return out

    results = await asyncio.gather(*(run_and_log(r) for r in rows))

    with out_path.open("w") as f:
        for row, user_msg, text in results:
            obj = {
                "id": row["id"],
                "task": row["task"],
                "condition": args.condition,
                "question": row["question"],
                "gold": row["gold"],
                "prompt": user_msg,
                "raw_output": text,
                "model": args.model,
                "renderer": args.renderer,
                "temperature": args.temperature,
                "budget": args.budget,
            }
            f.write(json.dumps(obj) + "\n")
    print(f"wrote {len(rows)} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Tinker-supported base model.",
    )
    parser.add_argument(
        "--renderer",
        default="qwen3_instruct",
        help="tinker_cookbook renderer name.",
    )
    parser.add_argument("--lora-rank", type=int, default=1)
    parser.add_argument("--task", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="reasoning-token budget for matched-budget conditions",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="max in-flight Tinker sample_async calls.",
    )
    args = parser.parse_args()

    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
