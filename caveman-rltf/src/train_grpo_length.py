"""GRPO with a correctness + length reward — the RL control arm.

This is the scientific control for the RLTF-SFT arm: it rewards "short AND
correct" DIRECTLY, with no text feedback. If RLTF-SFT does not beat this at
the same output-token budget, the text-feedback machinery is not earning its
keep.

Reward (per completion, single turn from the caveman prompt x0):
    reward = max(0, 1 - alpha * n_output_tokens / token_budget)   if correct
    reward = 0                                                     otherwise

Loop (mirrors the working GRPO pattern in
connect4_opponent_modeling/faithfulness_v2/train_move_only.py):
  - save current weights -> sampling client
  - for each prompt in the step batch, sample `group_size` completions at
    temperature `temperature`
  - grade each, compute the reward above
  - group-relative advantage = reward - group_mean (skip zero-variance groups)
  - build per-token Datum (target_tokens / logprobs / advantages) and step
    with forward_backward(loss_fn="importance_sampling") + optim_step
  - write manifest.json with final_sampler_path so src/eval.py can load it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from pathlib import Path

from grade import grade_one
from prompts import build_x0
from _tinker import resolve_async


def length_reward(correct: bool, n_tokens: int, alpha: float, budget: int) -> float:
    if not correct:
        return 0.0
    return max(0.0, 1.0 - alpha * n_tokens / max(budget, 1))


def _build_rl_datum(tinker, TensorData, torch, prompt, tokens, logprobs, advantage):
    """One per-token RL datum. Mirrors train_move_only._build_rl_datum."""
    if len(tokens) < 2:
        return None
    ob_len = prompt.length - 1
    model_input = prompt.append(tinker.EncodedTextChunk(tokens=list(tokens[:-1])))
    target_tokens = [0] * ob_len + list(tokens)
    padded_logprobs = [0.0] * ob_len + [float(x) for x in logprobs]
    padded_advantages = [0.0] * ob_len + [float(advantage)] * (
        model_input.length - ob_len
    )
    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
        },
    )


def _parse_text(renderer, tokenizer, get_text_content, tokens):
    try:
        msg, ok = renderer.parse_response(list(tokens))
        if ok:
            return str(get_text_content(msg))
    except Exception:
        pass
    return tokenizer.decode(list(tokens), skip_special_tokens=True)


async def amain(args):
    import tinker  # type: ignore[import-not-found]
    import torch
    from tinker import TensorData
    from tinker_cookbook.renderers import get_renderer, get_text_content

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.train) if l.strip()]
    if not rows:
        raise SystemExit(f"empty train file: {args.train}")

    service_client = tinker.ServiceClient()
    training_client = await resolve_async(
        service_client.create_lora_training_client_async(
            base_model=args.model, rank=args.lora_rank
        )
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer, model_name=args.model)
    adam = tinker.AdamParams(
        learning_rate=args.learning_rate, beta1=0.9, beta2=0.95
    )
    sp = tinker.SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=renderer.get_stop_sequences(),
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    sem = asyncio.Semaphore(args.concurrency)

    for step in range(args.steps):
        save = await resolve_async(
            training_client.save_weights_for_sampler_async(
                name=f"{args.run_name}-rollout-{step:05d}", ttl_seconds=3600
            )
        )
        sampling_client = await resolve_async(
            service_client.create_sampling_client_async(model_path=save.path)
        )

        batch = [rng.choice(rows) for _ in range(args.batch_size)]

        async def collect(row):
            prompt = renderer.build_generation_prompt(
                [{"role": "user", "content": build_x0(row["question"])}]
            )
            async with sem:
                res = await resolve_async(
                    sampling_client.sample_async(
                        prompt=prompt, num_samples=args.group_size, sampling_params=sp
                    )
                )
            recs = []
            for seq in res.sequences:
                tokens = list(seq.tokens)
                text = _parse_text(renderer, tokenizer, get_text_content, tokens)
                g = grade_one(text, row["gold"], row.get("task"))
                n_tok = len(tokens)
                r = length_reward(bool(g["correct"]), n_tok, args.alpha, args.token_budget)
                recs.append(
                    {
                        "prompt": prompt,
                        "tokens": tokens,
                        "logprobs": list(seq.logprobs),
                        "reward": r,
                        "correct": bool(g["correct"]),
                        "n_tokens": n_tok,
                    }
                )
            return recs

        groups = await asyncio.gather(*(collect(r) for r in batch))

        datums = []
        all_rewards, all_tokens, all_correct = [], [], []
        for recs in groups:
            rewards = [x["reward"] for x in recs]
            all_rewards.extend(rewards)
            all_tokens.extend(x["n_tokens"] for x in recs)
            all_correct.extend(x["correct"] for x in recs)
            if statistics.pstdev(rewards) < 1e-6:
                continue  # zero-variance group carries no GRPO signal
            mean_r = statistics.fmean(rewards)
            for x in recs:
                adv = x["reward"] - mean_r
                d = _build_rl_datum(
                    tinker, TensorData, torch,
                    x["prompt"], x["tokens"], x["logprobs"], adv,
                )
                if d is not None:
                    datums.append(d)

        loss = None
        if datums:
            fb = await resolve_async(
                training_client.forward_backward_async(
                    datums, loss_fn="importance_sampling"
                )
            )
            await resolve_async(training_client.optim_step_async(adam))
            loss = getattr(fb, "loss", None)

        entry = {
            "step": step,
            "datums": len(datums),
            "mean_reward": statistics.fmean(all_rewards) if all_rewards else 0.0,
            "accuracy": (sum(all_correct) / len(all_correct)) if all_correct else 0.0,
            "mean_output_tokens": statistics.fmean(all_tokens) if all_tokens else 0.0,
            "loss": loss,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        print(
            f"  step {step}/{args.steps} reward={entry['mean_reward']:.3f} "
            f"acc={entry['accuracy']:.3f} tok={entry['mean_output_tokens']:.1f} "
            f"datums={len(datums)} loss={loss}"
        )

    final = await resolve_async(
        training_client.save_weights_for_sampler_async(
            name=f"{args.run_name}-final", ttl_seconds=3600 * 24 * 7
        )
    )
    manifest = {
        "run_name": args.run_name,
        "base_model": args.model,
        "renderer": args.renderer,
        "alpha": args.alpha,
        "token_budget": args.token_budget,
        "group_size": args.group_size,
        "steps": args.steps,
        "final_sampler_path": final.path,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"final sampler path: {final.path}")
    print(f"manifest -> {out_dir / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--renderer", default="qwen2_5_instruct")
    parser.add_argument("--train", required=True, help="data/processed/train.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-name", default="grpo_length")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--token-budget", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
