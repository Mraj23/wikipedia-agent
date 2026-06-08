"""Scheduled GRPO with a correctness + thinking-length reward.

Produces the accuracy-vs-thinking-length "hook" trajectory: phase 1 is pure
correctness RL (length coefficient alpha = 0) so the model first learns to
finish-and-be-correct within the budget (expand); phase 2 ramps alpha up so
the length term compresses the thinking (compress). One run, scheduled — this
is the honest approximation of RLTF-SD's ramping distillation coefficient
(we use a length penalty rather than the full advantage-reweighted SD term;
see ../FAITHFULNESS.md).

Reward per rollout (single turn from the chosen prompt condition):
    correct?  reward = max(0, 1 - alpha(step) * n_think_tokens / token_budget)
    wrong / truncated:  reward = 0      (truncated => no </think> => wrong)

So a correct answer that FITS the budget beats one that overflows it — which
is exactly the "complete more problems within budget B" objective.

Mechanics (verbatim from the working connect4 GRPO trainer): save weights ->
sampling client; sample group_size completions per prompt; group-relative
advantage = reward - group_mean (skip zero-variance groups); per-token Datum
(target_tokens / logprobs / advantages); forward_backward(importance_sampling)
+ optim_step.

IMPORTANT: rollouts are decoded with clean_generation (keeps the <think>
block), and we train on the RAW sampled tokens, so the loss covers the
thinking — unlike the SFT path, which goes through the renderer and would
strip <think>.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from pathlib import Path

from grade import grade_one
from prompts import build_prompt
from _tinker import resolve_async, clean_generation


def length_reward(correct: bool, n_tokens: int, alpha: float, norm: int) -> float:
    """Length-shaped reward with a LIVE gradient.

    correct answers: 1 - alpha * min(n, norm)/norm  -> in [1-alpha, 1]
    wrong / incomplete: 0

    No zero-floor cliff: a correct answer ALWAYS out-scores a wrong one
    (>= 1-alpha > 0) and a shorter correct answer always out-scores a longer
    one. The old `max(0, 1 - alpha*n/budget)` floored to 0 above n=budget,
    so when the model's natural length exceeded the budget every correct
    answer scored 0 = same as wrong, killing the gradient (the v3 null).
    """
    if not correct:
        return 0.0
    return 1.0 - alpha * min(n_tokens, norm) / max(norm, 1)


def alpha_at(step: int, steps: int, warmup_frac: float, alpha_max: float) -> float:
    """0 during warmup (pure correctness RL), then linear ramp to alpha_max."""
    w = int(warmup_frac * steps)
    if step < w:
        return 0.0
    if steps - 1 <= w:
        return alpha_max
    return alpha_max * (step - w) / (steps - 1 - w)


def _build_rl_datum(tinker, TensorData, torch, prompt, tokens, logprobs, advantage):
    """One per-token RL datum. Mirrors connect4 train_move_only._build_rl_datum."""
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


async def _heldout_eval(sampling_client, tinker, renderer, tokenizer, rows, prompt_cond,
                        max_new_tokens, concurrency, step):
    """Greedy eval on a fixed held-out subset -> trajectory point."""
    sp = tinker.SamplingParams(
        max_tokens=max_new_tokens, temperature=0.0, top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )
    sem = asyncio.Semaphore(concurrency)

    async def one(row):
        prompt = renderer.build_generation_prompt(
            [{"role": "user", "content": build_prompt(prompt_cond, row["question"])}]
        )
        async with sem:
            res = await resolve_async(
                sampling_client.sample_async(prompt=prompt, num_samples=1, sampling_params=sp)
            )
        toks = list(res.sequences[0].tokens)
        text = clean_generation(tokenizer, toks)
        g = grade_one(text, row["gold"], row.get("task"))
        n_think = len(tokenizer.encode(g["thinking_text"], add_special_tokens=False))
        completed = bool(g["correct"]) and bool(g["parse_success"]) and bool(g["has_thinking"])
        return {"correct": bool(g["correct"]), "completed_correct": completed,
                "has_thinking": bool(g["has_thinking"]), "n_think": n_think, "n_total": len(toks)}

    out = await asyncio.gather(*(one(r) for r in rows))
    n = len(out)
    return {
        "step": step,
        # completed_accuracy = solved-within-budget (closed </think> + Answer);
        # this is the trustworthy metric. raw accuracy can be inflated by the
        # grader scraping letters from unclosed/rambling output.
        "completed_accuracy": sum(o["completed_correct"] for o in out) / n,
        "raw_accuracy": sum(o["correct"] for o in out) / n,
        "mean_total_tokens": statistics.fmean(o["n_total"] for o in out),
        "median_total_tokens": statistics.median(o["n_total"] for o in out),
        "mean_thinking_tokens": statistics.fmean(o["n_think"] for o in out),
        "completion_rate": sum(o["has_thinking"] for o in out) / n,
        "n": n,
    }


async def amain(args):
    import tinker  # type: ignore[import-not-found]
    import torch
    from tinker import TensorData
    from tinker_cookbook.renderers import get_renderer

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.train) if l.strip()]
    if not rows:
        raise SystemExit(f"empty train file: {args.train}")
    eval_rows = []
    if args.eval_data:
        eval_rows = [json.loads(l) for l in open(args.eval_data) if l.strip()][: args.eval_n]

    service_client = tinker.ServiceClient()
    training_client = await resolve_async(
        service_client.create_lora_training_client_async(
            base_model=args.model, rank=args.lora_rank
        )
    )
    # Resume after a JWT expiry / interruption: reload optimizer+weights state.
    if args.resume_state:
        print(f"resuming from state: {args.resume_state} (start_step={args.start_step})")
        await resolve_async(training_client.load_state_with_optimizer_async(args.resume_state))
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer, model_name=args.model)
    adam = tinker.AdamParams(learning_rate=args.learning_rate, beta1=0.9, beta2=0.95)
    sp = tinker.SamplingParams(
        max_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        stop=renderer.get_stop_sequences(),
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    traj_path = out_dir / "trajectory.jsonl"
    sem = asyncio.Semaphore(args.concurrency)

    sampled_tokens = 0  # hard spend guard: stop before exceeding the cap
    state_path_file = out_dir / "latest_state.txt"

    for step in range(args.start_step, args.steps):
        if sampled_tokens >= args.max_sample_tokens:
            print(f"  [cap] hit max_sample_tokens={args.max_sample_tokens:,} "
                  f"at step {step} (sampled {sampled_tokens:,}); stopping early")
            break
        save = await resolve_async(
            training_client.save_weights_for_sampler_async(
                name=f"{args.run_name}-rollout-{step:05d}", ttl_seconds=3600
            )
        )
        sampling_client = await resolve_async(
            service_client.create_sampling_client_async(model_path=save.path)
        )

        # Trajectory checkpoint eval (reflects weights after prior optim steps).
        if eval_rows and (step % args.eval_every == 0):
            tp = await _heldout_eval(
                sampling_client, tinker, renderer, tokenizer, eval_rows,
                args.prompt_condition, args.max_new_tokens, args.concurrency, step,
            )
            tp["alpha"] = alpha_at(step, args.steps, args.warmup_frac, args.alpha_max)
            sampled_tokens += int(tp["mean_total_tokens"] * tp["n"])
            with traj_path.open("a") as f:
                f.write(json.dumps(tp) + "\n")
            print(f"  [traj] step {step} completed_acc={tp['completed_accuracy']:.3f} "
                  f"raw_acc={tp['raw_accuracy']:.3f} total_tok={tp['mean_total_tokens']:.0f} "
                  f"complete={tp['completion_rate']:.2f}")

        alpha = alpha_at(step, args.steps, args.warmup_frac, args.alpha_max)
        batch = [rng.choice(rows) for _ in range(args.batch_size)]

        async def collect(row):
            prompt = renderer.build_generation_prompt(
                [{"role": "user", "content": build_prompt(args.prompt_condition, row["question"])}]
            )
            async with sem:
                res = await resolve_async(
                    sampling_client.sample_async(
                        prompt=prompt, num_samples=args.group_size, sampling_params=sp
                    )
                )
            recs = []
            for seq in res.sequences:
                toks = list(seq.tokens)
                text = clean_generation(tokenizer, toks)
                g = grade_one(text, row["gold"], row.get("task"))
                n_think = len(tokenizer.encode(g["thinking_text"], add_special_tokens=False))
                length_metric = n_think if args.length_target == "thinking" else len(toks)
                # Credit ONLY a fully completed, parseable answer: a CLOSED
                # </think> AND an "Answer:" line. Without the has_thinking gate
                # the model reward-hacks by never closing </think> (which zeroes
                # the thinking-token counter) while still rambling. Pairing this
                # with --length-target total makes the length signal ungameable.
                completed = (
                    bool(g["correct"]) and bool(g["parse_success"]) and bool(g["has_thinking"])
                )
                r = length_reward(completed, length_metric, alpha, args.token_budget)
                recs.append({"prompt": prompt, "tokens": toks, "logprobs": list(seq.logprobs),
                             "reward": r, "correct": completed,
                             "n_think": n_think, "n_total": len(toks)})
            return recs

        groups = await asyncio.gather(*(collect(r) for r in batch))
        sampled_tokens += sum(x["n_total"] for recs in groups for x in recs)

        datums = []
        all_rewards, all_think, all_correct = [], [], []
        for recs in groups:
            rewards = [x["reward"] for x in recs]
            all_rewards.extend(rewards)
            all_think.extend(x["n_think"] for x in recs)
            all_correct.extend(x["correct"] for x in recs)
            if statistics.pstdev(rewards) < 1e-6:
                continue
            mean_r = statistics.fmean(rewards)
            for x in recs:
                d = _build_rl_datum(tinker, TensorData, torch, x["prompt"],
                                    x["tokens"], x["logprobs"], x["reward"] - mean_r)
                if d is not None:
                    datums.append(d)

        loss = None
        if datums:
            fb = await resolve_async(
                training_client.forward_backward_async(datums, loss_fn="importance_sampling")
            )
            await resolve_async(training_client.optim_step_async(adam))
            loss = getattr(fb, "loss", None)

        entry = {
            "step": step, "alpha": alpha, "datums": len(datums),
            "mean_reward": statistics.fmean(all_rewards) if all_rewards else 0.0,
            "accuracy": (sum(all_correct) / len(all_correct)) if all_correct else 0.0,
            "mean_thinking_tokens": statistics.fmean(all_think) if all_think else 0.0,
            "loss": loss,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        print(f"  step {step}/{args.steps} alpha={alpha:.3f} "
              f"reward={entry['mean_reward']:.3f} acc={entry['accuracy']:.3f} "
              f"think={entry['mean_thinking_tokens']:.0f} datums={len(datums)} "
              f"loss={loss} sampled_tok={sampled_tokens:,}")

        # Resumable checkpoint, so a JWT expiry / interruption can continue
        # instead of restarting (and re-spending). Relaunch with
        # --resume-state <path> --start-step <step+1>.
        if args.save_state_every and (step + 1) % args.save_state_every == 0:
            st = await resolve_async(
                training_client.save_state_async(
                    name=f"{args.run_name}-state-{step:04d}", ttl_seconds=3600 * 24
                )
            )
            state_path_file.write_text(json.dumps({"state_path": st.path, "next_step": step + 1}))
            print(f"  [state] saved -> {st.path} (resume: --resume-state {st.path} "
                  f"--start-step {step + 1})")

    # Final trajectory point + checkpoint.
    final = await resolve_async(
        training_client.save_weights_for_sampler_async(
            name=f"{args.run_name}-final", ttl_seconds=3600 * 24 * 7
        )
    )
    if eval_rows:
        sc = await resolve_async(
            service_client.create_sampling_client_async(model_path=final.path)
        )
        tp = await _heldout_eval(sc, tinker, renderer, tokenizer, eval_rows,
                                 args.prompt_condition, args.max_new_tokens,
                                 args.concurrency, args.steps)
        tp["alpha"] = args.alpha_max
        with traj_path.open("a") as f:
            f.write(json.dumps(tp) + "\n")
        print(f"  [traj] FINAL completed_acc={tp['completed_accuracy']:.3f} "
              f"total_tok={tp['mean_total_tokens']:.0f} complete={tp['completion_rate']:.2f}")

    manifest = {
        "run_name": args.run_name, "base_model": args.model, "renderer": args.renderer,
        "prompt_condition": args.prompt_condition, "length_target": args.length_target,
        "token_budget": args.token_budget, "warmup_frac": args.warmup_frac,
        "alpha_max": args.alpha_max, "group_size": args.group_size, "steps": args.steps,
        "final_sampler_path": final.path,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"final sampler path: {final.path}\nmanifest -> {out_dir / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")  # MoE: best $/token
    parser.add_argument("--renderer", default="qwen3")
    parser.add_argument("--train", required=True, help="data/processed/train.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-name", default="traj")
    parser.add_argument("--prompt-condition", default="plain",
                        choices=["plain", "concise", "chain_of_draft", "caveman"],
                        help="rollout/eval prompt; plain => learned compression is intrinsic")
    parser.add_argument("--length-target", default="total", choices=["thinking", "total"],
                        help="total (default) is ungameable; thinking-before-</think> can be "
                        "evaded by not closing the tag")
    # schedule
    parser.add_argument("--warmup-frac", type=float, default=0.4,
                        help="fraction of steps with alpha=0 (pure correctness RL)")
    parser.add_argument("--alpha-max", type=float, default=0.5)
    parser.add_argument("--token-budget", type=int, default=3072,
                        help="length normalizer: penalty reaches alpha at this many "
                        "tokens. Keep >= the model's natural length so the gradient "
                        "stays live (no zero-floor).")
    # GRPO
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sample-tokens", type=int, default=2_500_000,
                        help="hard spend guard: stop before sampling more than this "
                        "many tokens (rollouts + evals)")
    # resume after JWT expiry / interruption
    parser.add_argument("--save-state-every", type=int, default=6,
                        help="save a resumable optimizer+weights checkpoint every N steps")
    parser.add_argument("--resume-state", default=None,
                        help="tinker:// state path to resume from")
    parser.add_argument("--start-step", type=int, default=0)
    # in-run trajectory eval
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--eval-every", type=int, default=6)
    parser.add_argument("--eval-n", type=int, default=40)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
