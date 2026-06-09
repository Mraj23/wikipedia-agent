"""Scheduled GRPO with a correctness + length reward, and the accuracy-vs-
thinking-length trajectory it traces.

Phase 1 is pure correctness RL (alpha = 0) so the model first learns to
finish-and-be-correct within the budget; phase 2 ramps alpha up so the length
term compresses the output. One run, scheduled — the honest approximation of
RLTF-SD's ramping distillation coefficient (we use a length penalty rather than
the full advantage-reweighted SD term; see ../FAITHFULNESS.md).

Reward per rollout (single turn from the chosen prompt condition):
    completed-and-correct?  reward = 1 - alpha(step) * min(n, norm)/norm
    wrong / not completed:  reward = 0
"completed" = closed </think> AND an "Answer:" line (= solved within budget).
The length penalty is on TOTAL output tokens (default) which is ungameable;
penalizing tokens-before-</think> let the model evade by never closing the tag.

Mechanics mirror the working connect4 GRPO trainer (save weights -> sampling
client; sample group_size per prompt; group-relative advantage = reward -
group_mean, skip zero-variance groups; per-token Datum; forward_backward with
importance_sampling + optim_step). Trains on RAW sampled tokens, so the loss
covers the <think> block (the SFT path would strip it via the renderer).

Survives Tinker JWT expiry: periodic save_state, and on a transient error the
clients are rebuilt (fresh JWT), state reloaded, and the run continues
in-process (no manual relaunch). A hard --max-sample-tokens guard bounds spend.
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

    correct: 1 - alpha * min(n, norm)/norm  -> in [1-alpha, 1]
    wrong:   0

    A correct answer always out-scores a wrong one (>= 1-alpha > 0) and a
    shorter correct answer always out-scores a longer one. No zero-floor cliff.
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
    padded_advantages = [0.0] * ob_len + [float(advantage)] * (model_input.length - ob_len)
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
    """Greedy eval on a fixed held-out subset -> one trajectory point."""
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
        g = grade_one(clean_generation(tokenizer, toks), row["gold"], row.get("task"))
        completed = bool(g["correct"]) and bool(g["parse_success"]) and bool(g["has_thinking"])
        return {"correct": bool(g["correct"]), "completed_correct": completed,
                "has_thinking": bool(g["has_thinking"]), "n_total": len(toks)}

    out = await asyncio.gather(*(one(r) for r in rows))
    n = len(out)
    return {
        "step": step,
        "completed_accuracy": sum(o["completed_correct"] for o in out) / n,
        "raw_accuracy": sum(o["correct"] for o in out) / n,
        "mean_total_tokens": statistics.fmean(o["n_total"] for o in out),
        "median_total_tokens": statistics.median(o["n_total"] for o in out),
        "completion_rate": sum(o["has_thinking"] for o in out) / n,
        "n": n,
    }


def _is_transient(msg: str) -> bool:
    m = msg.lower()
    return ("jwt" in m or "401" in m or "timed out" in m or "timeout" in m
            or "503" in m or "502" in m or "connection" in m)


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

    adam = tinker.AdamParams(learning_rate=args.learning_rate, beta1=0.9, beta2=0.95)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    traj_path = out_dir / "trajectory.jsonl"
    state_path_file = out_dir / "latest_state.txt"
    sem = asyncio.Semaphore(args.concurrency)

    # state["path"] is the latest saved optimizer+weights checkpoint; on a
    # transient failure we rebuild clients (fresh JWT), reload it, and continue
    # from state["resume_step"].
    state = {"path": args.resume_state, "resume_step": args.start_step}

    async def build():
        sc = tinker.ServiceClient()
        tc = await resolve_async(
            sc.create_lora_training_client_async(base_model=args.model, rank=args.lora_rank)
        )
        if state["path"]:
            print(f"  [resume] loading state {state['path']} (from step {state['resume_step']})")
            await resolve_async(tc.load_state_with_optimizer_async(state["path"]))
        tok = tc.get_tokenizer()
        rend = get_renderer(args.renderer, tok, model_name=args.model)
        spx = tinker.SamplingParams(
            max_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
            stop=rend.get_stop_sequences(),
        )
        return sc, tc, tok, rend, spx

    service_client, training_client, tokenizer, renderer, sp = await build()

    sampled_tokens = 0  # hard spend guard
    step = args.start_step
    retries = 0
    while step < args.steps:
        if sampled_tokens >= args.max_sample_tokens:
            print(f"  [cap] hit max_sample_tokens={args.max_sample_tokens:,} at step {step} "
                  f"(sampled {sampled_tokens:,}); stopping early")
            break
        try:
            save = await resolve_async(
                training_client.save_weights_for_sampler_async(
                    name=f"{args.run_name}-rollout-{step:05d}", ttl_seconds=3600
                )
            )
            sampling_client = await resolve_async(
                service_client.create_sampling_client_async(model_path=save.path)
            )

            # Trajectory checkpoint eval (weights after prior optim steps).
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
                    g = grade_one(clean_generation(tokenizer, toks), row["gold"], row.get("task"))
                    n_think = len(tokenizer.encode(g["thinking_text"], add_special_tokens=False))
                    length_metric = n_think if args.length_target == "thinking" else len(toks)
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

            if args.save_state_every and (step + 1) % args.save_state_every == 0:
                st = await resolve_async(
                    training_client.save_state_async(
                        name=f"{args.run_name}-state-{step:04d}", ttl_seconds=3600 * 24
                    )
                )
                state["path"] = st.path
                state["resume_step"] = step + 1
                state_path_file.write_text(json.dumps({"state_path": st.path, "next_step": step + 1}))
                print(f"  [state] saved -> {st.path}")

            step += 1
            retries = 0
        except Exception as e:  # noqa: BLE001 - non-transient is re-raised
            if _is_transient(str(e)) and retries < args.max_retries:
                retries += 1
                print(f"  [auto-resume] transient: {str(e)[:90]} -> rebuild + reload, "
                      f"retry from step {state['resume_step']} ({retries}/{args.max_retries})")
                service_client, training_client, tokenizer, renderer, sp = await build()
                step = state["resume_step"]
                continue
            raise

    # Final checkpoint + eval (retry on transient too).
    final = None
    for attempt in range(max(1, args.max_retries)):
        try:
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
            break
        except Exception as e:  # noqa: BLE001
            if _is_transient(str(e)) and attempt < args.max_retries - 1:
                print(f"  [auto-resume] final-block transient: {str(e)[:80]} -> rebuild")
                service_client, training_client, tokenizer, renderer, sp = await build()
                continue
            raise

    manifest = {
        "run_name": args.run_name, "base_model": args.model, "renderer": args.renderer,
        "prompt_condition": args.prompt_condition, "length_target": args.length_target,
        "token_budget": args.token_budget, "warmup_frac": args.warmup_frac,
        "alpha_max": args.alpha_max, "group_size": args.group_size, "steps": args.steps,
        "final_sampler_path": final.path if final else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"final sampler path: {final.path if final else None}\nmanifest -> {out_dir / 'manifest.json'}")


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
    parser.add_argument("--warmup-frac", type=float, default=0.3,
                        help="fraction of steps with alpha=0 (pure correctness RL)")
    parser.add_argument("--alpha-max", type=float, default=0.8)
    parser.add_argument("--token-budget", type=int, default=3072,
                        help="length normalizer: penalty reaches alpha at this many tokens. "
                        "Keep >= the model's natural length so the gradient stays live.")
    # GRPO
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sample-tokens", type=int, default=2_500_000,
                        help="hard spend guard: stop before sampling more than this many tokens")
    # resume / survival
    parser.add_argument("--save-state-every", type=int, default=4,
                        help="save a resumable optimizer+weights checkpoint every N steps")
    parser.add_argument("--resume-state", default=None, help="tinker:// state path to resume from")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=40,
                        help="max in-process rebuild+reload retries on transient errors")
    # in-run trajectory eval
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--eval-every", type=int, default=6)
    parser.add_argument("--eval-n", type=int, default=40)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
