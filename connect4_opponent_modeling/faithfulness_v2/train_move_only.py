"""Move-only GRPO trainer. Clean v2.

One condition. One reward in [0, 1] (1.0 = optimal, 0.0 = worst legal /
illegal / unparseable). No claims. No KL shaping. Group acceptance reads
RAW reward variance only — KL, when enabled, is monitoring only and is
never subtracted from reward.

Designed to mirror the working sister trainer at
`experiments/opponent_next_move/tinker_train.py` (which already produces
held-out learning on this game) without inheriting any of v1's drift:
positions_per_step=1, group_size=16 (configurable), max_tokens=256,
temperature=0.7, lr=1e-5, no retry loop, no shaped-reward gate, no
mid-training reward sharpening.

Pass criterion: held-out mean_regret on the balanced eval drops by ≥0.10
vs the pinned base baseline. If not, claims phase does not start.

Usage:
    TINKER_API_KEY=... python -m faithfulness_v2.train_move_only \
        --pool faithfulness_v2/data/pool_v2.jsonl \
        --eval-set faithfulness/data/eval_boards.jsonl \
        --steps 100 --output faithfulness_v2/runs/pilot1
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from spiral.rae import compute_advantages

logger = logging.getLogger("v2_train_move_only")


# ---------------------------------------------------------------------------
# Prompt + parsing. Must stay byte-identical to generate_pool.py and
# eval_move_quality.py — duplicated on purpose so each script is
# independently auditable.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Connect Four player. "
    "Respond with a single column index (0-6) and nothing else."
)


def _render_board(env: ConnectFourEnv) -> str:
    rows, cols = 6, 7
    grid = [["."] * cols for _ in range(rows)]
    heights = [0] * cols
    for i, col in enumerate(env._move_history):
        row = rows - 1 - heights[col]
        grid[row][col] = "X" if i % 2 == 0 else "O"
        heights[col] += 1
    return "\n".join(" ".join(r) for r in grid)


def make_messages(env: ConnectFourEnv) -> List[Dict[str, str]]:
    board = _render_board(env)
    legal = " ".join(str(m) for m in env.legal_moves())
    player = env.current_player()
    you = "X" if player == 1 else "O"
    user = (
        f"You are player {you} (X = first to move, O = second).\n"
        f"Pieces fall to the lowest empty row.\n\n"
        f"Board:\n{board}\n"
        f"Columns: 0 1 2 3 4 5 6\n\n"
        f"Legal columns: {legal}\n\n"
        f"Your move? Output a single integer (your column choice)."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_COLUMN_RE = re.compile(r"[0-6]")


def parse_column(text: str) -> Optional[int]:
    if not text:
        return None
    m = _COLUMN_RE.search(text)
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Reward. Solver-regret in [0, 1]. The whole reward, no shaping, no bonuses.
# Inlined from faithfulness/verifier/move_evaluator.py so v2 has no
# faithfulness/ imports.
# ---------------------------------------------------------------------------

REGRET_SCALE = 8.0
REGRET_CLIP = 2.0


def move_quality(
    env: ConnectFourEnv, chosen_col: Optional[int], solver: PonsSolver
) -> Tuple[float, float, bool, bool, bool]:
    """Return (reward in [0, 1], clipped_regret, is_optimal, is_legal, is_valid_response)."""
    legal = env.legal_moves()
    if chosen_col is None:
        return 0.0, REGRET_CLIP, False, False, False
    if chosen_col not in legal:
        return 0.0, REGRET_CLIP, False, False, True
    scores = solver.analyze(env)
    if not scores:
        return 1.0, 0.0, True, True, True
    best = max(scores.values())
    chosen = scores.get(chosen_col)
    if chosen is None:
        return 0.0, REGRET_CLIP, False, False, True
    raw = max(0.0, float(best - chosen))
    clipped = min(REGRET_CLIP, raw / REGRET_SCALE)
    reward = 1.0 - clipped / REGRET_CLIP
    return reward, clipped, chosen == best, True, True


def _env_from_moves(moves: List[int]) -> ConnectFourEnv:
    env = ConnectFourEnv()
    for m in moves:
        env.make_move(int(m))
    return env


def _most_common_pct(moves: Sequence[Optional[int]]) -> float:
    valid = [m for m in moves if m is not None]
    if not valid:
        return 0.0
    counts = Counter(valid)
    return max(counts.values()) / len(valid)


# ---------------------------------------------------------------------------
# Tinker scaffolding. Lifted from experiments/opponent_next_move/tinker_train.py
# which is the known-working pattern.
# ---------------------------------------------------------------------------


async def _resolve_tinker_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        value = await value
    result_async = getattr(value, "result_async", None)
    if callable(result_async):
        return await result_async()
    return value


def _parse_sampled_text(*, renderer, tokenizer, get_text_content, tokens) -> str:
    try:
        parsed_message, ok = renderer.parse_response(list(tokens))
        if ok:
            return str(get_text_content(parsed_message))
    except Exception:
        pass
    return tokenizer.decode(list(tokens), skip_special_tokens=True)


def _build_rl_datum(*, tinker, TensorData, prompt, tokens, logprobs, advantage):
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


async def _make_sampling_client_for_current_weights(
    *, service_client, training_client, name: str, ttl_seconds: Optional[int]
) -> Tuple[Any, str]:
    save = await _resolve_tinker_result(
        training_client.save_weights_for_sampler_async(
            name=name, ttl_seconds=ttl_seconds
        )
    )
    client = await _resolve_tinker_result(
        service_client.create_sampling_client_async(model_path=save.path)
    )
    return client, save.path


async def _save_training_state(
    *,
    training_client,
    name: str,
    ttl_seconds: Optional[int],
) -> str:
    """Save resumable training weights.

    `save_weights_for_sampler` produces inference-only sampler weights. To
    continue training after a process restart, Tinker needs a `weights/...`
    checkpoint from `save_state`, optionally loaded with optimizer state by
    `load_state_with_optimizer`.
    """
    save = await _resolve_tinker_result(
        training_client.save_state_async(name, ttl_seconds=ttl_seconds)
    )
    return save.path


async def _save_training_and_sampler_checkpoint(
    *,
    service_client,
    training_client,
    name: str,
    ttl_seconds: Optional[int],
) -> Tuple[str, str]:
    state_path = await _save_training_state(
        training_client=training_client,
        name=f"{name}-state",
        ttl_seconds=ttl_seconds,
    )
    _, sampler_path = await _make_sampling_client_for_current_weights(
        service_client=service_client,
        training_client=training_client,
        name=f"{name}-sampler",
        ttl_seconds=ttl_seconds,
    )
    return state_path, sampler_path


# ---------------------------------------------------------------------------
# Held-out eval (greedy, single sample per board). Inline so the trainer is
# self-contained; eval_move_quality.py has the same logic for standalone use.
# ---------------------------------------------------------------------------


async def _held_out_eval(
    *,
    sampling_client,
    eval_boards: List[Dict[str, Any]],
    sampling_params,
    renderer,
    tokenizer,
    get_text_content,
    solver: PonsSolver,
    max_concurrent: int = 16,
) -> Dict[str, float]:
    metrics: List[Dict[str, Any]] = []
    for i in range(0, len(eval_boards), max_concurrent):
        chunk = eval_boards[i : i + max_concurrent]
        prompts = [
            renderer.build_generation_prompt(make_messages(_env_from_moves(b["moves"])))
            for b in chunk
        ]
        sample_results = await asyncio.gather(
            *[
                _resolve_tinker_result(
                    sampling_client.sample_async(
                        prompt=p, num_samples=1, sampling_params=sampling_params
                    )
                )
                for p in prompts
            ]
        )
        for board, sr in zip(chunk, sample_results):
            env = _env_from_moves(board["moves"])
            text = _parse_sampled_text(
                renderer=renderer,
                tokenizer=tokenizer,
                get_text_content=get_text_content,
                tokens=sr.sequences[0].tokens,
            )
            col = parse_column(text)
            r, regret, opt, legal, valid = move_quality(env, col, solver)
            metrics.append(
                {"reward": r, "regret": regret, "optimal": opt, "legal": legal, "valid": valid}
            )
    n = len(metrics) or 1
    return {
        "mean_reward": sum(m["reward"] for m in metrics) / n,
        "mean_regret": sum(m["regret"] for m in metrics) / n,
        "optimal_rate": sum(m["optimal"] for m in metrics) / n,
        "legal_rate": sum(m["legal"] for m in metrics) / n,
        "valid_rate": sum(m["valid"] for m in metrics) / n,
        "n": len(metrics),
    }


# ---------------------------------------------------------------------------
# Main training loop.
# ---------------------------------------------------------------------------


def _load_tinker():
    import tinker
    from tinker import TensorData
    from tinker_cookbook.renderers import get_renderer, get_text_content

    return tinker, TensorData, get_renderer, get_text_content


async def run_rl(args: argparse.Namespace) -> Dict[str, Any]:
    tinker, TensorData, get_renderer, get_text_content = _load_tinker()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = output_dir / "train_log.jsonl"
    eval_log_path = output_dir / "eval_log.jsonl"
    rollouts_path = output_dir / "rollouts.jsonl"
    checkpoints_path = output_dir / "checkpoint_paths.jsonl"
    config_path = output_dir / "config.json"
    run_start_time = time.time()

    pool_lines = Path(args.pool).read_text().splitlines()
    pool = [json.loads(l) for l in pool_lines if l.strip()]
    if not pool:
        raise RuntimeError(f"Empty pool at {args.pool}")
    logger.info("Loaded %d positions from %s", len(pool), args.pool)

    eval_boards: List[Dict[str, Any]] = []
    if args.eval_set and args.eval_every > 0:
        eval_boards = [
            json.loads(l)
            for l in Path(args.eval_set).read_text().splitlines()
            if l.strip()
        ]
        logger.info(
            "Loaded %d eval boards from %s (full set, no random subsampling)",
            len(eval_boards),
            args.eval_set,
        )

    solver = PonsSolver(strict=args.strict_solver)

    service_client = tinker.ServiceClient()
    training_client = await _resolve_tinker_result(
        service_client.create_lora_training_client_async(
            base_model=args.base_model, rank=args.lora_rank
        )
    )
    if args.resume_state:
        load_fn = (
            training_client.load_state_async
            if args.resume_weights_only
            else training_client.load_state_with_optimizer_async
        )
        logger.info(
            "Loading training state from %s (optimizer=%s)",
            args.resume_state,
            not args.resume_weights_only,
        )
        await _resolve_tinker_result(load_fn(args.resume_state))

    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer, model_name=args.base_model)

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=args.temperature,
    )
    eval_sampling_params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=args.eval_temperature,
    )
    adam_params = tinker.AdamParams(
        learning_rate=args.learning_rate, beta1=0.9, beta2=0.95
    )

    reference_sampling_client = None
    if args.log_kl:
        ref_train = await _resolve_tinker_result(
            service_client.create_lora_training_client_async(
                base_model=args.base_model, rank=1
            )
        )
        ref_save = await _resolve_tinker_result(
            ref_train.save_weights_for_sampler_async(
                name=f"{output_dir.name}-kl-ref",
                ttl_seconds=args.final_ttl_seconds,
            )
        )
        reference_sampling_client = await _resolve_tinker_result(
            service_client.create_sampling_client_async(model_path=ref_save.path)
        )
        logger.info("KL monitoring enabled (logged only, never subtracted from reward)")

    config = vars(args).copy()
    config["pool_size"] = len(pool)
    config["eval_boards_loaded"] = len(eval_boards)
    config["jwt_safety"] = {
        "max_runtime_seconds": args.max_runtime_seconds,
        "save_on_time_limit": args.save_on_time_limit,
        "resume_state": args.resume_state,
        "resume_weights_only": args.resume_weights_only,
    }
    config_path.write_text(json.dumps(config, indent=2, default=str))

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or output_dir.name,
                config=config,
                tags=["v2", "move_only", "tinker", "connect4"],
            )
        except ImportError:
            logger.warning("wandb requested but not installed; continuing without")

    skipped_groups = 0
    time_limit_checkpoint: Optional[Dict[str, Any]] = None

    # Preserve the same position stream across chunked/resumed runs. Without
    # this, a resumed chunk with the same seed would replay the first chunk's
    # pool choices under new step numbers.
    for _ in range(args.start_step):
        rng.choice(pool)

    for step in range(args.start_step, args.start_step + args.steps):
        step_start = time.time()
        position = rng.choice(pool)
        env = _env_from_moves(position["moves"])
        prompt = renderer.build_generation_prompt(make_messages(env))

        rollout_sampling_client, rollout_sampler_path = await _make_sampling_client_for_current_weights(
            service_client=service_client,
            training_client=training_client,
            name=f"{output_dir.name}-rollout-{step:05d}",
            ttl_seconds=args.rollout_ttl_seconds,
        )

        sample_result = await _resolve_tinker_result(
            rollout_sampling_client.sample_async(
                prompt=prompt,
                num_samples=args.group_size,
                sampling_params=sampling_params,
            )
        )

        raw_rewards: List[float] = []
        moves: List[Optional[int]] = []
        regrets: List[float] = []
        optimals: List[bool] = []
        legals: List[bool] = []
        valids: List[bool] = []
        completions: List[str] = []
        all_logprobs: List[List[float]] = []
        all_tokens: List[List[int]] = []
        sample_text = ""

        for seq in sample_result.sequences:
            text = _parse_sampled_text(
                renderer=renderer,
                tokenizer=tokenizer,
                get_text_content=get_text_content,
                tokens=seq.tokens,
            )
            if not sample_text:
                sample_text = text[:300]
            col = parse_column(text)
            r, regret, opt, legal, valid = move_quality(env, col, solver)
            raw_rewards.append(r)
            moves.append(col if (col is not None and col in env.legal_moves()) else col)
            regrets.append(regret)
            optimals.append(opt)
            legals.append(legal)
            valids.append(valid)
            completions.append(text)
            all_logprobs.append(list(seq.logprobs))
            all_tokens.append(list(seq.tokens))

        kl_per_rollout = [0.0] * len(sample_result.sequences)
        if reference_sampling_client is not None:
            ref_inputs = [
                prompt.append(tinker.EncodedTextChunk(tokens=list(t)))
                for t in all_tokens
            ]
            ref_results = await asyncio.gather(
                *[
                    _resolve_tinker_result(
                        reference_sampling_client.compute_logprobs_async(mi)
                    )
                    for mi in ref_inputs
                ]
            )
            for i, base_lps in enumerate(ref_results):
                n_gen = len(all_tokens[i])
                if n_gen == 0:
                    continue
                tail = base_lps[-n_gen:] if n_gen <= len(base_lps) else base_lps
                policy_sum = sum(float(x) for x in all_logprobs[i] if x is not None)
                base_sum = sum(float(x) for x in tail if x is not None)
                kl_per_rollout[i] = policy_sum - base_sum

        # Group acceptance: RAW reward variance only.
        adv_tensor = compute_advantages(
            rewards=raw_rewards,
            reward_components=None,
            reward_weights={},
            use_rae=False,
        )
        all_zero = bool(torch.all(torch.abs(adv_tensor) < 1e-8))

        log_entry: Dict[str, Any] = {
            "step": step,
            "skipped": all_zero,
            "mean_reward": statistics.fmean(raw_rewards),
            "raw_reward_std": (
                statistics.stdev(raw_rewards) if len(raw_rewards) > 1 else 0.0
            ),
            "optimal_rate": sum(optimals) / max(1, len(optimals)),
            "legal_rate": sum(legals) / max(1, len(legals)),
            "valid_rate": sum(valids) / max(1, len(valids)),
            "unique_moves": len(set(m for m in moves if m is not None)),
            "most_common_pct": _most_common_pct(moves),
            "mean_kl_to_base": (
                statistics.fmean(kl_per_rollout) if kl_per_rollout else 0.0
            ),
            "max_kl_to_base": max(kl_per_rollout) if kl_per_rollout else 0.0,
            "step_time_s": time.time() - step_start,
            "sample": sample_text,
        }

        if all_zero:
            skipped_groups += 1
            log_entry["skip_reason"] = "zero_raw_variance"
            log_entry["datums"] = 0
            log_entry["loss"] = None
            log_entry["mean_advantage"] = 0.0
        else:
            advantages = adv_tensor.tolist()
            datums: List[Any] = []
            for tokens, logprobs, adv in zip(all_tokens, all_logprobs, advantages):
                datum = _build_rl_datum(
                    tinker=tinker,
                    TensorData=TensorData,
                    prompt=prompt,
                    tokens=tokens,
                    logprobs=logprobs,
                    advantage=adv,
                )
                if datum is not None:
                    datums.append(datum)

            loss = None
            if datums:
                fb_future = await training_client.forward_backward_async(
                    datums, loss_fn="importance_sampling"
                )
                optim_future = await training_client.optim_step_async(adam_params)
                fb_result = await _resolve_tinker_result(fb_future)
                await _resolve_tinker_result(optim_future)
                loss = getattr(fb_result, "loss", None)

            log_entry["datums"] = len(datums)
            log_entry["loss"] = loss
            log_entry["mean_advantage"] = float(adv_tensor.abs().mean())
            log_entry["max_advantage"] = float(adv_tensor.abs().max())

        log_entry["skipped_groups_total"] = skipped_groups

        with train_log_path.open("a") as fh:
            fh.write(json.dumps(log_entry, default=str) + "\n")

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"train/{k}": v
                    for k, v in log_entry.items()
                    if isinstance(v, (int, float, bool)) and v is not None
                },
                step=step,
            )

        logger.info(
            "step %d/%d reward=%.3f opt=%.2f loss=%s datums=%d kl=%.3f time=%.1fs%s",
            step,
            args.start_step + args.steps - 1,
            log_entry["mean_reward"],
            log_entry["optimal_rate"],
            log_entry.get("loss"),
            log_entry.get("datums", 0),
            log_entry["mean_kl_to_base"],
            log_entry["step_time_s"],
            " [SKIPPED]" if all_zero else "",
        )

        if args.rollout_dump_every > 0 and step % args.rollout_dump_every == 0:
            with rollouts_path.open("a") as fh:
                for i, completion in enumerate(completions):
                    fh.write(
                        json.dumps(
                            {
                                "step": step,
                                "moves": position["moves"],
                                "completion": completion,
                                "chosen_move": moves[i],
                                "reward": raw_rewards[i],
                                "regret": regrets[i],
                                "optimal": optimals[i],
                                "legal": legals[i],
                                "valid": valids[i],
                                "advantage": float(adv_tensor[i])
                                if not all_zero
                                else 0.0,
                                "kl_to_base": kl_per_rollout[i],
                            }
                        )
                        + "\n"
                    )

        do_eval = (
            bool(eval_boards)
            and args.eval_every > 0
            and step > 0
            and step % args.eval_every == 0
        )
        do_save = (
            args.save_every > 0
            and step > 0
            and step % args.save_every == 0
        )

        if do_eval or do_save:
            # Eval and checkpoint records must reflect the model state AFTER
            # this step's optim_step. If the step was skipped (no update),
            # the rollout sampler is still current.
            if all_zero:
                post_sampling_client = rollout_sampling_client
                post_sampler_path = rollout_sampler_path
            else:
                post_sampling_client, post_sampler_path = (
                    await _make_sampling_client_for_current_weights(
                        service_client=service_client,
                        training_client=training_client,
                        name=f"{output_dir.name}-post-{step:05d}",
                        ttl_seconds=args.rollout_ttl_seconds,
                    )
                )

            if do_eval:
                eval_summary = await _held_out_eval(
                    sampling_client=post_sampling_client,
                    eval_boards=eval_boards,
                    sampling_params=eval_sampling_params,
                    renderer=renderer,
                    tokenizer=tokenizer,
                    get_text_content=get_text_content,
                    solver=solver,
                )
                eval_summary["step"] = step
                eval_summary["sampler_path"] = post_sampler_path
                eval_summary["post_update"] = not all_zero
                with eval_log_path.open("a") as fh:
                    fh.write(json.dumps(eval_summary) + "\n")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            f"eval/{k}": v
                            for k, v in eval_summary.items()
                            if isinstance(v, (int, float))
                        },
                        step=step,
                    )
                logger.info(
                    "eval@%d optimal=%.3f mean_regret=%.3f legal=%.3f n=%d (post_update=%s)",
                    step,
                    eval_summary["optimal_rate"],
                    eval_summary["mean_regret"],
                    eval_summary["legal_rate"],
                    eval_summary["n"],
                    not all_zero,
                )

            if do_save:
                state_path = await _save_training_state(
                    training_client=training_client,
                    name=f"{output_dir.name}-step-{step:06d}-state",
                    ttl_seconds=args.final_ttl_seconds,
                )
                with checkpoints_path.open("a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "step": step,
                                "sampler_path": post_sampler_path,
                                "state_path": state_path,
                                "post_update": not all_zero,
                                "reason": "scheduled",
                            }
                        )
                        + "\n"
                    )

        elapsed_s = time.time() - run_start_time
        if (
            args.max_runtime_seconds > 0
            and elapsed_s >= args.max_runtime_seconds
        ):
            if not args.save_on_time_limit:
                logger.warning(
                    "Stopping at step %d after %.1fs due to --max-runtime-seconds "
                    "without saving a time-limit checkpoint.",
                    step,
                    elapsed_s,
                )
                break
            state_path, sampler_path = await _save_training_and_sampler_checkpoint(
                service_client=service_client,
                training_client=training_client,
                name=f"{output_dir.name}-time-limit-step-{step:06d}",
                ttl_seconds=args.final_ttl_seconds,
            )
            time_limit_checkpoint = {
                "step": step,
                "sampler_path": sampler_path,
                "state_path": state_path,
                "post_update": not all_zero,
                "reason": "time_limit",
                "elapsed_s": elapsed_s,
            }
            with checkpoints_path.open("a") as fh:
                fh.write(json.dumps(time_limit_checkpoint) + "\n")
            logger.warning(
                "Stopping before likely Tinker JWT expiry after %.1fs. "
                "Resume with: --resume-state %s --start-step %d",
                elapsed_s,
                state_path,
                step + 1,
            )
            break

    if time_limit_checkpoint is not None:
        final_state_path = time_limit_checkpoint["state_path"]
        final_sampler_path = time_limit_checkpoint["sampler_path"]
        logger.info(
            "Using time-limit checkpoint as final for this chunk: %s",
            final_state_path,
        )
    else:
        final_state_path, final_sampler_path = await _save_training_and_sampler_checkpoint(
            service_client=service_client,
            training_client=training_client,
            name=f"{output_dir.name}-final",
            ttl_seconds=args.final_ttl_seconds,
        )
        with checkpoints_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "step": "final",
                        "sampler_path": final_sampler_path,
                        "state_path": final_state_path,
                        "reason": "final",
                    }
                )
                + "\n"
            )
        logger.info("Final sampler: %s", final_sampler_path)
        logger.info("Final training state: %s", final_state_path)

    if wandb_run is not None:
        wandb_run.finish()

    return {
        "final_sampler_path": final_sampler_path,
        "final_state_path": final_state_path,
        "skipped_groups": skipped_groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, help="Path to entropy pool JSONL.")
    parser.add_argument("--output", required=True, help="Run output directory.")
    parser.add_argument("--eval-set", default=None,
                        help="JSONL of held-out boards. Eval is skipped if omitted.")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=0.0,
        help="0 = greedy. Eval always uses the FULL --eval-set (no subsampling). "
             "If you need a smaller eval, pre-build a stratified subset offline.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Step number to start logging from when resuming a chunk.",
    )
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--rollout-dump-every", type=int, default=5)
    parser.add_argument("--rollout-ttl-seconds", type=int, default=3600)
    parser.add_argument("--final-ttl-seconds", type=int, default=7 * 24 * 3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--renderer", default="qwen3_instruct")
    parser.add_argument(
        "--resume-state",
        default=None,
        help=(
            "Tinker training-state checkpoint from save_state, e.g. "
            "tinker://.../weights/... . Use a state_path from checkpoint_paths.jsonl, "
            "not sampler_weights."
        ),
    )
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="Load only model weights on resume; default restores optimizer state too.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=1200.0,
        help=(
            "Stop before Tinker JWT/session expiry, save a resumable checkpoint, "
            "and exit cleanly. Set 0 to disable. Default 1200s (~20min), "
            "intentionally conservative because individual Tinker futures can stall."
        ),
    )
    parser.add_argument(
        "--save-on-time-limit",
        dest="save_on_time_limit",
        action="store_true",
        default=True,
        help="Save state+sampler when --max-runtime-seconds is reached. ON by default.",
    )
    parser.add_argument(
        "--no-save-on-time-limit",
        dest="save_on_time_limit",
        action="store_false",
        help="Exit on --max-runtime-seconds without saving an extra checkpoint.",
    )
    parser.add_argument(
        "--strict-solver",
        dest="strict_solver",
        action="store_true",
        default=True,
        help="Fail if Pons binary is unavailable. ON by default.",
    )
    parser.add_argument(
        "--no-strict-solver",
        dest="strict_solver",
        action="store_false",
        help="Allow silent fallback to minimax. Dev/local only — never for pilot runs.",
    )
    parser.add_argument(
        "--log-kl",
        action="store_true",
        help="Log KL to base for monitoring. Never subtracted from reward.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="faithfulness_v2")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    for field, flag in (
        ("rollout_ttl_seconds", "--rollout-ttl-seconds"),
        ("final_ttl_seconds", "--final-ttl-seconds"),
    ):
        if getattr(args, field) < 3600:
            parser.error(f"{flag} must be at least 3600; Tinker rejects shorter TTLs")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    asyncio.run(run_rl(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
