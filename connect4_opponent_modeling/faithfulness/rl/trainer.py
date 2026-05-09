"""Tinker-based GRPO trainer for the faithfulness experiment.

Mirrors the canonical RL loop in tinker-cookbook
(tinker_cookbook/recipes/rl_loop.py): per batch, sample a group of size
`group_size` per Connect Four position, compute solver-regret rewards,
form per-token advantages = reward - group_mean, and step the model with
`forward_backward(..., loss_fn="importance_sampling")` followed by
`optim_step`.

This file imports `tinker` lazily so the rest of the package is usable
without the SDK installed.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import json
import os
import random
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.eval.board_generator import (
    env_from_record,
    load_eval_set,
    _seeded_position_pool,
)
from faithfulness.eval.training_pool import env_from_training_record, load_training_pool
from faithfulness.parse import parse_structured_response
from faithfulness.prompt import PromptCondition, make_messages
from faithfulness.rl.reward import FaithfulnessRewardCalculator
from spiral.rae import compute_advantages
from faithfulness.verifier.claim_verifier import verify_claims
from faithfulness.verifier.move_evaluator import evaluate_move
from faithfulness.rl.tinker_renderer import (
    RenderedPrompt,
    RolloutTokens,
    build_datum,
)

logger = logging.getLogger(__name__)


async def _resolve_tinker_result(value: Any) -> Any:
    """Resolve coroutine/APIFuture return shapes across Tinker SDK versions."""
    if inspect.isawaitable(value):
        value = await value
    result_async = getattr(value, "result_async", None)
    if callable(result_async):
        return await result_async()
    result = getattr(value, "result", None)
    if callable(result):
        return result()
    return value


@dataclass
class TrainerConfig:
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    condition: PromptCondition = "claims_rationale"
    renderer: str = "qwen3_instruct"
    base_url: Optional[str] = None
    project_id: Optional[str] = None
    lora_rank: int = 32
    learning_rate: float = 4e-5
    max_tokens: int = 1024
    batch_size: int = 32
    group_size: int = 8
    n_steps: int = 2000
    save_every: int = 100
    final_ttl_seconds: Optional[int] = None
    rollout_ttl_seconds: int = 3600
    fast_eval_every: int = 0
    fast_eval_n_boards: int = 25
    full_eval_every: int = 250
    eval_set_path: str = "faithfulness/data/eval_boards.jsonl"
    training_pool_path: str = "faithfulness/data/training_positions.jsonl"
    log_path: str = "faithfulness/data/runs/default"
    seed: int = 0
    train_pool_games: int = 2000
    truth_lambda: float = 0.0
    temperature: float = 1.0
    temperature_max: float = 1.5
    temperature_diversity_threshold: float = 0.8
    target_accepted_groups: int = 24
    candidate_batch_multiplier: int = 4
    max_group_attempts: int = 4
    min_reward_std: float = 1e-6


@dataclass
class StepLog:
    step: int
    mean_reward: float
    mean_regret: float
    legal_rate: float
    valid_json_rate: float
    optimal_rate: float
    walltime: float
    extra: dict = field(default_factory=dict)


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _write_run_config_snapshot(
    cfg: TrainerConfig,
    *,
    log_dir: Path,
    training_pool_loaded: bool,
) -> None:
    pool_path = Path(cfg.training_pool_path)
    eval_path = Path(cfg.eval_set_path)
    snapshot = {
        "trainer_config": asdict(cfg),
        "git_commit": _git_commit(),
        "training_pool": {
            "path": cfg.training_pool_path,
            "exists": pool_path.exists(),
            "loaded": training_pool_loaded,
            "sha256": _file_sha256(pool_path),
            "source": "explicit_pool" if training_pool_loaded else "fallback_random_self_play",
        },
        "eval_set": {
            "path": cfg.eval_set_path,
            "exists": eval_path.exists(),
            "sha256": _file_sha256(eval_path),
        },
    }
    (log_dir / "config.json").write_text(json.dumps(snapshot, indent=2))


def _reward_std(rewards: Sequence[float]) -> float:
    if len(rewards) <= 1:
        return 0.0
    return statistics.pstdev(rewards)


def group_selection_diagnostics(
    rewards: Sequence[float],
    moves: Sequence[Optional[int]],
    *,
    min_reward_std: float,
) -> Dict[str, Any]:
    """Pure helper for deciding whether a GRPO group carries signal."""
    reward_std = _reward_std(rewards)
    present_moves = [m for m in moves if m is not None]
    counts = Counter(present_moves)
    unique_moves = len(counts)
    most_common_pct = (
        max(counts.values()) / len(present_moves)
        if present_moves
        else 0.0
    )
    identical_move_group = bool(present_moves) and unique_moves == 1
    accepted = reward_std >= min_reward_std
    reason = None if accepted else "zero_reward_variance"
    return {
        "accepted": accepted,
        "skip_reason": reason,
        "reward_std": reward_std,
        "unique_moves": unique_moves,
        "most_common_move_pct": most_common_pct,
        "identical_move_group": identical_move_group,
    }


def _training_position_iterator(
    cfg: TrainerConfig,
    training_records: Optional[List[Dict]] = None,
) -> Callable[[], ConnectFourEnv]:
    pool_path = Path(cfg.training_pool_path)
    if training_records is None:
        training_records = load_training_pool(str(pool_path)) if pool_path.exists() else []
    pool = [] if training_records else _seeded_position_pool(
        seed=cfg.seed, n_games=cfg.train_pool_games
    )
    rng = random.Random(cfg.seed + 1)

    def next_env() -> ConnectFourEnv:
        if training_records:
            env = env_from_training_record(rng.choice(training_records))
        else:
            moves_str = rng.choice(pool)
            env = ConnectFourEnv()
            env.from_move_sequence([int(ch) for ch in moves_str])
        # Skip terminal positions or positions with one legal move.
        if env.is_terminal() or len(env.legal_moves()) <= 1:
            return next_env()
        return env

    return next_env


def _load_tinker() -> Tuple[Any, Any, Any, Any]:
    try:
        import tinker
        from tinker import TensorData
        from tinker_cookbook.renderers import get_renderer, get_text_content
    except ImportError as exc:
        raise RuntimeError(
            "Tinker support requires optional dependencies. Install with:\n"
            "  pip install tinker tinker-cookbook"
        ) from exc
    return tinker, TensorData, get_renderer, get_text_content


def _parse_sampled_text(
    *,
    renderer: Any,
    tokenizer: Any,
    get_text_content: Any,
    tokens: Sequence[int],
) -> str:
    try:
        parsed_message, parse_success = renderer.parse_response(list(tokens))
        if parse_success:
            return str(get_text_content(parsed_message))
    except Exception:
        pass
    return tokenizer.decode(list(tokens), skip_special_tokens=True)


async def _sample_prompt_group(
    *,
    sampling_client: Any,
    prompt: Any,
    group_size: int,
    sampling_params: Any,
) -> Any:
    sample = getattr(sampling_client, "sample_async", None) or sampling_client.sample
    return await _resolve_tinker_result(
        sample(
            prompt=prompt,
            num_samples=group_size,
            sampling_params=sampling_params,
        )
    )


async def _run_fast_eval(
    *,
    cfg: TrainerConfig,
    tinker_module: Any,
    service_client: Any,
    training_client: Any,
    renderer: Any,
    tokenizer: Any,
    get_text_content: Any,
    solver: PonsSolver,
    log_dir: Path,
    step: int,
) -> None:
    eval_path = Path(cfg.eval_set_path)
    if not eval_path.exists() or cfg.fast_eval_n_boards <= 0:
        return

    save_result = await _resolve_tinker_result(
        training_client.save_weights_for_sampler_async(
            name=f"{log_dir.name}-fast-eval-step-{step + 1:06d}",
            ttl_seconds=cfg.rollout_ttl_seconds,
        )
    )
    sampling_client = await _resolve_tinker_result(
        service_client.create_sampling_client_async(model_path=save_result.path)
    )
    sampling_params = tinker_module.SamplingParams(
        max_tokens=cfg.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=cfg.temperature,
    )

    records = load_eval_set(str(eval_path))[: cfg.fast_eval_n_boards]
    metrics = []
    truth_total = 0
    truth_true = 0
    claim_count = 0
    for item in records:
        env = env_from_record(item)
        prompt = renderer.build_generation_prompt(make_messages(env, cfg.condition))
        response = await _sample_prompt_group(
            sampling_client=sampling_client,
            prompt=prompt,
            group_size=1,
            sampling_params=sampling_params,
        )
        seq = response.sequences[0]
        completion = _parse_sampled_text(
            renderer=renderer,
            tokenizer=tokenizer,
            get_text_content=get_text_content,
            tokens=list(seq.tokens),
        )
        parsed = parse_structured_response(completion)
        if parsed.chosen_move is not None and parsed.chosen_move in env.legal_moves():
            move_eval = evaluate_move(env, parsed.chosen_move, solver)
            legal = True
            optimal = move_eval.is_optimal
            regret = move_eval.clipped_regret
        else:
            legal = False
            optimal = False
            regret = 2.0
        labels = verify_claims(parsed.claims, env, solver)
        for label in labels:
            if label is not None:
                truth_total += 1
                truth_true += 1 if label else 0
        claim_count += len(parsed.claims)
        metrics.append(
            {
                "valid_json": parsed.valid_json,
                "legal": legal,
                "optimal": optimal,
                "regret": regret,
            }
        )

    if not metrics:
        return
    entry = {
        "step": step,
        "n_boards": len(metrics),
        "valid_json_rate": sum(1 for m in metrics if m["valid_json"]) / len(metrics),
        "legal_rate": sum(1 for m in metrics if m["legal"]) / len(metrics),
        "optimal_rate": sum(1 for m in metrics if m["optimal"]) / len(metrics),
        "mean_regret": statistics.fmean(m["regret"] for m in metrics),
        "mean_claim_count": claim_count / len(metrics),
        "claim_truth_rate": truth_true / truth_total if truth_total else None,
        "sampler_path": save_result.path,
    }
    with (log_dir / "eval_log.jsonl").open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


def train(cfg: TrainerConfig) -> None:
    asyncio.run(_train_async(cfg))


async def _train_async(cfg: TrainerConfig) -> None:
    """Run the Tinker faithfulness GRPO loop.

    The Tinker SDK provides:
        - tinker.ServiceClient(base_url=...)
        - service_client.create_lora_training_client(base_model, rank)
        - training_client.save_weights_and_get_sampling_client()
        - sampling_client.sample(prompt, num_samples, sampling_params) -> Future
        - training_client.forward_backward(datums, loss_fn="importance_sampling")
        - training_client.optim_step(adam_params)
        - tinker.types.{SamplingParams, AdamParams, Datum, ModelInput, TensorData}

    The data shape for `Datum.loss_fn_inputs` is documented in
    tinker_cookbook/recipes/rl_loop.py. See faithfulness/rl/tinker_renderer.py
    for the build_datum implementation.
    """
    tinker, TensorData, get_renderer, get_text_content = _load_tinker()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log_dir = Path(cfg.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    pool_path = Path(cfg.training_pool_path)
    training_records = load_training_pool(str(pool_path)) if pool_path.exists() else []
    training_pool_loaded = bool(training_records)
    _write_run_config_snapshot(
        cfg,
        log_dir=log_dir,
        training_pool_loaded=training_pool_loaded,
    )

    service_kwargs = {}
    if cfg.base_url is not None:
        service_kwargs["base_url"] = cfg.base_url
    project_id = cfg.project_id or os.environ.get("TINKER_PROJECT_ID")
    if project_id is not None:
        service_kwargs["project_id"] = project_id
    service_client = tinker.ServiceClient(**service_kwargs)
    create_train = getattr(service_client, "create_lora_training_client_async", None)
    if create_train is None:
        create_train = service_client.create_lora_training_client
    training_client = await _resolve_tinker_result(
        create_train(
            base_model=cfg.base_model,
            rank=cfg.lora_rank,
        )
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(cfg.renderer, tokenizer, model_name=cfg.base_model)

    current_temperature = cfg.temperature
    adam_params = tinker.AdamParams(
        learning_rate=cfg.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    solver = PonsSolver(strict=True)
    reward_calc = FaithfulnessRewardCalculator(solver=solver, truth_lambda=cfg.truth_lambda)
    next_env = _training_position_iterator(cfg, training_records=training_records)
    train_log_path = log_dir / "train_log.jsonl"
    checkpoint_records = {
        "base_model": cfg.base_model,
        "renderer": cfg.renderer,
        "condition": cfg.condition,
        "sampler_paths": [],
    }

    for step in range(cfg.n_steps):
        t0 = time.time()
        sampling_params = tinker.SamplingParams(
            max_tokens=cfg.max_tokens,
            stop=renderer.get_stop_sequences(),
            temperature=current_temperature,
        )
        if hasattr(training_client, "save_weights_for_sampler_async"):
            save_result = await _resolve_tinker_result(
                training_client.save_weights_for_sampler_async(
                    name=f"faithfulness-rollout-{step:05d}",
                    ttl_seconds=cfg.rollout_ttl_seconds,
                )
            )
            sampling_client = await _resolve_tinker_result(
                service_client.create_sampling_client_async(model_path=save_result.path)
            )
        else:
            sampling_client = await _resolve_tinker_result(
                training_client.save_weights_and_get_sampling_client()
            )

        all_rewards: List[float] = []
        datums = []
        per_position_metrics = []
        generated_tokens = 0
        sample_text = ""
        step_moves: List[int] = []
        claim_type_counts: Counter = Counter()
        skipped_reasons: Counter = Counter()
        group_reward_stds: List[float] = []
        group_unique_moves: List[int] = []
        group_most_common_pcts: List[float] = []
        candidate_groups = 0
        accepted_groups = 0
        zero_variance_groups = 0
        identical_move_groups = 0
        sampled_groups = 0
        unused_sampled_groups = 0

        candidate_groups_per_attempt = max(
            cfg.batch_size,
            cfg.target_accepted_groups * cfg.candidate_batch_multiplier,
        )

        for _attempt in range(cfg.max_group_attempts):
            if accepted_groups >= cfg.target_accepted_groups:
                break
            envs: List[ConnectFourEnv] = [
                next_env() for _ in range(candidate_groups_per_attempt)
            ]
            prompts = [
                renderer.build_generation_prompt(make_messages(env, cfg.condition))
                for env in envs
            ]
            futures = [
                _sample_prompt_group(
                    sampling_client=sampling_client,
                    prompt=prompt,
                    group_size=cfg.group_size,
                    sampling_params=sampling_params,
                )
                for prompt in prompts
            ]
            sample_results = await asyncio.gather(*futures)
            sampled_groups += len(sample_results)

            for env, prompt, response in zip(envs, prompts, sample_results):
                if accepted_groups >= cfg.target_accepted_groups:
                    unused_sampled_groups += 1
                    continue
                candidate_groups += 1
                sequences = response.sequences
                group_rewards = []
                group_moves: List[Optional[int]] = []
                rollouts: List[RolloutTokens] = []

                for seq in sequences:
                    gen_ids = list(seq.tokens)
                    logprobs = list(seq.logprobs)
                    generated_tokens += len(gen_ids)
                    completion = _parse_sampled_text(
                        renderer=renderer,
                        tokenizer=tokenizer,
                        get_text_content=get_text_content,
                        tokens=gen_ids,
                    )
                    if not sample_text:
                        sample_text = completion[:700]
                    rollout = RolloutTokens(
                        prompt_token_ids=[],
                        generation_token_ids=gen_ids,
                        sample_logprobs=logprobs,
                        completion_text=completion,
                    )
                    breakdown = reward_calc.compute(env, completion)
                    parsed = parse_structured_response(completion)
                    for claim in parsed.claims:
                        claim_type_counts[claim.type.value] += 1
                    group_moves.append(parsed.chosen_move)
                    if parsed.chosen_move is not None:
                        step_moves.append(parsed.chosen_move)
                    group_rewards.append(breakdown.reward)
                    rollouts.append(rollout)
                    per_position_metrics.append(
                        {
                            "reward": breakdown.reward,
                            "regret": breakdown.regret,
                            "legal": breakdown.legal_move,
                            "valid_json": breakdown.valid_json,
                            "optimal": breakdown.debug.get("is_optimal", False),
                            "n_claims": len(parsed.claims),
                            "rationale_chars": len(parsed.rationale),
                        }
                    )

                diag = group_selection_diagnostics(
                    group_rewards,
                    group_moves,
                    min_reward_std=cfg.min_reward_std,
                )
                group_reward_stds.append(diag["reward_std"])
                group_unique_moves.append(diag["unique_moves"])
                group_most_common_pcts.append(diag["most_common_move_pct"])
                if diag["identical_move_group"]:
                    identical_move_groups += 1

                if not diag["accepted"]:
                    zero_variance_groups += 1
                    skipped_reasons[diag["skip_reason"]] += 1
                    all_rewards.extend(group_rewards)
                    continue

                adv_tensor = compute_advantages(
                    rewards=group_rewards,
                    reward_components=None,
                    reward_weights={},
                    use_rae=False,
                )
                if torch.all(torch.abs(adv_tensor) < 1e-8):
                    zero_variance_groups += 1
                    skipped_reasons["zero_advantage"] += 1
                    all_rewards.extend(group_rewards)
                    continue

                accepted_datums_before = len(datums)
                advantages = adv_tensor.tolist()
                for rollout, adv in zip(rollouts, advantages):
                    rendered = RenderedPrompt(
                        text="",
                        token_ids=[],
                        stop_sequences=renderer.get_stop_sequences(),
                        tinker_prompt=prompt,
                    )
                    datum = build_datum(
                        rendered=rendered,
                        rollout=rollout,
                        advantage=adv,
                        tinker_module=tinker,
                        tensor_data_cls=TensorData,
                    )
                    if datum is not None:
                        datums.append(datum)
                if len(datums) == accepted_datums_before:
                    skipped_reasons["no_datums"] += 1
                    all_rewards.extend(group_rewards)
                    continue
                accepted_groups += 1
                all_rewards.extend(group_rewards)

        if datums:
            forward_backward = (
                training_client.forward_backward_async
                if hasattr(training_client, "forward_backward_async")
                else training_client.forward_backward
            )
            optim_step = (
                training_client.optim_step_async
                if hasattr(training_client, "optim_step_async")
                else training_client.optim_step
            )
            fb = await _resolve_tinker_result(
                forward_backward(datums, loss_fn="importance_sampling")
            )
            await _resolve_tinker_result(optim_step(adam_params))

        t_elapsed = time.time() - t0
        legal_rate = (
            sum(1 for m in per_position_metrics if m["legal"]) / len(per_position_metrics)
            if per_position_metrics
            else 0.0
        )
        valid_rate = (
            sum(1 for m in per_position_metrics if m["valid_json"]) / len(per_position_metrics)
            if per_position_metrics
            else 0.0
        )
        optimal_rate = (
            sum(1 for m in per_position_metrics if m["optimal"]) / len(per_position_metrics)
            if per_position_metrics
            else 0.0
        )
        regret_mean = (
            statistics.fmean(m["regret"] for m in per_position_metrics)
            if per_position_metrics
            else 0.0
        )
        mean_claims = (
            statistics.fmean(m["n_claims"] for m in per_position_metrics)
            if per_position_metrics
            else 0.0
        )
        mean_rationale_chars = (
            statistics.fmean(m["rationale_chars"] for m in per_position_metrics)
            if per_position_metrics
            else 0.0
        )
        completion_count = len(per_position_metrics)

        unique_moves = len(set(step_moves))
        most_common_pct = (
            max(step_moves.count(m) for m in set(step_moves)) / len(step_moves)
            if step_moves
            else 0.0
        )
        logged_temperature = current_temperature
        if most_common_pct > cfg.temperature_diversity_threshold:
            current_temperature = min(cfg.temperature_max, current_temperature + 0.1)
        else:
            current_temperature = max(cfg.temperature, current_temperature - 0.02)
        under_target = accepted_groups < cfg.target_accepted_groups

        log_entry = {
            "step": step,
            "condition": cfg.condition,
            "mean_reward": statistics.fmean(all_rewards) if all_rewards else 0.0,
            "mean_clipped_regret": regret_mean,
            "legal_rate": legal_rate,
            "valid_json_rate": valid_rate,
            "optimal_rate": optimal_rate,
            "mean_claim_count": mean_claims,
            "mean_rationale_chars": mean_rationale_chars,
            "completion_count": completion_count,
            "generated_tokens": generated_tokens,
            "mean_generated_tokens": generated_tokens / max(completion_count, 1),
            "datums": len(datums),
            "accepted_datums": len(datums),
            "sampled_groups": sampled_groups,
            "unused_sampled_groups": unused_sampled_groups,
            "candidate_groups": candidate_groups,
            "accepted_groups": accepted_groups,
            "target_accepted_groups": cfg.target_accepted_groups,
            "zero_variance_groups": zero_variance_groups,
            "skipped_zero_variance_groups": zero_variance_groups,
            "identical_move_groups": identical_move_groups,
            "mean_group_reward_std": (
                statistics.fmean(group_reward_stds) if group_reward_stds else 0.0
            ),
            "mean_group_unique_moves": (
                statistics.fmean(group_unique_moves) if group_unique_moves else 0.0
            ),
            "mean_group_most_common_pct": (
                statistics.fmean(group_most_common_pcts) if group_most_common_pcts else 0.0
            ),
            "skipped_reasons": dict(skipped_reasons),
            "under_target_accepted_groups": under_target,
            "claim_type_counts": dict(claim_type_counts),
            "unique_moves": unique_moves,
            "most_common_move_pct": most_common_pct,
            "temperature": logged_temperature,
            "next_temperature": current_temperature,
            "step_time_s": t_elapsed,
            "sample": sample_text,
        }
        with train_log_path.open("a") as handle:
            handle.write(json.dumps(log_entry) + "\n")

        logger.info(
            (
                "step=%d reward=%.3f regret=%.3f legal=%.3f valid=%.3f "
                "optimal=%.3f claims=%.2f tokens=%.1f datums=%d accepted=%d/%d "
                "zero_var=%d time=%.1fs"
            ),
            step,
            log_entry["mean_reward"],
            regret_mean,
            legal_rate,
            valid_rate,
            optimal_rate,
            mean_claims,
            log_entry["mean_generated_tokens"],
            len(datums),
            accepted_groups,
            candidate_groups,
            zero_variance_groups,
            t_elapsed,
        )

        if cfg.fast_eval_every > 0 and (step + 1) % cfg.fast_eval_every == 0:
            await _run_fast_eval(
                cfg=cfg,
                tinker_module=tinker,
                service_client=service_client,
                training_client=training_client,
                renderer=renderer,
                tokenizer=tokenizer,
                get_text_content=get_text_content,
                solver=solver,
                log_dir=log_dir,
                step=step,
            )

        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            save_step = await _resolve_tinker_result(
                training_client.save_weights_for_sampler_async(
                    name=f"{log_dir.name}-step-{step + 1:06d}",
                    ttl_seconds=cfg.final_ttl_seconds,
                )
            )
            checkpoint_records["sampler_paths"].append(
                {"step": step + 1, "sampler_path": save_step.path}
            )
            checkpoint_records["latest_sampler_path"] = save_step.path
            (log_dir / "checkpoint_paths.json").write_text(
                json.dumps(checkpoint_records, indent=2)
            )
            logger.info("Saved sampler checkpoint for step %d: %s", step + 1, save_step.path)

    final_sampler = await _resolve_tinker_result(
        training_client.save_weights_for_sampler_async(
            name=f"{log_dir.name}-final",
            ttl_seconds=cfg.final_ttl_seconds,
        )
    )
    checkpoint_records["final_sampler_path"] = final_sampler.path
    checkpoint_records["latest_sampler_path"] = final_sampler.path
    (log_dir / "checkpoint_paths.json").write_text(
        json.dumps(checkpoint_records, indent=2)
    )
    logger.info("Final sampler checkpoint: %s", final_sampler.path)
