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
import inspect
import logging
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.eval.board_generator import (
    _seeded_position_pool,
)
from faithfulness.eval.training_pool import env_from_training_record, load_training_pool
from faithfulness.parse import parse_structured_response
from faithfulness.prompt import PromptCondition, make_messages
from faithfulness.rl.reward import FaithfulnessRewardCalculator
from spiral.rae import compute_advantages
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
    fast_eval_every: int = 50
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


def _training_position_iterator(
    cfg: TrainerConfig,
) -> Callable[[], ConnectFourEnv]:
    pool_path = Path(cfg.training_pool_path)
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
    next_env = _training_position_iterator(cfg)
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

        envs: List[ConnectFourEnv] = [next_env() for _ in range(cfg.batch_size)]
        prompts = [
            renderer.build_generation_prompt(make_messages(env, cfg.condition))
            for env in envs
        ]

        # Submit all sampling futures in parallel; resolve below.
        futures = [
            _resolve_tinker_result(
                (getattr(sampling_client, "sample_async", None) or sampling_client.sample)(
                    prompt=prompt,
                    num_samples=cfg.group_size,
                    sampling_params=sampling_params,
                )
            )
            for prompt in prompts
        ]
        sample_results = await asyncio.gather(*futures)

        all_rewards: List[float] = []
        datums = []
        per_position_metrics = []
        generated_tokens = 0
        skipped_groups = 0
        sample_text = ""
        step_moves: List[int] = []

        for env, prompt, response in zip(envs, prompts, sample_results):
            sequences = response.sequences
            group_rewards = []
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

            adv_tensor = compute_advantages(
                rewards=group_rewards,
                reward_components=None,
                reward_weights={},
                use_rae=False,
            )
            # Skip groups with zero variance: standardized advantages collapse to ~0.
            if torch.all(torch.abs(adv_tensor) < 1e-8):
                skipped_groups += 1
                all_rewards.extend(group_rewards)
                continue
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
            "skipped_zero_variance_groups": skipped_groups,
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
                "optimal=%.3f claims=%.2f tokens=%.1f datums=%d skipped=%d time=%.1fs"
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
            skipped_groups,
            t_elapsed,
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
