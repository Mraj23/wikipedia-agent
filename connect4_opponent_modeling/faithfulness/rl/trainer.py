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

import logging
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.eval.board_generator import (
    _seeded_position_pool,
    env_from_record,
    load_eval_set,
)
from faithfulness.parse import parse_structured_response
from faithfulness.prompt import make_messages
from faithfulness.rl.reward import FaithfulnessRewardCalculator
from faithfulness.rl.tinker_renderer import (
    FaithfulnessRenderer,
    RolloutTokens,
    build_datum,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    base_model: str = "Qwen/Qwen3-4B"
    base_url: Optional[str] = None
    lora_rank: int = 32
    learning_rate: float = 4e-5
    max_tokens: int = 384
    batch_size: int = 32
    group_size: int = 8
    n_steps: int = 2000
    save_every: int = 100
    fast_eval_every: int = 50
    full_eval_every: int = 250
    eval_set_path: str = "faithfulness/data/eval_boards.jsonl"
    log_path: str = "faithfulness/data/runs/default"
    seed: int = 0
    train_pool_games: int = 2000
    truth_lambda: float = 0.0


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
    pool = _seeded_position_pool(seed=cfg.seed, n_games=cfg.train_pool_games)
    rng = random.Random(cfg.seed + 1)

    def next_env() -> ConnectFourEnv:
        moves_str = rng.choice(pool)
        env = ConnectFourEnv()
        env.from_move_sequence([int(ch) for ch in moves_str])
        # Skip terminal positions or positions with one legal move.
        if env.is_terminal() or len(env.legal_moves()) <= 1:
            return next_env()
        return env

    return next_env


def train(cfg: TrainerConfig) -> None:
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
    import tinker  # type: ignore[import-not-found]
    from tinker import types  # type: ignore[import-not-found]
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log_dir = Path(cfg.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    renderer = FaithfulnessRenderer(tokenizer=tokenizer)

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=cfg.base_model,
        rank=cfg.lora_rank,
    )

    sampling_params = types.SamplingParams(
        max_tokens=cfg.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=cfg.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    solver = PonsSolver(strict=True)
    reward_calc = FaithfulnessRewardCalculator(solver=solver, truth_lambda=cfg.truth_lambda)
    next_env = _training_position_iterator(cfg)

    for step in range(cfg.n_steps):
        t0 = time.time()
        sampling_client = training_client.save_weights_and_get_sampling_client()

        envs: List[ConnectFourEnv] = [next_env() for _ in range(cfg.batch_size)]
        renders = [renderer.render_prompt(make_messages(env)) for env in envs]

        # Submit all sampling futures in parallel; resolve below.
        futures = [
            sampling_client.sample(
                prompt=r.text,
                num_samples=cfg.group_size,
                sampling_params=sampling_params,
            )
            for r in renders
        ]

        all_rewards: List[float] = []
        datums = []
        per_position_metrics = []

        for env, rendered, future in zip(envs, renders, futures):
            response = future.result()
            sequences = response.sequences
            group_rewards = []
            rollouts: List[RolloutTokens] = []

            for seq in sequences:
                gen_ids = list(seq.tokens)
                logprobs = list(seq.logprobs)
                completion = renderer.decode(gen_ids)
                rollout = RolloutTokens(
                    prompt_token_ids=rendered.token_ids,
                    generation_token_ids=gen_ids,
                    sample_logprobs=logprobs,
                    completion_text=completion,
                )
                breakdown = reward_calc.compute(env, completion)
                group_rewards.append(breakdown.reward)
                rollouts.append(rollout)
                per_position_metrics.append(
                    {
                        "reward": breakdown.reward,
                        "regret": breakdown.regret,
                        "legal": breakdown.legal_move,
                        "valid_json": breakdown.valid_json,
                        "optimal": breakdown.debug.get("is_optimal", False),
                    }
                )

            mean_r = statistics.fmean(group_rewards)
            advantages = [r - mean_r for r in group_rewards]
            # Skip groups with zero variance to avoid burning gradient on noise.
            if max(advantages) - min(advantages) < 1e-8:
                all_rewards.extend(group_rewards)
                continue
            for rollout, adv in zip(rollouts, advantages):
                datum = build_datum(
                    rendered=rendered,
                    rollout=rollout,
                    advantage=adv,
                )
                datums.append(datum)
            all_rewards.extend(group_rewards)

        if datums:
            fb = training_client.forward_backward(datums, loss_fn="importance_sampling")
            fb.result()
            training_client.optim_step(adam_params).result()

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

        logger.info(
            "step=%d reward=%.3f regret=%.3f legal=%.3f valid=%.3f optimal=%.3f time=%.1fs",
            step,
            statistics.fmean(all_rewards) if all_rewards else 0.0,
            regret_mean,
            legal_rate,
            valid_rate,
            optimal_rate,
            t_elapsed,
        )

        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            try:
                from tinker import checkpoint_utils  # type: ignore[import-not-found]

                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{step + 1:06d}",
                    log_path=str(log_dir),
                    kind="state",
                    loop_state={"step": step + 1},
                )
            except ImportError:
                logger.warning("tinker.checkpoint_utils unavailable; skipping save")
