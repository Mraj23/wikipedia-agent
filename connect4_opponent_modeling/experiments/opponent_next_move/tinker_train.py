"""Tinker-backed training for the Connect Four opponent-next-move experiment.

This mirrors the local narrow experiment while using Tinker's LoRA training
API for the expensive model work. Rewards, prompts, position sampling, and
advantage computation stay local so the scientific comparison is the same:

    Value:             reward = move_quality
    OpponentNextMove:  reward = 0.8 * move_quality + 0.2 * opponent_reply_quality

The script also supports an optional SFTBestMove baseline that behavior-clones
the solver's best move with the answer-only scaffold.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from spiral.position_buffer import PositionBuffer
from spiral.rae import compute_advantages
from training.grpo_config import get_config
from training.prompts import format_prompt, parse_response, validate_response
from training.reward import RewardCalculator

logger = logging.getLogger("tinker_connect4")


def _project_id_from_args(args: argparse.Namespace) -> Optional[str]:
    return args.project_id or os.environ.get("TINKER_PROJECT_ID") or None


def _load_tinker() -> Tuple[Any, Any, Any, Any, Any]:
    try:
        import tinker
        from tinker import TensorData
        from tinker_cookbook.renderers import TrainOnWhat, get_renderer, get_text_content
        from tinker_cookbook.supervised.data import conversation_to_datum
    except ImportError as exc:
        raise RuntimeError(
            "Tinker support requires optional dependencies. Install with:\n"
            "  pip install tinker tinker-cookbook"
        ) from exc
    return tinker, TensorData, get_renderer, get_text_content, (
        TrainOnWhat,
        conversation_to_datum,
    )


async def _resolve_tinker_result(value: Any) -> Any:
    """Resolve Tinker coroutine/APIFuture return shapes across SDK versions."""
    if inspect.isawaitable(value):
        value = await value
    result_async = getattr(value, "result_async", None)
    if callable(result_async):
        return await result_async()
    return value


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _load_or_build_position_buffer(path: Path, *, seed: int, pool_size: int) -> PositionBuffer:
    if path.exists():
        logger.info("Loading position buffer from %s", path)
        return PositionBuffer.load(str(path), seed=seed)

    logger.info("Building position buffer at %s", path)
    buffer = PositionBuffer(pool_size=pool_size, min_moves_remaining=2, seed=seed)
    buffer.save(str(path))
    return buffer


def _best_solver_move(solver: PonsSolver, env: ConnectFourEnv, rng: random.Random) -> int:
    scores = solver.analyze(env)
    if not scores:
        return rng.choice(env.legal_moves())
    max_score = max(scores.values())
    best = [col for col, score in scores.items() if score == max_score]
    return rng.choice(sorted(best))


def _sft_completion(move: int) -> str:
    return (
        "<reasoning>The solver's highest-scoring move is selected.</reasoning>\n"
        f"<answer>{move}</answer>"
    )


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


def _compute_reward(
    *,
    condition: str,
    env: ConnectFourEnv,
    response: str,
    solver: PonsSolver,
    reward_calc: RewardCalculator,
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    parsed = parse_response(response, condition)
    valid, reason = validate_response(parsed, condition, env.legal_moves())
    weights = get_config(condition).reward_weights or {}
    if not valid or parsed.get("move") is None:
        return (
            0.0,
            {name: 0.0 for name in weights},
            {
                "valid": False,
                "validity_reason": reason,
                "move": parsed.get("move"),
                "opponent_prediction": parsed.get("opponent_prediction"),
            },
        )

    move = int(parsed["move"])
    move_quality = solver.normalize_reward(env, move)
    if condition == "Value":
        components = {"move": move_quality}
    elif condition == "OpponentNextMove":
        predicted = parsed.get("opponent_prediction")
        if predicted is None:
            predicted = -1
        pred_quality = reward_calc._prediction_accuracy(env, move, int(predicted))
        components = {"move": move_quality, "pred": pred_quality}
    else:
        raise ValueError(f"Unsupported RL condition for Tinker: {condition}")

    reward = sum(weights.get(name, 0.0) * value for name, value in components.items())
    return (
        reward,
        components,
        {
            "valid": True,
            "validity_reason": "",
            "move": move,
            "opponent_prediction": parsed.get("opponent_prediction"),
        },
    )


def _build_rl_datum(
    *,
    tinker: Any,
    TensorData: Any,
    prompt: Any,
    tokens: Sequence[int],
    logprobs: Sequence[float],
    advantage: float,
) -> Optional[Any]:
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


async def _make_sampling_client_for_current_weights(
    *,
    service_client: Any,
    training_client: Any,
    name: str,
    ttl_seconds: Optional[int],
) -> Tuple[Any, str]:
    save_result = await _resolve_tinker_result(
        training_client.save_weights_for_sampler_async(
            name=name,
            ttl_seconds=ttl_seconds,
        )
    )
    sampler_path = save_result.path
    sampling_client = await _resolve_tinker_result(
        service_client.create_sampling_client_async(model_path=sampler_path)
    )
    return sampling_client, sampler_path


async def run_rl(args: argparse.Namespace) -> Dict[str, Any]:
    if args.condition not in {"Value", "OpponentNextMove"}:
        raise ValueError("--condition must be Value or OpponentNextMove for --mode rl")

    tinker, TensorData, get_renderer, get_text_content, _ = _load_tinker()
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required for Tinker training.")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    solver = PonsSolver(strict=True)
    if not solver.is_available():
        raise RuntimeError("Pons solver is required. Run scripts/bootstrap_gpu.sh first.")
    reward_calc = RewardCalculator(solver)
    position_buffer = _load_or_build_position_buffer(
        Path(args.position_buffer),
        seed=args.seed,
        pool_size=args.position_pool_size,
    )

    service_client = tinker.ServiceClient(project_id=_project_id_from_args(args))
    training_client = await _resolve_tinker_result(
        service_client.create_lora_training_client_async(
            base_model=args.base_model,
            rank=args.lora_rank,
        )
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer)

    adam_params = tinker.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
    )
    reward_weights = get_config(args.condition).reward_weights or {}
    current_temperature = args.temperature

    config = {
        "mode": "rl",
        "condition": args.condition,
        "base_model": args.base_model,
        "renderer": args.renderer,
        "steps": args.steps,
        "positions_per_step": args.positions_per_step,
        "group_size": args.group_size,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "loss_fn": args.loss_fn,
        "reward_weights": reward_weights,
        "use_rae": args.use_rae,
        "seed": args.seed,
        "position_buffer": args.position_buffer,
        "position_buffer_size": len(position_buffer),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    train_log_path = output_dir / "train_log.jsonl"
    checkpoint_records: Dict[str, Any] = {"rollout_sampler_paths": []}

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"tinker_{args.condition}_seed{args.seed}",
                config=config,
                tags=["tinker", "connect4", args.condition],
                reinit=True,
            )
        except ImportError:
            logger.warning("wandb requested but not installed; continuing without W&B.")

    for step in range(args.steps):
        step_start = time.time()
        envs = position_buffer.sample(batch_size=args.positions_per_step)
        prompts = [
            renderer.build_generation_prompt(
                [{"role": "user", "content": format_prompt(args.condition, env)}]
            )
            for env in envs
        ]
        sampling_params = tinker.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=current_temperature,
            stop=renderer.get_stop_sequences(),
        )

        sampling_client, sampler_path = await _make_sampling_client_for_current_weights(
            service_client=service_client,
            training_client=training_client,
            name=f"{args.run_name_prefix}-rollout-{step:05d}",
            ttl_seconds=args.rollout_ttl_seconds,
        )
        checkpoint_records["rollout_sampler_paths"].append(
            {"step": step, "path": sampler_path}
        )

        sample_results = await asyncio.gather(
            *[
                _resolve_tinker_result(
                    sampling_client.sample_async(
                        prompt=prompt,
                        num_samples=args.group_size,
                        sampling_params=sampling_params,
                    )
                )
                for prompt in prompts
            ]
        )

        datums: List[Any] = []
        step_rewards: List[float] = []
        valid_count = 0
        completion_count = 0
        generated_tokens = 0
        skipped_groups = 0
        all_components: List[Dict[str, float]] = []
        moves: List[Optional[int]] = []
        sample_text = ""

        for env, prompt, sample_result in zip(envs, prompts, sample_results):
            group_rewards: List[float] = []
            group_components: List[Dict[str, float]] = []
            group_tokens: List[Sequence[int]] = []
            group_logprobs: List[Sequence[float]] = []
            group_texts: List[str] = []

            for sequence in sample_result.sequences:
                generated_tokens += len(sequence.tokens)
                text = _parse_sampled_text(
                    renderer=renderer,
                    tokenizer=tokenizer,
                    get_text_content=get_text_content,
                    tokens=sequence.tokens,
                )
                if not sample_text:
                    sample_text = text[:700]
                reward, components, meta = _compute_reward(
                    condition=args.condition,
                    env=env,
                    response=text,
                    solver=solver,
                    reward_calc=reward_calc,
                )
                group_rewards.append(float(reward))
                group_components.append(components)
                group_tokens.append(sequence.tokens)
                group_logprobs.append(sequence.logprobs)
                group_texts.append(text)
                all_components.append(components)
                step_rewards.append(float(reward))
                valid_count += int(bool(meta["valid"]))
                completion_count += 1
                moves.append(meta.get("move"))

            advantages = compute_advantages(
                rewards=group_rewards,
                reward_components=group_components,
                reward_weights=reward_weights,
                use_rae=args.use_rae,
            )
            if torch.all(torch.abs(advantages) < 1e-8):
                skipped_groups += 1
                continue

            for tokens, logprobs, advantage in zip(
                group_tokens, group_logprobs, advantages.tolist()
            ):
                datum = _build_rl_datum(
                    tinker=tinker,
                    TensorData=TensorData,
                    prompt=prompt,
                    tokens=tokens,
                    logprobs=logprobs,
                    advantage=float(advantage),
                )
                if datum is not None:
                    datums.append(datum)

        loss = None
        if datums:
            fwdbwd_future = await training_client.forward_backward_async(
                datums,
                loss_fn=args.loss_fn,
            )
            optim_future = await training_client.optim_step_async(adam_params)
            fwdbwd_result = await _resolve_tinker_result(fwdbwd_future)
            await _resolve_tinker_result(optim_future)
            loss = getattr(fwdbwd_result, "loss", None)

        move_values = [move for move in moves if move is not None]
        unique_moves = len(set(move_values))
        most_common_pct = 0.0
        if move_values:
            most_common_pct = max(move_values.count(move) for move in set(move_values)) / len(
                move_values
            )
        logged_temperature = current_temperature
        if most_common_pct > 0.8:
            current_temperature = min(1.5, current_temperature + 0.1)
        else:
            current_temperature = max(args.temperature, current_temperature - 0.02)

        component_means = {
            name: _mean([comp.get(name, 0.0) for comp in all_components])
            for name in reward_weights
        }
        log_entry = {
            "step": step,
            "loss": loss,
            "mean_reward": _mean(step_rewards),
            "max_reward": max(step_rewards) if step_rewards else 0.0,
            "min_reward": min(step_rewards) if step_rewards else 0.0,
            "reward_std": _std(step_rewards),
            "component_means": component_means,
            "valid_rate": valid_count / max(completion_count, 1),
            "completion_count": completion_count,
            "generated_tokens": generated_tokens,
            "mean_generated_tokens": generated_tokens / max(completion_count, 1),
            "datums": len(datums),
            "skipped_groups": skipped_groups,
            "unique_moves": unique_moves,
            "most_common_move_pct": most_common_pct,
            "temperature": logged_temperature,
            "next_temperature": current_temperature,
            "rollout_sampler_path": sampler_path,
            "step_time_s": time.time() - step_start,
            "sample": sample_text,
        }
        with train_log_path.open("a") as handle:
            handle.write(json.dumps(log_entry, default=_json_default) + "\n")

        logger.info(
            "Step %d/%d reward=%.3f valid=%.0f%% datums=%d skipped=%d moves=%d time=%.1fs",
            step,
            args.steps,
            log_entry["mean_reward"],
            100 * log_entry["valid_rate"],
            log_entry["datums"],
            skipped_groups,
            unique_moves,
            log_entry["step_time_s"],
        )
        if wandb_run is not None:
            wandb_payload = {
                "trainer_step": step,
                "train/reward_mean": log_entry["mean_reward"],
                "train/reward_std": log_entry["reward_std"],
                "train/valid_rate": log_entry["valid_rate"],
                "train/datums": log_entry["datums"],
                "train/generated_tokens": generated_tokens,
                "train/mean_generated_tokens": log_entry["mean_generated_tokens"],
                "train/skipped_groups": skipped_groups,
                "train/unique_moves": unique_moves,
                "train/most_common_move_pct": most_common_pct,
                "train/temperature": logged_temperature,
                "train/next_temperature": current_temperature,
                "train/step_time_s": log_entry["step_time_s"],
            }
            if loss is not None:
                wandb_payload["train/loss"] = loss
            for name, value in component_means.items():
                wandb_payload[f"reward/{name}_mean"] = value
            if step == 0 or step % max(1, args.log_sample_every) == 0:
                wandb_payload["samples/sample_completion"] = sample_text[:500]
            wandb_run.log(wandb_payload, step=step)

    final_sampler = await _resolve_tinker_result(
        training_client.save_weights_for_sampler_async(
            name=f"{args.run_name_prefix}-final",
            ttl_seconds=args.final_ttl_seconds,
        )
    )
    checkpoint_records["final_sampler_path"] = final_sampler.path

    if args.save_state:
        final_state = await _resolve_tinker_result(
            training_client.save_state_async(
                name=f"{args.run_name_prefix}-state-final",
                ttl_seconds=args.final_ttl_seconds,
            )
        )
        checkpoint_records["final_state_path"] = final_state.path

    (output_dir / "checkpoint_paths.json").write_text(
        json.dumps(checkpoint_records, indent=2, default=_json_default)
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "trainer_step": args.steps,
                "final/total_steps": args.steps,
                "final/sampler_path": final_sampler.path,
            },
            step=args.steps,
        )
        wandb_run.finish()

    logger.info("Final sampler checkpoint: %s", final_sampler.path)
    return checkpoint_records


async def run_sft(args: argparse.Namespace) -> Dict[str, Any]:
    tinker, _, get_renderer, _, supervised = _load_tinker()
    TrainOnWhat, conversation_to_datum = supervised
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required for Tinker training.")

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    solver = PonsSolver(strict=True)
    if not solver.is_available():
        raise RuntimeError("Pons solver is required. Run scripts/bootstrap_gpu.sh first.")
    position_buffer = _load_or_build_position_buffer(
        Path(args.position_buffer),
        seed=args.seed,
        pool_size=args.position_pool_size,
    )

    service_client = tinker.ServiceClient(project_id=_project_id_from_args(args))
    training_client = await _resolve_tinker_result(
        service_client.create_lora_training_client_async(
            base_model=args.base_model,
            rank=args.lora_rank,
        )
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer)
    adam_params = tinker.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
    )

    config = {
        "mode": "sft",
        "condition": "SFTBestMove",
        "prompt_condition": args.sft_prompt_condition,
        "base_model": args.base_model,
        "renderer": args.renderer,
        "steps": args.steps,
        "positions_per_step": args.positions_per_step,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "seed": args.seed,
        "position_buffer": args.position_buffer,
        "position_buffer_size": len(position_buffer),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    train_log_path = output_dir / "train_log.jsonl"
    for step in range(args.steps):
        step_start = time.time()
        envs = position_buffer.sample(batch_size=args.positions_per_step)
        datums: List[Any] = []
        chosen_moves: List[int] = []
        for env in envs:
            best_move = _best_solver_move(solver, env, rng)
            chosen_moves.append(best_move)
            messages = [
                {"role": "user", "content": format_prompt(args.sft_prompt_condition, env)},
                {"role": "assistant", "content": _sft_completion(best_move)},
            ]
            datum = conversation_to_datum(
                messages,
                renderer,
                max_length=args.sft_max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            datums.append(datum)

        fwdbwd_future = await training_client.forward_backward_async(
            datums,
            loss_fn="cross_entropy",
        )
        optim_future = await training_client.optim_step_async(adam_params)
        fwdbwd_result = await _resolve_tinker_result(fwdbwd_future)
        await _resolve_tinker_result(optim_future)
        loss = getattr(fwdbwd_result, "loss", None)

        log_entry = {
            "step": step,
            "loss": loss,
            "datums": len(datums),
            "unique_best_moves": len(set(chosen_moves)),
            "step_time_s": time.time() - step_start,
        }
        with train_log_path.open("a") as handle:
            handle.write(json.dumps(log_entry, default=_json_default) + "\n")
        logger.info(
            "SFT step %d/%d loss=%s datums=%d moves=%d time=%.1fs",
            step,
            args.steps,
            loss,
            len(datums),
            log_entry["unique_best_moves"],
            log_entry["step_time_s"],
        )

    final_sampler = await _resolve_tinker_result(
        training_client.save_weights_for_sampler_async(
            name=f"{args.run_name_prefix}-sft-final",
            ttl_seconds=args.final_ttl_seconds,
        )
    )
    records = {"final_sampler_path": final_sampler.path}
    if args.save_state:
        final_state = await _resolve_tinker_result(
            training_client.save_state_async(
                name=f"{args.run_name_prefix}-sft-state-final",
                ttl_seconds=args.final_ttl_seconds,
            )
        )
        records["final_state_path"] = final_state.path

    (output_dir / "checkpoint_paths.json").write_text(
        json.dumps(records, indent=2, default=_json_default)
    )
    logger.info("Final SFT sampler checkpoint: %s", final_sampler.path)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Connect Four opponent-next-move conditions on Tinker."
    )
    parser.add_argument("--mode", choices=["rl", "sft"], default="rl")
    parser.add_argument("--condition", default="Value", choices=["Value", "OpponentNextMove"])
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--renderer", default="qwen3")
    parser.add_argument("--project_id", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name_prefix", default="connect4-tinker")
    parser.add_argument("--position_buffer", default="data/position_buffer.json")
    parser.add_argument("--position_pool_size", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--positions_per_step", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--loss_fn", default="importance_sampling")
    parser.add_argument("--no_rae", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--rollout_ttl_seconds", type=int, default=7 * 24 * 3600)
    parser.add_argument("--final_ttl_seconds", type=int, default=None)
    parser.add_argument("--save_state", action="store_true")
    parser.add_argument("--sft_prompt_condition", default="BaseSimple")
    parser.add_argument("--sft_max_length", type=int, default=2048)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="connect4-opponent-modeling")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--log_sample_every", type=int, default=10)
    args = parser.parse_args()
    args.use_rae = not args.no_rae
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = parse_args()
    if args.mode == "rl":
        asyncio.run(run_rl(args))
    else:
        asyncio.run(run_sft(args))


if __name__ == "__main__":
    main()
