"""Evaluate Connect Four move quality using a Tinker SamplingClient."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.prompts import format_prompt, parse_response, validate_response

from experiments.opponent_next_move.eval_move_quality import (
    _load_records,
    _mean,
    _norm_from_scores,
    _reply_job,
    _validate_bank_record,
)

logger = logging.getLogger("tinker_eval")


def _project_id_from_args(args_project_id: Optional[str]) -> Optional[str]:
    return args_project_id or os.environ.get("TINKER_PROJECT_ID") or None


def _load_tinker() -> Tuple[Any, Any, Any]:
    try:
        import tinker
        from tinker_cookbook.renderers import get_renderer, get_text_content
    except ImportError as exc:
        raise RuntimeError(
            "Tinker evaluation requires optional dependencies. Install with:\n"
            "  pip install tinker tinker-cookbook"
        ) from exc
    return tinker, get_renderer, get_text_content


async def _resolve_tinker_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        value = await value
    result_async = getattr(value, "result_async", None)
    if callable(result_async):
        return await result_async()
    return value


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


async def _sample_prompts(
    *,
    sampling_client: Any,
    renderer: Any,
    tokenizer: Any,
    get_text_content: Any,
    tinker: Any,
    prompts: List[Any],
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> Tuple[List[str], List[int]]:
    sampling_params = tinker.SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )
    responses: List[str] = []
    token_counts: List[int] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        results = await asyncio.gather(
            *[
                _resolve_tinker_result(
                    sampling_client.sample_async(
                        prompt=prompt,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                )
                for prompt in chunk
            ]
        )
        for result in results:
            sequence = result.sequences[0]
            responses.append(
                _parse_sampled_text(
                    renderer=renderer,
                    tokenizer=tokenizer,
                    get_text_content=get_text_content,
                    tokens=sequence.tokens,
                )
            )
            token_counts.append(len(sequence.tokens))
    return responses, token_counts


async def evaluate_async(
    *,
    model: str,
    condition: str,
    banks_path: Path,
    output: Path,
    model_label: str,
    renderer_name: str,
    max_new_tokens: int,
    temperature: float,
    max_positions_per_split: Optional[int],
    batch_size: int,
    project_id: Optional[str],
) -> Dict[str, Any]:
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is required for Tinker evaluation.")

    tinker, get_renderer, get_text_content = _load_tinker()
    solver = PonsSolver(strict=True)
    if not solver.is_available():
        raise RuntimeError("Pons solver is required. Run scripts/bootstrap_gpu.sh first.")

    service_client = tinker.ServiceClient(project_id=project_id)
    if model.startswith("tinker://"):
        sampling_client = await _resolve_tinker_result(
            service_client.create_sampling_client_async(model_path=model)
        )
    else:
        sampling_client = await _resolve_tinker_result(
            service_client.create_sampling_client_async(base_model=model)
        )

    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer)

    records = _load_records(banks_path, max_positions_per_split)
    prompts: List[Any] = []
    envs: List[ConnectFourEnv] = []
    score_dicts: List[Dict[int, int]] = []
    position_validity: List[Tuple[bool, str]] = []

    for rec in records:
        position_valid, position_reason, env = _validate_bank_record(rec)
        position_validity.append((position_valid, position_reason))
        if not position_valid or env is None:
            raise RuntimeError(
                f"Invalid bank position {rec.get('split', 'unknown')}:{rec.get('moves')}: "
                f"{position_reason}"
            )
        envs.append(env)
        score_dicts.append({int(col): int(score) for col, score in rec["scores"].items()})
        prompts.append(
            renderer.build_generation_prompt(
                [{"role": "user", "content": format_prompt(condition, env)}]
            )
        )

    responses, token_counts = await _sample_prompts(
        sampling_client=sampling_client,
        renderer=renderer,
        tokenizer=tokenizer,
        get_text_content=get_text_content,
        tinker=tinker,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
    )

    outputs: List[Dict[str, Any]] = []
    reply_envs: List[ConnectFourEnv] = []
    reply_meta: List[Tuple[int, Optional[int]]] = []
    for idx, (
        rec,
        env,
        scores,
        response,
        generated_tokens,
        (position_valid, position_reason),
    ) in enumerate(
        zip(records, envs, score_dicts, responses, token_counts, position_validity),
        start=1,
    ):
        parsed = parse_response(response, condition)
        schema_valid, schema_reason = validate_response(parsed, condition, env.legal_moves())
        move = parsed.get("move")
        answer_valid = move is not None and move in env.legal_moves()
        move_quality = _norm_from_scores(scores, move)
        reply_env, predicted_reply, opponent_quality = _reply_job(
            env,
            move if answer_valid else None,
            parsed.get("opponent_prediction"),
        )

        best_moves = [int(col) for col in rec.get("best_moves", [])]
        outputs.append(
            {
                "index": idx,
                "split": rec.get("split", "unknown"),
                "moves": rec["moves"],
                "move_count": rec.get("move_count"),
                "position_valid": position_valid,
                "position_validity_reason": position_reason,
                "score_spread": rec.get("score_spread"),
                "model_move": move,
                "answer_valid": answer_valid,
                "schema_valid": schema_valid,
                "schema_reason": schema_reason,
                "generated_tokens": generated_tokens,
                "move_quality": move_quality,
                "optimal": answer_valid and move in best_moves,
                "opponent_prediction": parsed.get("opponent_prediction"),
                "opponent_reply_quality": opponent_quality,
                "response": response[:1000],
            }
        )
        if reply_env is not None:
            reply_meta.append((len(outputs) - 1, predicted_reply))
            reply_envs.append(reply_env)

    if reply_envs:
        for (output_idx, predicted_reply), reply_scores in zip(
            reply_meta,
            solver.analyze_batch(reply_envs),
        ):
            outputs[output_idx]["opponent_reply_quality"] = _norm_from_scores(
                reply_scores,
                predicted_reply,
            )

    by_split = {}
    for split in sorted({row["split"] for row in outputs}):
        rows = [row for row in outputs if row["split"] == split]
        opp_rows = [row for row in rows if row["opponent_reply_quality"] is not None]
        by_split[split] = {
            "n": len(rows),
            "position_valid_rate": _mean(float(row["position_valid"]) for row in rows),
            "position_invalid_count": sum(1 for row in rows if not row["position_valid"]),
            "mean_move_quality": _mean(row["move_quality"] for row in rows),
            "pct_optimal": _mean(float(row["optimal"]) for row in rows),
            "answer_valid_rate": _mean(float(row["answer_valid"]) for row in rows),
            "schema_valid_rate": _mean(float(row["schema_valid"]) for row in rows),
            "mean_opponent_reply_quality": _mean(
                row["opponent_reply_quality"] for row in opp_rows
            ),
            "opponent_reply_n": len(opp_rows),
        }

    summary = {
        "model": model,
        "model_label": model_label,
        "condition": condition,
        "banks_path": str(banks_path),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "generation_backend": "tinker",
        "renderer": renderer_name,
        "total": len(outputs),
        "generated_tokens": sum(token_counts),
        "mean_generated_tokens": _mean(token_counts),
        "position_valid_rate": _mean(float(row["position_valid"]) for row in outputs),
        "position_invalid_count": sum(1 for row in outputs if not row["position_valid"]),
        "by_split": by_split,
        "primary_mean_move_quality": _mean(
            item["mean_move_quality"] for item in by_split.values()
        ),
    }
    result = {"summary": summary, "records": outputs}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate move quality through Tinker.")
    parser.add_argument("--model", required=True, help="Tinker base model or tinker:// sampler path")
    parser.add_argument("--condition", default="BaseScaffold")
    parser.add_argument("--model_label", default=None)
    parser.add_argument(
        "--banks",
        default="experiments/opponent_next_move/data/connect4_eval_banks.jsonl",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--renderer", default="qwen3")
    parser.add_argument("--project_id", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_positions_per_split", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = parse_args()
    result = asyncio.run(
        evaluate_async(
            model=args.model,
            condition=args.condition,
            banks_path=Path(args.banks),
            output=Path(args.output),
            model_label=args.model_label or args.condition,
            renderer_name=args.renderer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_positions_per_split=args.max_positions_per_split,
            batch_size=args.batch_size,
            project_id=_project_id_from_args(args.project_id),
        )
    )
    print(json.dumps(result["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
