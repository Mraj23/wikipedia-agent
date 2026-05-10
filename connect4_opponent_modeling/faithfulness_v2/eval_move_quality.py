"""Deterministic, balanced eval. Base vs checkpoint.

Scores one or more named runs (base or trained checkpoint) on a fixed eval
set. Outputs per-run summary metrics for direct comparison; the eval set
and the prompt format are pinned so every run sees the same problem.

Modes:
  greedy      temperature=0, num_samples=1. The headline number — fully
              reproducible, single best move per board.
  stochastic  temperature=T, num_samples=N. Distribution-aware: reports the
              mean of metrics over N samples per board, plus column-share
              statistics that surface mode collapse.

Each --run argument is "<label>:<checkpoint_path_or_BASE>".
Use BASE as the path to score the un-trained base model via the
LoRA-rank-1 + save_weights_for_sampler trick.

Usage:
    TINKER_API_KEY=... python -m faithfulness_v2.eval_move_quality \
        --eval-set faithfulness/data/eval_boards.jsonl \
        --run base:BASE \
        --run pilot1:tinker://... \
        --mode greedy \
        --output faithfulness_v2/runs/base_vs_pilot1.json
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver

logger = logging.getLogger("v2_eval_move_quality")


# ---- Prompt + parse (must match generate_pool.py / train_move_only.py) ----

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


# ---- Reward (same math as train_move_only.py) ----

REGRET_SCALE = 8.0
REGRET_CLIP = 2.0


def move_quality(
    env: ConnectFourEnv, chosen_col: Optional[int], solver: PonsSolver
) -> Tuple[float, float, bool, bool, bool]:
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


# ---- Tinker scaffolding ----


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


async def _build_sample_fn(
    *,
    checkpoint_path: str,
    base_model: str,
    renderer_name: str,
    max_tokens: int,
    temperature: float,
    num_samples: int,
):
    """Returns a coroutine fn(env) -> List[str] that yields N completions."""
    import tinker  # type: ignore[import-not-found]
    from tinker_cookbook.renderers import get_renderer, get_text_content

    service_client = tinker.ServiceClient()

    if checkpoint_path == "BASE":
        training_client = await _resolve_tinker_result(
            service_client.create_lora_training_client_async(
                base_model=base_model, rank=1
            )
        )
        save = await _resolve_tinker_result(
            training_client.save_weights_for_sampler_async(
                name="v2-eval-base", ttl_seconds=3600
            )
        )
        path = save.path
    else:
        path = checkpoint_path

    sampling_client = await _resolve_tinker_result(
        service_client.create_sampling_client_async(model_path=path)
    )
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer, model_name=base_model)
    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
    )

    async def sample_fn(env: ConnectFourEnv) -> List[str]:
        prompt = renderer.build_generation_prompt(make_messages(env))
        result = await _resolve_tinker_result(
            sampling_client.sample_async(
                prompt=prompt,
                num_samples=num_samples,
                sampling_params=sampling_params,
            )
        )
        return [
            _parse_sampled_text(
                renderer=renderer,
                tokenizer=tokenizer,
                get_text_content=get_text_content,
                tokens=seq.tokens,
            )
            for seq in result.sequences
        ]

    return sample_fn


# ---- Eval scoring ----


async def score_run(
    *,
    label: str,
    sample_fn,
    boards: List[Dict[str, Any]],
    solver: PonsSolver,
    concurrency: int,
) -> Dict[str, Any]:
    """Score one named run on the eval set. Concurrent across boards."""
    per_board: List[Dict[str, Any]] = []

    for i in range(0, len(boards), concurrency):
        chunk = boards[i : i + concurrency]
        envs = [_env_from_moves(b["moves"]) for b in chunk]
        completions_per_board = await asyncio.gather(*[sample_fn(e) for e in envs])

        for board, env, completions in zip(chunk, envs, completions_per_board):
            per_sample = []
            cols_chosen: List[Optional[int]] = []
            for text in completions:
                col = parse_column(text)
                r, regret, opt, legal, valid = move_quality(env, col, solver)
                per_sample.append(
                    {
                        "reward": r,
                        "regret": regret,
                        "optimal": opt,
                        "legal": legal,
                        "valid": valid,
                    }
                )
                cols_chosen.append(col if (legal and col is not None) else None)

            n = len(per_sample) or 1
            mean_metrics = {
                "mean_reward": sum(s["reward"] for s in per_sample) / n,
                "mean_regret": sum(s["regret"] for s in per_sample) / n,
                "optimal_rate": sum(s["optimal"] for s in per_sample) / n,
                "legal_rate": sum(s["legal"] for s in per_sample) / n,
                "valid_rate": sum(s["valid"] for s in per_sample) / n,
            }
            valid_cols = [c for c in cols_chosen if c is not None]
            mode_share = (
                max(Counter(valid_cols).values()) / len(valid_cols)
                if valid_cols
                else 0.0
            )
            per_board.append(
                {
                    "moves": board["moves"],
                    "samples": len(per_sample),
                    "cols_chosen": cols_chosen,
                    "mode_share": mode_share,
                    **mean_metrics,
                }
            )
        logger.info("[%s] %d/%d boards", label, len(per_board), len(boards))

    n = len(per_board) or 1
    summary = {
        "label": label,
        "n_boards": len(per_board),
        "samples_per_board": per_board[0]["samples"] if per_board else 0,
        "mean_reward": sum(b["mean_reward"] for b in per_board) / n,
        "mean_regret": sum(b["mean_regret"] for b in per_board) / n,
        "optimal_rate": sum(b["optimal_rate"] for b in per_board) / n,
        "legal_rate": sum(b["legal_rate"] for b in per_board) / n,
        "valid_rate": sum(b["valid_rate"] for b in per_board) / n,
        "mean_per_board_mode_share": sum(b["mode_share"] for b in per_board) / n,
    }
    return {"summary": summary, "per_board": per_board}


# ---- CLI ----


async def _amain(args: argparse.Namespace) -> int:
    rows = [
        json.loads(l)
        for l in Path(args.eval_set).read_text().splitlines()
        if l.strip()
    ]
    if 0 < args.n_boards < len(rows):
        import random

        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.n_boards]
    logger.info("Eval set: %d boards", len(rows))

    if args.mode == "greedy":
        temperature = 0.0
        num_samples = 1
    else:
        temperature = args.temperature
        num_samples = args.n_samples

    solver = PonsSolver(strict=args.strict_solver)
    results = []
    for spec in args.run:
        label, ckpt = spec.split(":", 1)
        logger.info("Scoring %s (path=%s)", label, ckpt)
        sample_fn = await _build_sample_fn(
            checkpoint_path=ckpt,
            base_model=args.base_model,
            renderer_name=args.renderer,
            max_tokens=args.max_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )
        result = await score_run(
            label=label,
            sample_fn=sample_fn,
            boards=rows,
            solver=solver,
            concurrency=args.concurrency,
        )
        results.append(result)
        print(json.dumps(result["summary"], indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "n_boards": len(rows),
                "n_samples_per_board": num_samples,
                "temperature": temperature,
                "results": [r["summary"] for r in results],
                "per_board": {r["summary"]["label"]: r["per_board"] for r in results},
            },
            indent=2,
        )
    )
    logger.info("Wrote %s", out_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", required=True, help="JSONL of held-out boards.")
    parser.add_argument("--n-boards", type=int, default=0,
                        help="Subsample N boards. 0 = all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help='Pair "<label>:<checkpoint_path_or_BASE>" (repeatable).',
    )
    parser.add_argument("--mode", choices=("greedy", "stochastic"), default="greedy")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Samples per board (stochastic mode only).")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (stochastic mode only).")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--renderer", default="qwen3_instruct")
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
