"""Generate a Connect 4 training pool for v2.

Two modes:

  random   (default)  Stratified random self-play, ply-balanced, filtered by
                      Pons solver-spread. NO Tinker calls. Fast, cheap,
                      reproducible by seed. This is the headline training
                      distribution for v2.

  entropy  (opt-in)   Random mode pipeline + a base-model column-entropy
                      filter on top: drop positions where any single column
                      gets >`--max-column-share` of N base completions.
                      Tinker required. Diagnostic / sensitivity check, not
                      the headline distribution.

A position is kept iff:
  - the solver shows non-trivial regret spread across legal moves
    (max - min Pons score >= --min-solver-spread); AND
  - in entropy mode, the base model's column distribution over N samples
    has no single column above `--max-column-share`.

Output JSONL fields per row (always):
  moves           list[int]  move history from the initial state
  current_player  int        1 or 2 (whose turn at this position)
  legal_moves     list[int]
  ply             int        len(moves)
  stratum         str        "early" | "mid" | "late"
  solver_spread   int        max-min Pons score over legal columns
  solver_scores   dict[str,int]  Pons score per legal column

Entropy mode adds:
  base_dist       dict[str,int]  parsed column → count over N base samples
  base_max_share  float          max count / total parseable
  n_parseable     int            samples that parsed to a legal column

Usage (random mode, no Tinker):
    python -m faithfulness_v2.generate_pool \\
        --output faithfulness_v2/data/pool_v2.jsonl \\
        --target 5000

Usage (entropy diagnostic, Tinker required):
    TINKER_API_KEY=... python -m faithfulness_v2.generate_pool \\
        --mode entropy \\
        --output faithfulness_v2/data/pool_v2_entropy.jsonl \\
        --target 5000 --candidates-per-target 4 --max-column-share 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt + parse. Must stay byte-identical to train_move_only.py and
# eval_move_quality.py — the prompt-consistency test enforces this.
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
# Stratified self-play
# ---------------------------------------------------------------------------

DEFAULT_STRATA: Tuple[Tuple[str, int, int], ...] = (
    ("early", 4, 12),
    ("mid", 13, 22),
    ("late", 23, 35),
)


def _env_from_moves(moves: Sequence[int]) -> ConnectFourEnv:
    env = ConnectFourEnv()
    for m in moves:
        env.make_move(int(m))
    return env


def _stratum_for_ply(
    ply: int, strata: Sequence[Tuple[str, int, int]]
) -> Optional[str]:
    for name, lo, hi in strata:
        if lo <= ply <= hi:
            return name
    return None


def _stratified_self_play(
    n_target: int,
    rng: random.Random,
    *,
    strata: Sequence[Tuple[str, int, int]] = DEFAULT_STRATA,
    min_moves_remaining: int = 2,
) -> List[Tuple[List[int], str]]:
    """Random self-play with per-stratum caps. Returns (history, stratum_name).

    Caps each stratum at ceil(n_target / len(strata)) so the output is
    ply-balanced. Snapshots all eligible mid-game positions during a game,
    not one per game — much cheaper.
    """
    per_stratum_cap = (n_target + len(strata) - 1) // len(strata)
    counts: Dict[str, int] = {name: 0 for name, _, _ in strata}
    out: List[Tuple[List[int], str]] = []

    safety_games = 0
    while sum(counts.values()) < n_target and safety_games < n_target * 50:
        safety_games += 1
        env = ConnectFourEnv()
        history: List[int] = []
        while not env.is_terminal():
            legal = env.legal_moves()
            if not legal:
                break
            move = rng.choice(legal)
            env.make_move(move)
            history.append(move)
            if env.is_terminal():
                break
            n_remaining = 42 - len(history)
            if (
                len(env.legal_moves()) < 2
                or n_remaining < min_moves_remaining
            ):
                continue
            stratum = _stratum_for_ply(len(history), strata)
            if stratum is None:
                continue
            if counts[stratum] >= per_stratum_cap:
                continue
            out.append((list(history), stratum))
            counts[stratum] += 1
            if sum(counts.values()) >= n_target:
                break

    if sum(counts.values()) < n_target:
        logger.warning(
            "Self-play exhausted before hitting target: kept=%d target=%d counts=%s",
            sum(counts.values()),
            n_target,
            counts,
        )
    return out


# ---------------------------------------------------------------------------
# Solver-spread filter (random + entropy modes)
# ---------------------------------------------------------------------------


def _solver_spread_record(
    history: Sequence[int],
    stratum: str,
    solver: PonsSolver,
    *,
    min_solver_spread: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    env = _env_from_moves(history)
    legal = env.legal_moves()
    scores = solver.analyze(env)
    if not scores:
        return None, "solver_no_scores"
    spread = max(scores.values()) - min(scores.values())
    if spread < min_solver_spread:
        return None, "solver_no_spread"
    return (
        {
            "moves": list(history),
            "current_player": env.current_player(),
            "legal_moves": list(legal),
            "ply": len(history),
            "stratum": stratum,
            "solver_spread": int(spread),
            "solver_scores": {str(k): int(v) for k, v in scores.items()},
        },
        "kept",
    )


# ---------------------------------------------------------------------------
# Entropy filter (entropy mode only)
# ---------------------------------------------------------------------------


def _entropy_overlay(
    env: ConnectFourEnv,
    completions: Sequence[str],
    *,
    max_column_share: float,
    min_parseable: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    legal = env.legal_moves()
    cols = [parse_column(c) for c in completions]
    cols = [c for c in cols if c is not None and c in legal]
    if len(cols) < min_parseable:
        return None, "few_parseable"
    counts = Counter(cols)
    max_share = max(counts.values()) / len(cols)
    if max_share > max_column_share:
        return None, "single_column_dominant"
    return (
        {
            "base_dist": {str(k): v for k, v in counts.items()},
            "base_max_share": max_share,
            "n_parseable": len(cols),
        },
        "kept",
    )


# ---------------------------------------------------------------------------
# Tinker scaffolding (entropy mode only)
# ---------------------------------------------------------------------------


def _resolve_tinker_value(value: Any) -> Any:
    result = getattr(value, "result", None)
    if callable(result):
        return result()
    return value


async def _resolve_async(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return _resolve_tinker_value(value)


async def _build_base_sample_fn(
    base_model: str,
    renderer_name: str,
    max_tokens: int,
    temperature: float,
    lora_rank: int,
) -> Callable[[ConnectFourEnv, int], Awaitable[List[str]]]:
    import tinker  # type: ignore[import-not-found]
    from tinker_cookbook.renderers import get_renderer, get_text_content

    service_client = tinker.ServiceClient()
    training_client = await _resolve_async(
        service_client.create_lora_training_client_async(
            base_model=base_model, rank=lora_rank
        )
    )
    save_result = await _resolve_async(
        training_client.save_weights_for_sampler_async(
            name="v2-pool-base", ttl_seconds=3600
        )
    )
    sampling_client = await _resolve_async(
        service_client.create_sampling_client_async(model_path=save_result.path)
    )
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer, model_name=base_model)
    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
    )

    async def sample_fn(env: ConnectFourEnv, num_samples: int) -> List[str]:
        prompt = renderer.build_generation_prompt(make_messages(env))
        result = await _resolve_async(
            sampling_client.sample_async(
                prompt=prompt,
                num_samples=num_samples,
                sampling_params=sampling_params,
            )
        )
        completions: List[str] = []
        for seq in result.sequences:
            try:
                msg, ok = renderer.parse_response(list(seq.tokens))
                if ok:
                    completions.append(str(get_text_content(msg)))
                    continue
            except Exception:
                pass
            completions.append(tokenizer.decode(list(seq.tokens), skip_special_tokens=True))
        return completions

    return sample_fn


async def _score_in_chunks(
    candidates: List[Dict[str, Any]],
    sample_fn: Callable[[ConnectFourEnv, int], Awaitable[List[str]]],
    candidates_per_target: int,
    concurrency: int,
) -> List[Tuple[Dict[str, Any], List[str]]]:
    out: List[Tuple[Dict[str, Any], List[str]]] = []
    for i in range(0, len(candidates), concurrency):
        chunk = candidates[i : i + concurrency]
        envs = [_env_from_moves(c["moves"]) for c in chunk]
        completions = await asyncio.gather(
            *[sample_fn(env, candidates_per_target) for env in envs]
        )
        out.extend(zip(chunk, completions))
        logger.info("Scored %d/%d candidates", len(out), len(candidates))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _amain(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)

    over_factor = max(2.0, 1.0 / max(args.expected_keep_rate, 0.05))
    n_candidates = int(args.target * over_factor)
    logger.info(
        "Stratified self-play: target=%d, oversample=%d (expected_keep=%.2f, factor=%.2f)",
        args.target,
        n_candidates,
        args.expected_keep_rate,
        over_factor,
    )
    histories = _stratified_self_play(
        n_candidates,
        rng,
        min_moves_remaining=args.min_moves_remaining,
    )
    logger.info("Generated %d candidate positions across strata", len(histories))

    solver = PonsSolver(strict=args.strict_solver)
    if args.strict_solver and not solver.is_available():
        raise RuntimeError(
            "Pons solver binary is required (--strict-solver). "
            "Run scripts/bootstrap_gpu.sh or pass --no-strict-solver for dev only."
        )

    spread_records: List[Dict[str, Any]] = []
    drop_reasons: Counter = Counter()
    per_stratum_kept: Counter = Counter()
    for history, stratum in histories:
        record, reason = _solver_spread_record(
            history,
            stratum,
            solver,
            min_solver_spread=args.min_solver_spread,
        )
        if record is None:
            drop_reasons[reason] += 1
            continue
        spread_records.append(record)
        per_stratum_kept[stratum] += 1
        if len(spread_records) >= args.target and args.mode == "random":
            break

    logger.info(
        "After solver-spread filter: kept=%d dropped=%s per_stratum=%s",
        len(spread_records),
        dict(drop_reasons),
        dict(per_stratum_kept),
    )

    final_records: List[Dict[str, Any]] = []

    if args.mode == "random":
        final_records = spread_records[: args.target]
    else:
        # Entropy mode: layer base-model column-entropy filter on top.
        sample_fn = await _build_base_sample_fn(
            base_model=args.base_model,
            renderer_name=args.renderer,
            max_tokens=args.sample_max_tokens,
            temperature=args.sample_temperature,
            lora_rank=args.lora_rank,
        )
        scored = await _score_in_chunks(
            spread_records,
            sample_fn,
            candidates_per_target=args.candidates_per_target,
            concurrency=args.concurrency,
        )
        for record, completions in scored:
            env = _env_from_moves(record["moves"])
            overlay, reason = _entropy_overlay(
                env,
                completions,
                max_column_share=args.max_column_share,
                min_parseable=args.min_parseable,
            )
            if overlay is None:
                drop_reasons[reason] += 1
                continue
            final_records.append({**record, **overlay})
            if len(final_records) >= args.target:
                break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in final_records:
            fh.write(json.dumps(r) + "\n")

    final_per_stratum: Counter = Counter(r["stratum"] for r in final_records)
    summary = {
        "mode": args.mode,
        "kept": len(final_records),
        "candidates_generated": len(histories),
        "spread_filter_kept": len(spread_records),
        "drop_reasons": dict(drop_reasons),
        "per_stratum_kept": dict(final_per_stratum),
        "per_stratum_after_spread": dict(per_stratum_kept),
        "config": {k: v for k, v in vars(args).items()},
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Wrote %d positions to %s", len(final_records), out_path)
    logger.info("Summary: %s", summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument(
        "--mode",
        choices=("random", "entropy"),
        default="random",
        help="random = stratified self-play + solver-spread (default, no Tinker). "
             "entropy = random + base-model entropy filter (Tinker required, diagnostic).",
    )

    # Filters used by both modes.
    parser.add_argument(
        "--min-solver-spread",
        type=int,
        default=2,
        help="Min (max-min) Pons score across legal moves to keep position.",
    )
    parser.add_argument("--min-moves-remaining", type=int, default=2)
    parser.add_argument(
        "--expected-keep-rate",
        type=float,
        default=0.4,
        help="Used to oversample candidates so we hit target.",
    )
    parser.add_argument("--seed", type=int, default=0)
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

    # Entropy-mode-only knobs.
    entropy_grp = parser.add_argument_group("entropy mode (--mode entropy)")
    entropy_grp.add_argument(
        "--candidates-per-target",
        type=int,
        default=4,
        help="Base completions per candidate position.",
    )
    entropy_grp.add_argument(
        "--max-column-share",
        type=float,
        default=0.5,
        help="Reject positions where one column gets > this share of base samples.",
    )
    entropy_grp.add_argument(
        "--min-parseable",
        type=int,
        default=2,
        help="Min number of base completions that parse to a legal column.",
    )
    entropy_grp.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    entropy_grp.add_argument("--renderer", default="qwen3_instruct")
    entropy_grp.add_argument("--lora-rank", type=int, default=1)
    entropy_grp.add_argument("--sample-max-tokens", type=int, default=16)
    entropy_grp.add_argument("--sample-temperature", type=float, default=0.7)
    entropy_grp.add_argument("--concurrency", type=int, default=16)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
