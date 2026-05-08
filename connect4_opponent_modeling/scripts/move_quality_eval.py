"""Oracle move-quality eval: per-position, vLLM-batched, no minimax games.

For each Pons benchmark position, we ask the model for a move and score it
against the solver's optimal move (1.0 if the model's column matches, else
partial credit based on how much worse the chosen column is).

Designed as the primary in-domain metric for the Base/Value/State/Opp
comparison. Lower variance and ~5x more data points than ladder win rate.

Usage:
    python scripts/move_quality_eval.py --model Qwen/Qwen3-4B --condition E \\
        --out /tmp/mq_base_E.json
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.prompts import format_prompt, parse_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("mq_eval")


SET_NAMES = [
    ("Test_L3_R1", "end_easy"),
    ("Test_L2_R1", "end_hard"),
    ("Test_L2_R2", "middle_easy"),
    ("Test_L1_R1", "middle_hard"),
    ("Test_L1_R2", "beginning_easy"),
    ("Test_L1_R3", "beginning_hard"),
]


def _load_set(filepath: Path) -> List[Dict]:
    out = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                out.append({"moves": parts[0], "expected_score": int(parts[1])})
            elif len(parts) == 1:
                out.append({"moves": parts[0], "expected_score": None})
    return out


def _score_move(
    solver: PonsSolver, env: ConnectFourEnv, model_col: int
) -> float:
    """Oracle move quality in [0, 1].

    1.0 if model_col matches solver's best move. Otherwise partial credit
    based on the score gap between optimal and chosen.
    """
    if model_col is None or model_col not in env.legal_moves():
        return 0.0
    optimal = solver.best_move(env)
    if optimal == model_col:
        return 1.0
    scores = solver.analyze(env)
    if not scores:
        return 0.0
    vals = list(scores.values())
    s_max, s_min = max(vals), min(vals)
    if s_max == s_min:
        return 1.0
    chosen = scores.get(model_col, s_min)
    return max(0.0, min(1.0, (chosen - s_min) / (s_max - s_min)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--condition", default="E")
    ap.add_argument("--max_positions_per_set", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--benchmark_dir", default="data/pons_benchmark")
    ap.add_argument("--out", default="/tmp/move_quality.json")
    args = ap.parse_args()

    # Load all positions across sets
    bench = Path(args.benchmark_dir)
    position_set: List[Dict] = []
    for set_name, label in SET_NAMES:
        fp = None
        for ext in [".csv", ".txt", ""]:
            cand = bench / f"{set_name}{ext}"
            if cand.exists():
                fp = cand
                break
        if fp is None:
            logger.warning("Set %s not found in %s — skipping", set_name, bench)
            continue
        positions = _load_set(fp)[: args.max_positions_per_set]
        for p in positions:
            position_set.append({**p, "set": label})
    logger.info("Loaded %d positions across %d sets", len(position_set), len(SET_NAMES))

    # Build envs and prompts
    envs: List[ConnectFourEnv] = []
    prompts: List[str] = []
    keep_idx: List[int] = []
    for i, pos in enumerate(position_set):
        env = ConnectFourEnv()
        try:
            env.from_move_sequence([int(c) for c in pos["moves"]])
        except (ValueError, Exception):
            continue
        if env.is_terminal():
            continue
        envs.append(env)
        prompts.append(format_prompt(args.condition, env))
        keep_idx.append(i)
    logger.info("Kept %d non-terminal positions", len(envs))

    # Tokenizer + chat template
    tok = AutoTokenizer.from_pretrained(args.model)
    rendered = []
    for p in prompts:
        try:
            rendered.append(tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            ))
        except Exception:
            rendered.append(p)

    # Init vLLM
    logger.info("Initializing vLLM for %s ...", args.model)
    t0 = time.time()
    llm = LLM(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True, max_model_len=2048,
    )
    logger.info("vLLM ready in %.1fs", time.time() - t0)

    sp = SamplingParams(n=1, max_tokens=args.max_tokens, temperature=0.0)

    # Batched generation
    logger.info("Generating %d completions (batched)...", len(rendered))
    t0 = time.time()
    outs = llm.generate(rendered, sp)
    logger.info("Done in %.1fs", time.time() - t0)

    # Score
    solver = PonsSolver(strict=True)
    scores_by_set: Dict[str, List[float]] = {}
    invalid_by_set: Dict[str, int] = {}
    per_pos: List[Dict] = []
    for env, pos_idx, out in zip(envs, keep_idx, outs):
        text = out.outputs[0].text
        parsed = parse_response(text, args.condition)
        move = parsed.get("move")
        valid = move is not None and move in env.legal_moves()
        score = _score_move(solver, env, move) if valid else 0.0
        s = position_set[pos_idx]["set"]
        scores_by_set.setdefault(s, []).append(score)
        if not valid:
            invalid_by_set[s] = invalid_by_set.get(s, 0) + 1
        per_pos.append({
            "set": s,
            "moves": position_set[pos_idx]["moves"],
            "model_move": move,
            "valid": valid,
            "score": score,
        })

    by_set: Dict[str, Dict] = {}
    all_scores: List[float] = []
    for label in [lab for _, lab in SET_NAMES]:
        scores = scores_by_set.get(label, [])
        if not scores:
            continue
        by_set[label] = {
            "n": len(scores),
            "mean_oracle_score": sum(scores) / len(scores),
            "exact_match_rate": sum(1 for s in scores if s == 1.0) / len(scores),
            "invalid_outputs": invalid_by_set.get(label, 0),
        }
        all_scores.extend(scores)

    overall = {
        "n_total": len(all_scores),
        "mean_oracle_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "exact_match_rate": (sum(1 for s in all_scores if s == 1.0) / len(all_scores)) if all_scores else 0.0,
        "invalid_outputs": sum(invalid_by_set.values()),
    }

    out = {
        "model": args.model,
        "condition": args.condition,
        "max_positions_per_set": args.max_positions_per_set,
        "max_tokens": args.max_tokens,
        "by_set": by_set,
        "overall": overall,
        "per_position": per_pos,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    logger.info("=== Overall: mean=%.3f exact=%.3f invalid=%d/%d ===",
                overall["mean_oracle_score"], overall["exact_match_rate"],
                overall["invalid_outputs"], overall["n_total"])
    for label, st in by_set.items():
        logger.info("  %-15s n=%3d mean=%.3f exact=%.3f invalid=%d",
                    label, st["n"], st["mean_oracle_score"],
                    st["exact_match_rate"], st["invalid_outputs"])
    logger.info("Saved → %s", args.out)


if __name__ == "__main__":
    main()
