"""Multi-seed, multi-condition ladder eval. Reuses one vLLM engine.

Default protocol: conditions A/C/E/F, seeds 42/43/44, 50 games × minimax 1/2/4.
Identical max_tokens / sampling / chat-template across all (condition, seed).
"""
import argparse
import json
import logging
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from training.minimax import MinimaxSolver

# Reuse the play_games_batched implementation
import sys
sys.path.insert(0, str(Path(__file__).parent))
from baseline_ladder_vllm import play_games_batched  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ladder_multi")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--conditions", nargs="+", default=["A", "C", "E", "F"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--n_games", type=int, default=50)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--gpu_mem_util", type=float, default=0.85)
    parser.add_argument("--out", default="/tmp/ladder_multi.json")
    args = parser.parse_args()

    logger.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    logger.info("Initializing vLLM engine for %s ...", args.model)
    t0 = time.time()
    llm = LLM(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True, max_model_len=2048,
    )
    logger.info("vLLM ready in %.1fs", time.time() - t0)

    sampling = SamplingParams(n=1, max_tokens=args.max_tokens, temperature=0.0)

    results: Dict = {
        "model": args.model,
        "conditions": args.conditions,
        "seeds": args.seeds,
        "n_games": args.n_games,
        "depths": args.depths,
        "max_tokens": args.max_tokens,
        "runs": [],  # list of {condition, seed, depth, win_rate, ...}
    }

    t_total = time.time()
    for cond in args.conditions:
        for seed in args.seeds:
            for depth in args.depths:
                logger.info("=== cond=%s seed=%d depth=%d ===", cond, seed, depth)
                t0 = time.time()
                opponent = MinimaxSolver(depth=depth)
                r = play_games_batched(
                    llm, tokenizer, sampling, opponent,
                    n_games=args.n_games, condition=cond, seed=seed,
                )
                elapsed = time.time() - t0
                run = {
                    "condition": cond, "seed": seed, "depth": depth,
                    "win_rate": r["win_rate"], "wins": r["wins"],
                    "invalid_games": r["invalid_games"],
                    "invalid_moves_total": r["invalid_moves_total"],
                    "elapsed_s": elapsed,
                }
                results["runs"].append(run)
                logger.info(
                    "  cond=%s seed=%d depth=%d: win=%.2f (%d/%d) inv_games=%d (%.1fs)",
                    cond, seed, depth, r["win_rate"], r["wins"], args.n_games,
                    r["invalid_games"], elapsed,
                )

    # Aggregate by (condition, depth) across seeds
    agg: Dict[str, Dict[int, Dict]] = {}
    for cond in args.conditions:
        agg[cond] = {}
        for depth in args.depths:
            wrs = [r["win_rate"] for r in results["runs"]
                   if r["condition"] == cond and r["depth"] == depth]
            invs = [r["invalid_games"] for r in results["runs"]
                    if r["condition"] == cond and r["depth"] == depth]
            agg[cond][depth] = {
                "mean_win_rate": mean(wrs) if wrs else 0.0,
                "stdev_win_rate": stdev(wrs) if len(wrs) > 1 else 0.0,
                "n_seeds": len(wrs),
                "mean_invalid_games": mean(invs) if invs else 0.0,
            }
    results["aggregate"] = agg
    results["elapsed_total_s"] = time.time() - t_total
    results["elapsed_total_min"] = results["elapsed_total_s"] / 60

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    logger.info("=== TOTAL %.1f min → %s ===",
                results["elapsed_total_min"], args.out)
    logger.info("=== Aggregate (mean ± stdev across %d seeds) ===", len(args.seeds))
    for cond in args.conditions:
        for depth in args.depths:
            a = agg[cond][depth]
            logger.info("  %s d%d: %.2f ± %.2f (inv: %.1f)",
                        cond, depth, a["mean_win_rate"], a["stdev_win_rate"],
                        a["mean_invalid_games"])


if __name__ == "__main__":
    main()
