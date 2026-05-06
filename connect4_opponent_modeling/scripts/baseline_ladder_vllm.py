"""20-game ladder eval using vLLM with batched inference across games.

At each turn, all games where the model is to move have their prompts batched
into a single vllm.LLM.generate() call. Roughly 10x faster than the per-call
HF version when running 20 games × 3 depths.
"""
import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from vllm import LLM, SamplingParams

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.minimax import MinimaxSolver
from training.prompts import format_prompt, parse_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ladder_vllm")


def play_games_batched(
    llm: LLM,
    tokenizer,
    sampling: SamplingParams,
    opponent: MinimaxSolver,
    n_games: int,
    condition: str,
    seed: int,
) -> Dict:
    """Play n_games against opponent, batching all model moves per turn."""
    rng = random.Random(seed)
    envs: List[ConnectFourEnv] = [ConnectFourEnv() for _ in range(n_games)]
    # Alternate first move
    model_player = [1 if i % 2 == 0 else 2 for i in range(n_games)]
    invalid_moves = [0] * n_games
    done = [False] * n_games

    turn = 0
    while not all(done):
        turn += 1
        # Identify games where model is to move
        model_idxs = []
        prompts = []
        for i, env in enumerate(envs):
            if done[i]:
                continue
            if env.is_terminal():
                done[i] = True
                continue
            if env.current_player() == model_player[i]:
                model_idxs.append(i)
                prompts.append(format_prompt(condition, env))
            else:
                # Minimax to move
                move = opponent.best_move(env)
                env.make_move(move)
                if env.is_terminal():
                    done[i] = True

        # Batched model generation
        if prompts:
            # Apply chat template
            rendered = []
            for p in prompts:
                try:
                    rendered.append(tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False, add_generation_prompt=True,
                    ))
                except Exception:
                    rendered.append(p)
            outputs = llm.generate(rendered, sampling, use_tqdm=False)
            for idx, out in zip(model_idxs, outputs):
                env = envs[idx]
                response = out.outputs[0].text
                parsed = parse_response(response, condition)
                move = parsed.get("move")
                if move is None or move not in env.legal_moves():
                    invalid_moves[idx] += 1
                    move = rng.choice(env.legal_moves())
                env.make_move(move)
                if env.is_terminal():
                    done[idx] = True

        if turn % 5 == 0:
            n_done = sum(done)
            logger.info("  turn %d: %d/%d games complete", turn, n_done, n_games)

    # Tally
    wins = 0
    invalid_games = 0
    for i, env in enumerate(envs):
        winner = env.winner()
        if winner == model_player[i]:
            wins += 1
        if invalid_moves[i] > 0:
            invalid_games += 1
    return {
        "win_rate": wins / n_games,
        "wins": wins,
        "invalid_games": invalid_games,
        "invalid_game_rate": invalid_games / n_games,
        "invalid_moves_total": sum(invalid_moves),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--condition", default="E")
    parser.add_argument("--n_games", type=int, default=20)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=160)
    parser.add_argument("--gpu_mem_util", type=float, default=0.85)
    parser.add_argument("--out", default="/tmp/ladder_result.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    logger.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    logger.info("Initializing vLLM engine for %s ...", args.model)
    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True,
        max_model_len=1024,
    )
    logger.info("vLLM ready in %.1fs", time.time() - t0)

    sampling = SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        temperature=0.0,  # greedy for deterministic eval
    )

    win_rates: Dict[int, Dict] = {}
    t_total = time.time()
    for depth in args.depths:
        logger.info("=== Minimax depth %d, %d games ===", depth, args.n_games)
        t0 = time.time()
        opponent = MinimaxSolver(depth=depth)
        result = play_games_batched(
            llm, tokenizer, sampling, opponent,
            n_games=args.n_games, condition=args.condition, seed=args.seed,
        )
        elapsed = time.time() - t0
        result["elapsed_s"] = elapsed
        win_rates[depth] = result
        logger.info(
            "  depth %d done in %.1fs: win_rate=%.2f (%d/%d) invalid_games=%d",
            depth, elapsed, result["win_rate"], result["wins"], args.n_games,
            result["invalid_games"],
        )

    total = time.time() - t_total
    out = {
        "model": args.model,
        "condition": args.condition,
        "n_games": args.n_games,
        "depths": args.depths,
        "win_rates": win_rates,
        "elapsed_total_s": total,
        "elapsed_total_min": total / 60,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    logger.info("=== TOTAL %.1fs (%.1f min) → %s ===", total, total / 60, args.out)


if __name__ == "__main__":
    main()
