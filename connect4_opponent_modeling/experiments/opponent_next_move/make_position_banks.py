"""Create locked Connect Four evaluation banks for the narrow experiment.

The primary metric is Pons-normalized move quality, so each position bank is a
set of move sequences that can be replayed into ConnectFourEnv and scored by
Pons. The IID bank samples ordinary random/minimax-mixed positions. The hard
bank keeps positions where legal moves have meaningfully different oracle
scores, making the per-move metric more sensitive.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver


def _sample_position(rng: random.Random, solver: PonsSolver) -> Optional[ConnectFourEnv]:
    env = ConnectFourEnv()
    target_len = rng.randint(4, 30)

    for _ in range(target_len):
        if env.is_terminal() or not env.legal_moves():
            break
        if rng.random() < 0.35:
            move = solver.best_move(env)
        else:
            move = rng.choice(env.legal_moves())
        env.make_move(move)

    if env.is_terminal() or len(env.legal_moves()) < 2:
        return None
    return env


def _record_for(env: ConnectFourEnv, solver: PonsSolver, split: str) -> Optional[Dict]:
    scores = solver.analyze(env)
    if len(scores) < 2:
        return None

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    spread = max_score - min_score
    best_moves = [col for col, score in scores.items() if score == max_score]

    return {
        "split": split,
        "moves": env.to_move_sequence(),
        "move_count": len(env.to_move_sequence()),
        "legal_moves": sorted(scores.keys()),
        "scores": {str(col): score for col, score in sorted(scores.items())},
        "best_moves": best_moves,
        "score_spread": spread,
    }


def build_banks(
    output: Path,
    *,
    n_iid: int,
    n_hard: int,
    seed: int,
    hard_min_spread: int,
    max_attempts: int,
) -> Dict[str, int]:
    solver = PonsSolver(strict=True)
    if not solver.is_available():
        raise RuntimeError("Pons solver is required. Run scripts/bootstrap_gpu.sh first.")

    rng = random.Random(seed)
    seen = set()
    iid: List[Dict] = []
    hard: List[Dict] = []

    attempts = 0
    while (len(iid) < n_iid or len(hard) < n_hard) and attempts < max_attempts:
        attempts += 1
        env = _sample_position(rng, solver)
        if env is None:
            continue

        moves = env.to_move_sequence()
        if moves in seen:
            continue
        seen.add(moves)

        rec = _record_for(env, solver, "iid")
        if rec is None:
            continue

        if len(iid) < n_iid:
            iid.append(rec)

        if len(hard) < n_hard and rec["score_spread"] >= hard_min_spread:
            hard_rec = dict(rec)
            hard_rec["split"] = "hard"
            hard.append(hard_rec)

    if len(iid) < n_iid or len(hard) < n_hard:
        raise RuntimeError(
            f"Only built iid={len(iid)}/{n_iid}, hard={len(hard)}/{n_hard} "
            f"after {attempts} attempts. Try lowering --hard_min_spread or raising --max_attempts."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"{output} already exists. Delete it manually if you intentionally want a new locked bank."
        )

    with output.open("w") as handle:
        for rec in iid + hard:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")

    return {"iid": len(iid), "hard": len(hard), "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock narrow-experiment position banks.")
    parser.add_argument(
        "--output",
        default="experiments/opponent_next_move/data/connect4_eval_banks.jsonl",
    )
    parser.add_argument("--n_iid", type=int, default=500)
    parser.add_argument("--n_hard", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--hard_min_spread", type=int, default=2)
    parser.add_argument("--max_attempts", type=int, default=20000)
    args = parser.parse_args()

    counts = build_banks(
        Path(args.output),
        n_iid=args.n_iid,
        n_hard=args.n_hard,
        seed=args.seed,
        hard_min_spread=args.hard_min_spread,
        max_attempts=args.max_attempts,
    )
    print(json.dumps({"output": args.output, **counts}, indent=2))


if __name__ == "__main__":
    main()
