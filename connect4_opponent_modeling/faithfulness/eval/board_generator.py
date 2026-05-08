"""Stratified eval-set generator for the faithfulness experiment.

Generates and locks an immutable held-out set of Connect Four positions,
stratified into five tactical categories that exercise different claim
types:

- immediate_win_available     — current player has a one-move win.
- opponent_immediate_threat   — opponent has an immediate winning move
                                that the current player must address.
- blunder_state               — exactly one move avoids losing immediately
                                or substantially.
- no_immediate_tactic         — no immediate tactic; quiet midgame-ish.
- hard_midgame                — quiet but deep (>=20 pieces); rewards
                                position-evaluation rather than spotting
                                tactics.

Storage convention: jsonl, one record per row, with the move sequence
serialization used elsewhere in this repo (see env/connect_four_env.py and
data/probe_positions_locked.jsonl). Each record:

    {"moves": "33245...", "category": "blunder_state",
     "verifier_meta": {"pons_scores": {0: -3, 1: 5, ...}, "best_move": 1}}

verifier_meta is cached at lock time so eval reproduces identically even
if Pons becomes unavailable.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.strategic_moves import PositionTag, classify_position

logger = logging.getLogger(__name__)

CATEGORIES = (
    "immediate_win_available",
    "opponent_immediate_threat",
    "blunder_state",
    "no_immediate_tactic",
    "hard_midgame",
)

# Positions whose category is decided by board rules alone — no Pons call
# needed for these. The solver is still called once per accepted position to
# fill verifier_meta (see _build_record), but never on rejected candidates.
_TAG_TO_CATEGORY = {
    PositionTag.HAS_IMMEDIATE_WIN: "immediate_win_available",
    PositionTag.MUST_BLOCK_THREAT: "opponent_immediate_threat",
}


def _quiet_category_from_scores(
    env: ConnectFourEnv, scores: Dict[int, int]
) -> Optional[str]:
    """Decide between blunder_state / no_immediate_tactic / hard_midgame
    given an already-computed score dict. Pure logic; no solver call.
    """
    if not scores:
        return None
    best_value = max(scores.values())
    non_losing = [c for c, v in scores.items() if v >= 0]
    matches_best = [c for c, v in scores.items() if v == best_value]
    if best_value > 0 and len(matches_best) == 1:
        return "blunder_state"
    if best_value == 0 and len(non_losing) == 1:
        return "blunder_state"

    pieces = len(env.to_move_sequence())
    if pieces >= 20:
        return "hard_midgame"
    return "no_immediate_tactic"


def _quiet_category(env: ConnectFourEnv, solver: PonsSolver) -> Optional[str]:
    """Decide between blunder_state / no_immediate_tactic / hard_midgame.

    Calls solver.analyze; for the hot path use _quiet_category_from_scores
    to share one solver call across categorization and verifier_meta.
    """
    scores = solver.analyze(env)
    return _quiet_category_from_scores(env, scores)


def _categorize_cheap(env: ConnectFourEnv) -> Optional[str]:
    """Cheap (no-solver) category decision. Returns None if a solver call is
    needed to disambiguate.
    """
    if env.is_terminal() or not env.legal_moves():
        return None
    tag = classify_position(env)
    if tag in _TAG_TO_CATEGORY:
        return _TAG_TO_CATEGORY[tag]
    return None  # one of the quiet categories — needs the solver


def _seeded_position_pool(seed: int, n_games: int) -> List[str]:
    """Generate candidate move sequences via random self-play.

    Returns move sequences (string of digits 0-6). Mirrors the stylistic
    intent of eval.probe._generate_diverse_positions but uses a pure-random
    rollout to avoid pulling in the minimax solver here. Diversity comes
    from variety of random rollouts and the long candidate pool.
    """
    rng = random.Random(seed)
    sequences: List[str] = []
    for _ in range(n_games):
        env = ConnectFourEnv()
        moves: List[int] = []
        while not env.is_terminal():
            legal = env.legal_moves()
            move = rng.choice(legal)
            env.make_move(move)
            moves.append(move)
            if not env.is_terminal() and len(env.legal_moves()) > 1:
                sequences.append("".join(str(m) for m in moves))
    return sequences


def generate_stratified_eval_set(
    n_per_category: int = 100,
    *,
    seed: int = 42,
    solver: Optional[PonsSolver] = None,
    candidate_games: int = 1500,
    max_attempts: Optional[int] = None,
) -> List[Dict]:
    """Generate a stratified eval set.

    Args:
        n_per_category: Target number of positions per category.
        seed: RNG seed for reproducibility.
        solver: Pons solver (strict mode in production). Created if None.
        candidate_games: Number of random self-play games to seed the pool.
        max_attempts: Hard cap on candidate iterations (defaults to
            ~50x candidate_games via the pool size).

    Returns:
        List of records, one per locked position.
    """
    if solver is None:
        solver = PonsSolver(strict=True)

    pool = _seeded_position_pool(seed, candidate_games)
    rng = random.Random(seed)
    rng.shuffle(pool)

    buckets: Dict[str, List[Dict]] = {c: [] for c in CATEGORIES}
    target = n_per_category
    iterations = 0
    cap = max_attempts or len(pool)
    # Categories whose acceptance does not depend on a solver call; we never
    # invoke Pons before the position is accepted into one of these.
    cheap_categories = set(_TAG_TO_CATEGORY.values())

    for moves_str in pool:
        if iterations >= cap:
            break
        iterations += 1
        if all(len(buckets[c]) >= target for c in CATEGORIES):
            break
        env = ConnectFourEnv()
        try:
            env.from_move_sequence([int(ch) for ch in moves_str])
        except Exception:
            continue

        cheap = _categorize_cheap(env)
        scores: Optional[Dict[int, int]] = None
        if cheap is not None:
            if len(buckets[cheap]) >= target:
                continue
            cat = cheap
        else:
            # Skip the expensive solver call when every quiet bucket is full.
            quiet_full = all(
                len(buckets[c]) >= target
                for c in CATEGORIES
                if c not in cheap_categories
            )
            if quiet_full:
                continue
            scores = solver.analyze(env)
            cat = _quiet_category_from_scores(env, scores)
            if cat is None or len(buckets[cat]) >= target:
                continue

        # One solver call per accepted position to populate verifier_meta —
        # reuse the call from _quiet_category when we already have it.
        if scores is None:
            scores = solver.analyze(env)
        if not scores:
            continue
        best_value = max(scores.values())
        best_moves = sorted(c for c, v in scores.items() if v == best_value)
        record = {
            "moves": moves_str,
            "category": cat,
            "verifier_meta": {
                "pons_scores": {str(k): int(v) for k, v in scores.items()},
                "best_value": int(best_value),
                "best_moves": best_moves,
                "current_player": env.current_player(),
            },
        }
        buckets[cat].append(record)

    out: List[Dict] = []
    for cat in CATEGORIES:
        if len(buckets[cat]) < target:
            logger.warning(
                "Eval set under-filled for category %s: got %d of %d",
                cat,
                len(buckets[cat]),
                target,
            )
        out.extend(buckets[cat][:target])
    return out


def lock_eval_set(
    output_path: str,
    *,
    n_per_category: int = 100,
    seed: int = 42,
    solver: Optional[PonsSolver] = None,
    candidate_games: int = 1500,
) -> None:
    """Write a stratified eval set to disk. Refuses to overwrite.

    Mirrors eval.probe.lock_probe_positions in spirit.
    """
    out = Path(output_path)
    if out.exists():
        raise FileExistsError(
            f"Faithfulness eval set already locked at {output_path}. "
            "This file must NEVER be regenerated. Delete it manually only if "
            "you also document why in the commit message."
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    records = generate_stratified_eval_set(
        n_per_category=n_per_category,
        seed=seed,
        solver=solver,
        candidate_games=candidate_games,
    )
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Locked %d positions to %s", len(records), output_path)


def load_eval_set(path: str) -> List[Dict]:
    out: List[Dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def env_from_record(record: Dict) -> ConnectFourEnv:
    env = ConnectFourEnv()
    env.from_move_sequence([int(ch) for ch in record["moves"]])
    return env
