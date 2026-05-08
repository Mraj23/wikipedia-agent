"""Large-scale training-position pool for the faithfulness experiment.

This is a *training* dataset, distinct from the locked eval set:

- Eval set (`eval_boards.jsonl`): small (~250-500 positions), needs Pons
  scores cached in verifier_meta, immutable.
- Training pool (`training_positions.jsonl`): large (10K-100K+), pure
  deterministic rule-based metadata (no Pons), regenerable.

Training-pool records contain only what can be derived without the solver:
move sequence, position-level strategic tag, per-move strategic tags, legal
columns, current player. The RL trainer can sample positions from this pool
and call Pons at training time for the regret reward; pre-computing Pons
scores for tens of thousands of positions is wasteful when the trainer
needs them anyway as it generates rollouts.

Schema per JSONL row:
    {
      "moves": "33245...",
      "position_tag": "must_block_threat",
      "current_player": 1,
      "legal_moves": [0, 1, 2, 4, 5, 6],
      "move_tags": {"0": "neutral", "4": "block_immediate_threat", ...},
      "ply": 6
    }

The pool is stratified by `position_tag` so RL training can curriculum or
upweight tactically charged positions.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from env.connect_four_env import ConnectFourEnv
from faithfulness.strategic_moves import (
    PositionTag,
    analyze_position,
    classify_position,
)

logger = logging.getLogger(__name__)


def _record_for_env(env: ConnectFourEnv, moves_str: str) -> Optional[Dict]:
    if env.is_terminal():
        return None
    legal = env.legal_moves()
    if not legal:
        return None
    pos_tag = classify_position(env)
    move_tags = {str(c): a.tag.value for c, a in analyze_position(env).items()}
    return {
        "moves": moves_str,
        "position_tag": pos_tag.value,
        "current_player": env.current_player(),
        "legal_moves": list(legal),
        "move_tags": move_tags,
        "ply": len(moves_str),
    }


def generate_training_pool(
    *,
    n_positions: int = 50_000,
    seed: int = 42,
    max_games: Optional[int] = None,
    dedup: bool = True,
    min_ply: int = 1,
    max_ply: int = 41,
    target_per_tag: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """Generate a flat (but stratified-aware) training pool of positions.

    Args:
        n_positions: Target number of positions. Generation stops early when
            this is reached or `max_games` is exhausted.
        seed: RNG seed for reproducibility.
        max_games: Hard cap on random self-play games. Defaults to
            8 * n_positions / 30 ≈ enough plies on average.
        dedup: If True, drop duplicate positions by move sequence.
        min_ply: Skip positions before this ply (avoid trivial empty boards).
        max_ply: Skip positions past this ply (avoid forced endgames).
        target_per_tag: Optional per-PositionTag soft cap. When supplied,
            stop accepting a tag once it reaches its cap. Useful for
            balancing rare tags. None = no caps.

    Returns:
        List of records.
    """
    rng = random.Random(seed)
    n_games = max_games if max_games is not None else max(1000, n_positions // 4)

    seen: set = set()
    by_tag: Counter = Counter()
    records: List[Dict] = []
    games_played = 0

    while games_played < n_games and len(records) < n_positions:
        env = ConnectFourEnv()
        moves: List[int] = []
        while not env.is_terminal():
            legal = env.legal_moves()
            move = rng.choice(legal)
            env.make_move(move)
            moves.append(move)

            if env.is_terminal() or len(env.legal_moves()) <= 1:
                continue
            ply = len(moves)
            if ply < min_ply or ply > max_ply:
                continue

            moves_str = "".join(str(m) for m in moves)
            if dedup and moves_str in seen:
                continue

            rec = _record_for_env(env, moves_str)
            if rec is None:
                continue

            tag = rec["position_tag"]
            if target_per_tag is not None and by_tag[tag] >= target_per_tag.get(tag, 0):
                continue

            if dedup:
                seen.add(moves_str)
            by_tag[tag] += 1
            records.append(rec)

            if len(records) >= n_positions:
                break
        games_played += 1

    logger.info(
        "Training pool generated: %d positions across %d games (tags: %s)",
        len(records),
        games_played,
        dict(by_tag),
    )
    return records


def write_training_pool(records: Iterable[Dict], output_path: str) -> None:
    """Write records to JSONL. Overwrites existing file (training data is
    regenerable, unlike the locked eval set)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def load_training_pool(path: str) -> List[Dict]:
    out: List[Dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def env_from_training_record(rec: Dict) -> ConnectFourEnv:
    env = ConnectFourEnv()
    env.from_move_sequence([int(ch) for ch in rec["moves"]])
    return env


def stratify_summary(records: List[Dict]) -> Dict[str, int]:
    return dict(Counter(r["position_tag"] for r in records))
