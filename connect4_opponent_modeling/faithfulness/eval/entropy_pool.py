"""Entropy-filtered training-position pool generation.

The tactical pool answers "does the solver have a clear preference?"  This
module answers the more useful GRPO question: "does the base model sample
multiple moves on this board while the solver says those moves matter?"
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.eval.board_generator import _seeded_position_pool
from faithfulness.eval.training_pool import _record_for_env, generate_training_pool
from faithfulness.parse import parse_structured_response
from faithfulness.prompt import make_messages
from faithfulness.verifier.move_evaluator import (
    REGRET_CLIP_DEFAULT,
    REGRET_SCALE_DEFAULT,
)

logger = logging.getLogger(__name__)

SampleFn = Callable[[List[dict], int], List[str]]


def shannon_entropy(distribution: Dict[str, float]) -> float:
    """Return Shannon entropy in bits for a normalized distribution."""
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in distribution.values():
        if value <= 0:
            continue
        p = value / total
        entropy -= p * math.log2(p)
    return entropy


def solver_score_spread(
    env: ConnectFourEnv,
    solver: PonsSolver,
    *,
    regret_scale: float = REGRET_SCALE_DEFAULT,
    clip: float = REGRET_CLIP_DEFAULT,
) -> float:
    """Best-vs-worst legal move spread in clipped regret units."""
    scores = solver.analyze(env)
    if not scores:
        return 0.0
    raw = float(max(scores.values()) - min(scores.values()))
    return max(0.0, min(clip, raw / regret_scale))


def score_entropy_candidate(
    env: ConnectFourEnv,
    moves_str: str,
    completions: List[str],
    solver: PonsSolver,
    *,
    condition: str = "claims_rationale",
) -> Optional[Dict]:
    """Score one candidate board from base-model completions."""
    base = _record_for_env(env, moves_str)
    if base is None:
        return None

    legal = set(env.legal_moves())
    valid_count = 0
    legal_count = 0
    moves: List[int] = []
    invalid_json = 0
    invalid_move = 0

    for text in completions:
        parsed = parse_structured_response(text, condition=condition)
        if parsed.valid_json:
            valid_count += 1
        else:
            invalid_json += 1
        if parsed.chosen_move is not None:
            moves.append(parsed.chosen_move)
            if parsed.chosen_move in legal:
                legal_count += 1
            else:
                invalid_move += 1
        else:
            invalid_move += 1

    n = max(len(completions), 1)
    counts = Counter(str(m) for m in moves if m in legal)
    distribution = {k: v / n for k, v in sorted(counts.items())}
    most_common_pct = (max(counts.values()) / n) if counts else 0.0
    unique_legal_moves = len(counts)
    spread = solver_score_spread(env, solver)

    record = dict(base)
    record["entropy_filter"] = {
        "samples_per_board": len(completions),
        "valid_json_rate": valid_count / n,
        "legal_move_rate": legal_count / n,
        "invalid_json_count": invalid_json,
        "invalid_move_count": invalid_move,
        "base_move_distribution": distribution,
        "entropy_bits": shannon_entropy(distribution),
        "most_common_move_pct": most_common_pct,
        "unique_legal_moves_sampled": unique_legal_moves,
        "solver_score_spread": spread,
    }
    return record


def passes_entropy_filter(
    record: Dict,
    *,
    min_valid_rate: float = 0.75,
    min_legal_rate: float = 0.75,
    max_most_common_pct: float = 0.625,
    min_score_spread: float = 0.5,
    min_unique_legal_moves: int = 2,
) -> bool:
    meta = record.get("entropy_filter", {})
    return (
        meta.get("valid_json_rate", 0.0) >= min_valid_rate
        and meta.get("legal_move_rate", 0.0) >= min_legal_rate
        and meta.get("most_common_move_pct", 1.0) <= max_most_common_pct
        and meta.get("solver_score_spread", 0.0) >= min_score_spread
        and meta.get("unique_legal_moves_sampled", 0) >= min_unique_legal_moves
    )


def candidate_move_sequences(
    *,
    seed: int,
    candidate_games: int,
    include_tactical: bool = False,
    tactical_candidates: int = 1000,
) -> List[str]:
    """Build candidate move sequences from random play plus optional tactics."""
    sequences = _seeded_position_pool(seed=seed, n_games=candidate_games)
    if include_tactical:
        tactical = generate_training_pool(
            n_positions=tactical_candidates,
            seed=seed + 17,
            max_games=max(1000, tactical_candidates * 2),
            dedup=True,
        )
        sequences.extend(r["moves"] for r in tactical)
    rng = random.Random(seed + 29)
    rng.shuffle(sequences)
    return sequences


def generate_entropy_pool(
    *,
    sample_fn: SampleFn,
    solver: PonsSolver,
    n_positions: int,
    seed: int = 42,
    candidate_games: int = 1000,
    samples_per_board: int = 8,
    include_tactical_candidates: bool = False,
    max_candidates: Optional[int] = None,
    min_valid_rate: float = 0.75,
    min_legal_rate: float = 0.75,
    max_most_common_pct: float = 0.625,
    min_score_spread: float = 0.5,
    progress_every: int = 25,
    condition: str = "claims_rationale",
) -> List[Dict]:
    """Generate accepted entropy-filtered records.

    `sample_fn` is deliberately injected so tests can use a fake sampler and
    production can use the Tinker base-model sampler without changing logic.
    """
    out: List[Dict] = []
    seen = set()
    candidates = candidate_move_sequences(
        seed=seed,
        candidate_games=candidate_games,
        include_tactical=include_tactical_candidates,
    )
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    inspected = 0
    for moves_str in candidates:
        if len(out) >= n_positions:
            break
        if moves_str in seen:
            continue
        seen.add(moves_str)
        inspected += 1

        env = ConnectFourEnv()
        try:
            env.from_move_sequence([int(ch) for ch in moves_str])
        except Exception:
            continue
        if env.is_terminal() or len(env.legal_moves()) <= 1:
            continue

        completions = sample_fn(make_messages(env, condition), samples_per_board)
        record = score_entropy_candidate(
            env, moves_str, completions, solver, condition=condition
        )
        if record is None:
            continue
        if passes_entropy_filter(
            record,
            min_valid_rate=min_valid_rate,
            min_legal_rate=min_legal_rate,
            max_most_common_pct=max_most_common_pct,
            min_score_spread=min_score_spread,
        ):
            out.append(record)
        if progress_every > 0 and inspected % progress_every == 0:
            logger.info(
                "Entropy pool progress: inspected=%d accepted=%d",
                inspected,
                len(out),
            )

    logger.info(
        "Entropy pool accepted %d/%d inspected candidates",
        len(out),
        inspected,
    )
    return out


def write_entropy_pool(records: Iterable[Dict], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
