"""End-to-end faithfulness evaluator for a model checkpoint.

Per locked board:
    1. Generate one response.
    2. Parse and verify all claims.
    3. Evaluate move quality (legal, optimal, regret).
    4. (Optional) Run causal pipeline: per-claim, per-intervention resampling.
    5. Update per-category FaithfulnessMetrics.

The model API is a `sample_fn(messages, num_samples) -> list[str]`. Adapt
your local model loader, vLLM, or Tinker sampling client to this signature.
For original generation we call sample_fn with num_samples=1.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.causal.pipeline import (
    DEFAULT_N_RESAMPLES,
    DEFAULT_THRESHOLD,
    SampleFn,
    evaluate_response_causality,
)
from faithfulness.eval.board_generator import env_from_record, load_eval_set
from faithfulness.eval.metrics import FaithfulnessMetrics
from faithfulness.parse import parse_structured_response
from faithfulness.prompt import make_messages
from faithfulness.verifier.claim_verifier import verify_claims
from faithfulness.verifier.move_evaluator import evaluate_move

logger = logging.getLogger(__name__)


@dataclass
class PerBoardRecord:
    moves: str
    category: str
    raw_response: str
    chosen_move: Optional[int]
    rationale: str
    valid_json: bool
    legal: bool
    optimal: bool
    regret: float
    truth_labels: List[Optional[bool]]
    causal_labels: List[Optional[bool]] = field(default_factory=list)
    max_change_rates: List[float] = field(default_factory=list)
    original_distribution: Dict[str, float] = field(default_factory=dict)
    original_invalid_json_rate: float = 0.0
    original_invalid_move_rate: float = 0.0
    intervention_distributions: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    intervention_value_drops: List[Dict[str, float]] = field(default_factory=list)
    intervention_invalid_json_rates: List[Dict[str, float]] = field(default_factory=list)
    intervention_invalid_move_rates: List[Dict[str, float]] = field(default_factory=list)
    parse_error: Optional[str] = None


@dataclass
class EvalResult:
    by_category: Dict[str, FaithfulnessMetrics]
    overall: FaithfulnessMetrics
    records: List[PerBoardRecord]

    def summary(self) -> Dict:
        return {
            "overall": self.overall.summary(),
            "by_category": {k: v.summary() for k, v in self.by_category.items()},
        }


def evaluate_checkpoint(
    sample_fn: SampleFn,
    eval_set_path: str,
    solver: PonsSolver,
    *,
    run_causal: bool = True,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = 0,
) -> EvalResult:
    rng = random.Random(seed)
    items = load_eval_set(eval_set_path)
    by_category: Dict[str, FaithfulnessMetrics] = {}
    overall = FaithfulnessMetrics()
    records: List[PerBoardRecord] = []

    for item in items:
        category = item.get("category", "unknown")
        env = env_from_record(item)

        messages = make_messages(env)
        completions = sample_fn(messages, 1)
        raw = completions[0] if completions else ""
        parsed = parse_structured_response(raw)

        chosen = parsed.chosen_move
        if chosen is not None and chosen in env.legal_moves():
            move_eval = evaluate_move(env, chosen, solver)
            legal = True
            optimal = move_eval.is_optimal
            regret = move_eval.clipped_regret
        else:
            legal = False
            optimal = False
            regret = 2.0

        m = by_category.setdefault(category, FaithfulnessMetrics())
        m.add_response(
            valid_json=parsed.valid_json,
            legal=legal,
            optimal=optimal,
            regret=regret,
        )
        overall.add_response(
            valid_json=parsed.valid_json,
            legal=legal,
            optimal=optimal,
            regret=regret,
        )

        truth_labels = verify_claims(parsed.claims, env, solver)

        causal_labels: List[Optional[bool]] = []
        max_changes: List[float] = []
        original_distribution: Dict[str, float] = {}
        original_invalid_json_rate = 0.0
        original_invalid_move_rate = 0.0
        intervention_distributions: List[Dict[str, Dict[str, float]]] = []
        intervention_value_drops: List[Dict[str, float]] = []
        intervention_invalid_json_rates: List[Dict[str, float]] = []
        intervention_invalid_move_rates: List[Dict[str, float]] = []

        if run_causal and parsed.claims:
            causal = evaluate_response_causality(
                env,
                parsed,
                sample_fn,
                solver,
                n_resamples=n_resamples,
                threshold=threshold,
                rng=rng,
                truth_labels=truth_labels,
            )
            original_distribution = causal.original_distribution
            original_invalid_json_rate = causal.original_invalid_json_rate
            original_invalid_move_rate = causal.original_invalid_move_rate
            for cr in causal.per_claim:
                max_change = max(cr.distribution_shifts.values(), default=0.0)
                max_changes.append(max_change)
                m.add_claim(
                    claim_type=cr.claim_type,
                    truth=cr.truth,
                    max_change_rate=max_change,
                )
                overall.add_claim(
                    claim_type=cr.claim_type,
                    truth=cr.truth,
                    max_change_rate=max_change,
                )
                causal_labels.append(cr.is_causal)
                intervention_distributions.append(cr.intervention_distributions)
                intervention_value_drops.append(cr.value_drops)
                intervention_invalid_json_rates.append(cr.invalid_json_rates)
                intervention_invalid_move_rates.append(cr.invalid_move_rates)
        else:
            for claim, truth in zip(parsed.claims, truth_labels):
                m.add_claim(
                    claim_type=claim.type.value,
                    truth=truth,
                    max_change_rate=0.0,
                )
                overall.add_claim(
                    claim_type=claim.type.value,
                    truth=truth,
                    max_change_rate=0.0,
                )
                causal_labels.append(False)
                max_changes.append(0.0)
                intervention_distributions.append({})
                intervention_value_drops.append({})
                intervention_invalid_json_rates.append({})
                intervention_invalid_move_rates.append({})

        records.append(
            PerBoardRecord(
                moves=item["moves"],
                category=category,
                raw_response=raw,
                chosen_move=chosen,
                rationale=parsed.rationale,
                valid_json=parsed.valid_json,
                legal=legal,
                optimal=optimal,
                regret=regret,
                truth_labels=truth_labels,
                causal_labels=causal_labels,
                max_change_rates=max_changes,
                original_distribution=original_distribution,
                original_invalid_json_rate=original_invalid_json_rate,
                original_invalid_move_rate=original_invalid_move_rate,
                intervention_distributions=intervention_distributions,
                intervention_value_drops=intervention_value_drops,
                intervention_invalid_json_rates=intervention_invalid_json_rates,
                intervention_invalid_move_rates=intervention_invalid_move_rates,
                parse_error=parsed.parse_error,
            )
        )

    return EvalResult(by_category=by_category, overall=overall, records=records)
