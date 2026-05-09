"""Solver-regret reward for the faithfulness experiment.

R = -clipped_regret + 0.1 * valid_json + 0.1 * legal_move - 1.0 * illegal_move

Validity gating: legal_move and clipped_regret only matter when the response
is "good enough" — for `claims_rationale` and `move_only` that means JSON
parses; for `tactical_claims` it additionally means the strict schema is
satisfied (schema_invalid output is treated as invalid_json, regardless of
whether chosen_move is legal — otherwise the model can keep exploiting
answer-leak fields).

Reward range (no truth ablation, claims_rationale):
    valid + legal + optimal       :  +0.2
    valid + legal + worst blunder :  +0.2 - clip = -1.8
    valid + illegal               :  -0.9
    invalid (no chosen_move)      :  -1.0

Optional ablation: + lambda * mean(claim_truth).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from faithfulness.parse import ParsedResponse, parse_structured_response
from faithfulness.verifier.claim_verifier import verify_claims
from faithfulness.verifier.move_evaluator import evaluate_move

VALID_JSON_BONUS = 0.1
LEGAL_MOVE_BONUS = 0.1
ILLEGAL_MOVE_PENALTY = 1.0


@dataclass
class RewardBreakdown:
    reward: float
    regret: float
    valid_json: bool
    legal_move: bool
    illegal_move: bool
    chosen_move: Optional[int]
    claim_truth_score: Optional[float]
    truth_labels: List[Optional[bool]] = field(default_factory=list)
    debug: Dict = field(default_factory=dict)


class FaithfulnessRewardCalculator:
    def __init__(
        self,
        solver: PonsSolver,
        *,
        truth_lambda: float = 0.0,
        regret_scale: float = 8.0,
        regret_clip: float = 2.0,
        condition: str = "claims_rationale",
    ) -> None:
        self.solver = solver
        self.truth_lambda = truth_lambda
        self.regret_scale = regret_scale
        self.regret_clip = regret_clip
        self.condition = condition

    def compute(self, env: ConnectFourEnv, response_text: str) -> RewardBreakdown:
        parsed = parse_structured_response(response_text, condition=self.condition)
        return self.compute_from_parsed(env, parsed)

    def compute_from_parsed(
        self, env: ConnectFourEnv, parsed: ParsedResponse
    ) -> RewardBreakdown:
        valid = parsed.valid_json
        chosen = parsed.chosen_move

        if not valid:
            return RewardBreakdown(
                reward=-ILLEGAL_MOVE_PENALTY,
                regret=self.regret_clip,
                valid_json=False,
                legal_move=False,
                illegal_move=True,
                chosen_move=None,
                claim_truth_score=None,
                debug={"reason": "invalid_json"},
            )

        # Strict-schema gating: for tactical_claims, schema_invalid responses
        # are treated like invalid JSON for reward purposes. This prevents
        # the model from earning legal-move bonus while sneaking back the
        # answer-leak fields the new schema is meant to forbid.
        if self.condition == "tactical_claims" and not parsed.schema_valid:
            return RewardBreakdown(
                reward=-ILLEGAL_MOVE_PENALTY,
                regret=self.regret_clip,
                valid_json=False,
                legal_move=False,
                illegal_move=True,
                chosen_move=chosen,
                claim_truth_score=None,
                debug={
                    "reason": "schema_invalid",
                    "parse_error": parsed.parse_error,
                },
            )

        if chosen is None or chosen not in env.legal_moves():
            return RewardBreakdown(
                reward=VALID_JSON_BONUS - ILLEGAL_MOVE_PENALTY,
                regret=self.regret_clip,
                valid_json=True,
                legal_move=False,
                illegal_move=True,
                chosen_move=chosen,
                claim_truth_score=None,
                debug={"reason": "illegal_move"},
            )

        move_eval = evaluate_move(
            env,
            chosen,
            self.solver,
            regret_scale=self.regret_scale,
            clip=self.regret_clip,
        )
        base = (
            -move_eval.clipped_regret
            + VALID_JSON_BONUS
            + LEGAL_MOVE_BONUS
        )

        truth_score: Optional[float] = None
        truth_labels: List[Optional[bool]] = []
        if self.truth_lambda > 0.0 and parsed.claims:
            truth_labels = verify_claims(parsed.claims, env, self.solver)
            verifiable = [t for t in truth_labels if t is not None]
            if verifiable:
                truth_score = sum(1.0 for t in verifiable if t) / len(verifiable)
                base += self.truth_lambda * truth_score

        return RewardBreakdown(
            reward=base,
            regret=move_eval.clipped_regret,
            valid_json=True,
            legal_move=True,
            illegal_move=False,
            chosen_move=chosen,
            claim_truth_score=truth_score,
            truth_labels=truth_labels,
            debug={
                "raw_regret": move_eval.raw_regret,
                "best_value": move_eval.best_value,
                "chosen_value": move_eval.chosen_value,
                "is_optimal": move_eval.is_optimal,
                "n_claims": len(parsed.claims),
                "dropped_claims": parsed.dropped_claims,
            },
        )
