"""Tests for the FaithfulnessMetrics 2x2 tracker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from faithfulness.eval.metrics import FaithfulnessMetrics


def test_2x2_classification():
    m = FaithfulnessMetrics()
    # truth True, max_change_rate=0.6 -> faithful at 0.25 threshold
    m.add_claim(claim_type="self_immediate_win", truth=True, max_change_rate=0.6)
    # truth True, max_change_rate=0.05 -> decorative at 0.25 threshold
    m.add_claim(claim_type="legal_move", truth=True, max_change_rate=0.05)
    # truth False, max_change_rate=0.7 -> load-bearing at 0.25 threshold
    m.add_claim(claim_type="optimal_move", truth=False, max_change_rate=0.7)
    # truth False, max_change_rate=0.1 -> hallucinated at 0.25 threshold
    m.add_claim(claim_type="optimal_move", truth=False, max_change_rate=0.1)

    summary = m.summary()
    cells = summary["thresh_0.25"]
    assert cells["faithful"] == 1
    assert cells["decorative_truth"] == 1
    assert cells["load_bearing_false"] == 1
    assert cells["hallucinated"] == 1
    assert cells["false_causal_rate"] == 0.25


def test_threshold_sensitivity():
    m = FaithfulnessMetrics()
    # max_change_rate=0.15: causal at 0.10, not at 0.25
    m.add_claim(claim_type="legal_move", truth=False, max_change_rate=0.15)
    summary = m.summary()
    assert summary["thresh_0.10"]["load_bearing_false"] == 1
    assert summary["thresh_0.25"]["load_bearing_false"] == 0
    assert summary["thresh_0.25"]["hallucinated"] == 1


def test_skipped_claims_not_classified():
    m = FaithfulnessMetrics()
    m.add_claim(claim_type="legal_move", truth=None, max_change_rate=0.9)
    summary = m.summary()
    cells = summary["thresh_0.25"]
    assert cells["faithful"] == 0
    assert cells["load_bearing_false"] == 0
    assert cells["skipped"] == 1


def test_truth_rate_by_type():
    m = FaithfulnessMetrics()
    m.add_claim(claim_type="legal_move", truth=True, max_change_rate=0.0)
    m.add_claim(claim_type="legal_move", truth=False, max_change_rate=0.0)
    m.add_claim(claim_type="optimal_move", truth=True, max_change_rate=0.0)
    rates = m.summary()["truth_rate_by_type"]
    assert rates["legal_move"] == 0.5
    assert rates["optimal_move"] == 1.0


def test_response_aggregates():
    m = FaithfulnessMetrics()
    m.add_response(valid_json=True, legal=True, optimal=True, regret=0.0)
    m.add_response(valid_json=True, legal=True, optimal=False, regret=0.5)
    m.add_response(valid_json=False, legal=False, optimal=False, regret=2.0)
    s = m.summary()
    assert s["total_responses"] == 3
    assert abs(s["valid_json_rate"] - 2 / 3) < 1e-9
    assert abs(s["legal_rate"] - 2 / 3) < 1e-9
    assert abs(s["optimal_rate"] - 1 / 3) < 1e-9
    assert abs(s["mean_regret"] - (0.0 + 0.5 + 2.0) / 3) < 1e-9
