"""Tests for training prompt variants used by the faithfulness experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from faithfulness.prompt import make_messages


def test_claims_rationale_prompt_requests_claims_and_move():
    env = ConnectFourEnv()
    messages = make_messages(env, "claims_rationale")
    system = messages[0]["content"]

    assert '"claims"' in system
    assert '"rationale"' in system
    assert '"chosen_move"' in system
    assert "legal_move" not in system
    assert "1-4 claim objects" in system


def test_move_only_prompt_suppresses_reasoning_channels():
    env = ConnectFourEnv()
    messages = make_messages(env, "move_only")
    system = messages[0]["content"]

    assert '"chosen_move"' in system
    assert '"claims"' not in system
    assert '"rationale"' not in system
