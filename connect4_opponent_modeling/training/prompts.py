"""Prompt templates for the active experimental conditions and response parsers.

Conditions A-F form a causal ladder:
  A: SFT only — imitation baseline
  B: Self-play RL — adds adversarial pressure
  C: Solver-RL (value) — adds position evaluation via Pons rewards
  D: Solver-RL + future-state — adds forward projection after own move
  E: Solver-RL + opponent modeling — adds adversarial projection
  F: Prompt-only baseline — inference-time opponent reasoning, no RL
"""

import re
from typing import Dict, List, Optional, Tuple

from env.connect_four_env import ConnectFourEnv

SYSTEM_PROMPT = "You are playing Connect Four. Think carefully before each move."

# All conditions use /no_think (suppresses Qwen's verbose internal thinking)
# and our own <reasoning>/<answer> tags for structured, concise analysis.
#
# The causal ladder differs ONLY in what the model is asked to reason about:
#   B: Just pick a move (sparse reward, no structured reasoning)
#   C: Analyze threats and opportunities
#   D/E: Same structured output contract, but reward different auxiliary fields
#   D: auxiliary reward on future-state prediction
#   E: auxiliary reward on opponent-response prediction
#   F: Same as E but inference-time only (no RL training)

# Shared header — same for every condition. Eliminates prompt-structure
# confound between C/D/E. Earlier per-condition prompts used the literal text
# "One short sentence." as the <reasoning> placeholder, which Qwen3-4B echoed
# back verbatim instead of treating as an instruction. Placeholders here are
# imperative ("Identify…", "Show…", "The column you play…") so the model reads
# them as directions and replaces them with content.
_HEADER = (
    "Connect Four. You are X. Your opponent is O. "
    "Drop a piece into a column 0-6. First to four in a row wins.\n\n"
    "Board:\n{board}\n\n"
    "Legal columns: {legal_moves}\n\n"
    "Respond in this exact format. Replace each placeholder with your own content; "
    "do not echo the placeholder text.\n"
)
_REASONING = (
    "<reasoning>Think through the position. Identify threats and opportunities, "
    "then explain why your chosen move is best.</reasoning>\n"
)
_FUTURE_STATE = (
    "<future_state>The cell your piece will land in after this move, written "
    "as `row=R col=C` where R is 0-5 and C is 0-6.</future_state>\n"
)
_OPP_PREDICTION = (
    "<opponent_prediction>The column 0-6 the opponent will most likely play "
    "next.</opponent_prediction>\n"
)
_ANSWER = "<answer>The column 0-6 you play.</answer>\n\n/no_think"
_PIECE_COUNT = (
    "<piece_count>Total number of pieces currently on the board, modulo 7."
    "</piece_count>\n"
)

# Conditions C/D/E/F/G all share the same scaffold (reasoning + future_state +
# opponent_prediction + answer) so any post-training C-vs-D-vs-E delta is
# attributable to the reward, not the prompt. A and B remain minimal because
# they are imitation/sparse baselines that don't use the auxiliary fields.
_FULL_SCAFFOLD = _HEADER + _REASONING + _FUTURE_STATE + _OPP_PREDICTION + _ANSWER
_OPPONENT_NEXT_MOVE_SCAFFOLD = _HEADER + _REASONING + _OPP_PREDICTION + _ANSWER

_FULL_SCAFFOLD_CONDITIONS = {
    "C",
    "D",
    "E",
    "F",
}
_OPPONENT_PREDICTION_CONDITIONS = {
    *_FULL_SCAFFOLD_CONDITIONS,
    "Value",
    "OpponentNextMove",
    "BaseScaffold",
}

CONDITION_PROMPTS: Dict[str, str] = {
    "A": _HEADER + _REASONING + _ANSWER,
    "B": _HEADER + _REASONING + _ANSWER,
    "C": _FULL_SCAFFOLD,
    "D": _FULL_SCAFFOLD,
    "E": _FULL_SCAFFOLD,
    "F": _FULL_SCAFFOLD,
    "G": _HEADER + _REASONING + _PIECE_COUNT + _ANSWER,
    "BaseSimple": _HEADER + _REASONING + _ANSWER,
    "BaseScaffold": _OPPONENT_NEXT_MOVE_SCAFFOLD,
    "Value": _OPPONENT_NEXT_MOVE_SCAFFOLD,
    "OpponentNextMove": _OPPONENT_NEXT_MOVE_SCAFFOLD,
}

# Required tags per condition. C/D/E/F now share the full-scaffold tag set
# because they share the prompt; the *reward* is what differentiates them.
REQUIRED_TAGS: Dict[str, List[str]] = {
    "A": ["reasoning", "answer"],
    "B": ["reasoning", "answer"],
    "C": ["reasoning", "future_state", "opponent_prediction", "answer"],
    "D": ["reasoning", "future_state", "opponent_prediction", "answer"],
    "E": ["reasoning", "future_state", "opponent_prediction", "answer"],
    "F": ["reasoning", "future_state", "opponent_prediction", "answer"],
    "G": ["reasoning", "piece_count", "answer"],
    "BaseSimple": ["reasoning", "answer"],
    "BaseScaffold": ["reasoning", "opponent_prediction", "answer"],
    "Value": ["reasoning", "opponent_prediction", "answer"],
    "OpponentNextMove": ["reasoning", "opponent_prediction", "answer"],
}


def extract_tag_text(response: str, tag: str) -> Optional[str]:
    """Extract the contents of an XML-like tag, or None if absent."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _clean_board(env: ConnectFourEnv) -> str:
    """Return the canonical board representation.

    Same string used by env.to_text_grid() — also the reference that the
    future_state reward and the probe compare against. Keeping all uses
    aligned means the model never sees one rendering in the prompt and is
    asked to produce a different one.
    """
    return env.to_text_grid()


def _extract_first_int(text: str, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> Optional[int]:
    """Extract the first integer from text, optionally bounded."""
    if not text:
        return None
    matches = re.findall(r"-?\d+", text)
    for match in matches:
        value = int(match)
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        return value
    return None


def format_prompt(condition: str, env: ConnectFourEnv) -> str:
    """Format a prompt template for the given condition and board state.

    Uses /no_think to suppress Qwen's internal thinking, with our own
    <reasoning>/<answer> tags for structured, concise analysis.

    Args:
        condition: One of 'A', 'B', 'C', 'D', 'E', 'F', 'G'.
        env: Current game environment.

    Returns:
        Formatted prompt string.

    Raises:
        KeyError: If condition is not recognized.
    """
    template = CONDITION_PROMPTS[condition]
    board = _clean_board(env)
    legal = ", ".join(str(m) for m in env.legal_moves())
    return template.format(board=board, legal_moves=legal)


def parse_response(response: str, condition: str) -> Dict:
    """Parse a model response and extract structured fields.

    Args:
        response: Raw model output string.
        condition: One of 'A'-'F'.

    Returns:
        Dict with keys: move (int|None), think (str), and optionally
        opponent_prediction (int|None), future_state (str|None).
    """
    result: Dict = {
        "reasoning": None,
        "move": None,
        "opponent_prediction": None,
        "future_state": None,           # raw string (kept for back-compat)
        "future_cell": None,            # parsed (row, col) tuple, or None
        "piece_count": None,
        "raw": response,
    }

    # Extract <reasoning>...</reasoning>
    result["reasoning"] = extract_tag_text(response, "reasoning")

    # Extract <answer>...</answer> → move
    answer_text = extract_tag_text(response, "answer")
    if answer_text is not None:
        result["move"] = _extract_first_int(answer_text, min_value=0, max_value=6)

    # Extract <opponent_prediction>...</opponent_prediction>. C/D/E/F now share
    # the full scaffold prompt so all four parse this field; reward weighting is
    # the only thing that differs between them.
    if condition in _OPPONENT_PREDICTION_CONDITIONS:
        pred_text = extract_tag_text(response, "opponent_prediction")
        if pred_text is not None:
            result["opponent_prediction"] = _extract_first_int(
                pred_text, min_value=0, max_value=6
            )

    # Extract <future_state>...</future_state>. New compact format: the model
    # writes "row=R col=C" identifying the cell its piece will land in.
    # Reward grades whether (R, C) matches the actual landing cell.
    if condition in _FULL_SCAFFOLD_CONDITIONS:
        fs_text = extract_tag_text(response, "future_state")
        result["future_state"] = fs_text
        if fs_text is not None:
            ints = re.findall(r"\d+", fs_text)
            if len(ints) >= 2:
                r_val, c_val = int(ints[0]), int(ints[1])
                if 0 <= r_val <= 5 and 0 <= c_val <= 6:
                    result["future_cell"] = (r_val, c_val)

    # Extract <piece_count>...</piece_count> (condition G)
    if condition == "G":
        piece_count_text = extract_tag_text(response, "piece_count")
        if piece_count_text is not None:
            result["piece_count"] = _extract_first_int(piece_count_text)

    return result


def validate_response(
    parsed: Dict, condition: str, legal_moves: List[int]
) -> Tuple[bool, str]:
    """Validate a parsed response for correctness.

    Args:
        parsed: Output of parse_response.
        condition: One of 'A'-'F'.
        legal_moves: List of currently legal columns.

    Returns:
        Tuple of (is_valid, reason). reason is empty string if valid.
    """
    required = REQUIRED_TAGS.get(condition, [])

    # Check required tags are present
    for tag in required:
        if tag == "reasoning" and parsed.get("reasoning") is None:
            return False, f"Missing <reasoning> tag for condition {condition}"
        elif tag == "answer" and parsed.get("move") is None:
            return False, f"Missing <answer> tag for condition {condition}"
        elif tag == "opponent_prediction" and parsed.get("opponent_prediction") is None:
            return False, f"Missing <opponent_prediction> tag for condition {condition}"
        elif tag == "future_state" and parsed.get("future_state") is None:
            return False, f"Missing <future_state> tag for condition {condition}"
        elif tag == "piece_count" and parsed.get("piece_count") is None:
            return False, f"Missing <piece_count> tag for condition {condition}"

    # Check move is legal
    if parsed.get("move") is not None and parsed["move"] not in legal_moves:
        return False, f"Move {parsed['move']} not in legal moves {legal_moves}"

    # Check opponent_prediction is a valid column
    if parsed.get("opponent_prediction") is not None:
        if parsed["opponent_prediction"] < 0 or parsed["opponent_prediction"] > 6:
            return False, f"Opponent prediction {parsed['opponent_prediction']} not a valid column"

    return True, ""


if __name__ == "__main__":
    env = ConnectFourEnv()
    for col in [3, 3, 4, 2]:
        env.make_move(col)

    print("=== Prompt Templates Demo ===\n")
    for cond in ["A", "B", "C", "D", "E", "F"]:
        print(f"--- Condition {cond} ---")
        prompt = format_prompt(cond, env)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print()

    # Test parsing
    print("=== Parse/Validate Demo ===\n")
    test_resp = "<reasoning>I should play center.</reasoning><answer>3</answer>"
    parsed = parse_response(test_resp, "A")
    valid, reason = validate_response(parsed, "A", env.legal_moves())
    print(f"Response: {test_resp}")
    print(f"Parsed: {parsed}")
    print(f"Valid: {valid}, Reason: '{reason}'")
