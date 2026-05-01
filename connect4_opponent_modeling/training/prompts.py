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

CONDITION_PROMPTS: Dict[str, str] = {
    "A": (
        "Connect Four. You are X. Your opponent is O. Drop into column 0-6. Get 4 in a row to win.\n\n"
        "Board:\n{board}\n\n"
        "Legal columns: {legal_moves}\n\n"
        "Respond in this exact format:\n"
        "<reasoning>One short sentence.</reasoning>\n"
        "<answer>single_digit_column</answer>\n\n"
        "/no_think"
    ),
    "B": (
        "Connect Four. You are X. Your opponent is O. Drop into column 0-6. Get 4 in a row to win.\n\n"
        "Board:\n{board}\n\n"
        "Legal columns: {legal_moves}\n\n"
        "Respond in this exact format:\n"
        "<reasoning>One short sentence.</reasoning>\n"
        "<answer>single_digit_column</answer>\n\n"
        "/no_think"
    ),
    "C": (
        "Connect Four. You are X. Your opponent is O. Drop into column 0-6. Get 4 in a row to win.\n\n"
        "Board:\n{board}\n\n"
        "Legal columns: {legal_moves}\n\n"
        "Respond in this exact format:\n"
        "<reasoning>One or two short sentences about the best move.</reasoning>\n"
        "<answer>single_digit_column</answer>\n\n"
        "/no_think"
    ),
    "D": (
        "Connect Four. You are X. Your opponent is O. Drop into column 0-6. Get 4 in a row to win.\n\n"
        "Board:\n{board}\n\n"
        "Legal columns: {legal_moves}\n\n"
        "Respond in this exact format:\n"
        "<reasoning>One or two short sentences about the best move.</reasoning>\n"
        "<future_state>Six board rows only, using X O .</future_state>\n"
        "<opponent_prediction>single_digit_column</opponent_prediction>\n"
        "<answer>single_digit_column</answer>\n\n"
        "/no_think"
    ),
    "E": (
        "Connect Four. You are X. Your opponent is O. Drop into column 0-6. Get 4 in a row to win.\n\n"
        "Board:\n{board}\n\n"
        "Legal columns: {legal_moves}\n\n"
        "Respond in this exact format:\n"
        "<reasoning>One or two short sentences about the best move.</reasoning>\n"
        "<future_state>Six board rows only, using X O .</future_state>\n"
        "<opponent_prediction>single_digit_column</opponent_prediction>\n"
        "<answer>single_digit_column</answer>\n\n"
        "/no_think"
    ),
    "F": (
        "Connect Four. You are X. Your opponent is O. Drop into column 0-6. Get 4 in a row to win.\n\n"
        "Board:\n{board}\n\n"
        "Legal columns: {legal_moves}\n\n"
        "Respond in this exact format:\n"
        "<reasoning>One or two short sentences. Mention the opponent's likely reply.</reasoning>\n"
        "<opponent_prediction>single_digit_column</opponent_prediction>\n"
        "<answer>single_digit_column</answer>\n\n"
        "/no_think"
    ),
    "G": (
        "Connect Four. You are X. Your opponent is O. Drop into column 0-6. Get 4 in a row to win.\n\n"
        "Board:\n{board}\n\n"
        "Legal columns: {legal_moves}\n\n"
        "Respond in this exact format:\n"
        "<reasoning>One or two short sentences about the best move.</reasoning>\n"
        "<piece_count>Count total pieces on board mod 7</piece_count>\n"
        "<answer>single_digit_column</answer>\n\n"
        "/no_think"
    ),
}

# Required tags per condition (used for format validation)
REQUIRED_TAGS: Dict[str, List[str]] = {
    "A": ["reasoning", "answer"],
    "B": ["reasoning", "answer"],
    "C": ["reasoning", "answer"],
    "D": ["reasoning", "future_state", "opponent_prediction", "answer"],
    "E": ["reasoning", "future_state", "opponent_prediction", "answer"],
    "F": ["reasoning", "opponent_prediction", "answer"],
    "G": ["reasoning", "piece_count", "answer"],
}


def extract_tag_text(response: str, tag: str) -> Optional[str]:
    """Extract the contents of an XML-like tag, or None if absent."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _clean_board(env: ConnectFourEnv) -> str:
    """Create a clean, spaced board with column labels."""
    grid = env.to_text_grid()
    lines = grid.split("\n")
    # Space out each row and add column numbers
    clean = []
    for line in lines:
        if line.startswith("Columns"):
            continue
        if line.startswith("Your"):
            clean.append("  X = you, O = opponent")
            continue
        clean.append("  " + " ".join(list(line.replace(" ", ""))) if "." in line or "X" in line or "O" in line else line)
    clean.append("  0 1 2 3 4 5 6")
    return "\n".join(clean)


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
        "future_state": None,
        "piece_count": None,
        "raw": response,
    }

    # Extract <reasoning>...</reasoning>
    result["reasoning"] = extract_tag_text(response, "reasoning")

    # Extract <answer>...</answer> → move
    answer_text = extract_tag_text(response, "answer")
    if answer_text is not None:
        result["move"] = _extract_first_int(answer_text, min_value=0, max_value=6)

    # Extract <opponent_prediction>...</opponent_prediction> (conditions D, E, F)
    if condition in ("D", "E", "F"):
        pred_text = extract_tag_text(response, "opponent_prediction")
        if pred_text is not None:
            result["opponent_prediction"] = _extract_first_int(
                pred_text, min_value=0, max_value=6
            )

    # Extract <future_state>...</future_state> (conditions D and E)
    if condition in ("D", "E"):
        result["future_state"] = extract_tag_text(response, "future_state")

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
