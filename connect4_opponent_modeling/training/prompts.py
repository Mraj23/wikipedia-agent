"""Prompt templates for all six experimental conditions and response parsers.

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

CONDITION_PROMPTS: Dict[str, str] = {
    "A": (
        "You are playing Connect Four.\n\n"
        "{board}\n\n"
        "Legal moves: {legal_moves}\n\n"
        "Think about the position, then choose a column to play.\n"
        "Respond with your reasoning in <think> tags, then your move in <move> tags.\n"
        "Example: <think>I should play center.</think><move>3</move>"
    ),
    "B": (
        "You are playing Connect Four.\n\n"
        "{board}\n\n"
        "Legal moves: {legal_moves}\n\n"
        "Think carefully about the position and choose the best move.\n"
        "Respond with your reasoning in <think> tags, then your move in <move> tags.\n"
        "Example: <think>I need to block the opponent.</think><move>3</move>"
    ),
    "C": (
        "You are playing Connect Four.\n\n"
        "{board}\n\n"
        "Legal moves: {legal_moves}\n\n"
        "Analyze this position carefully. Consider threats, opportunities, and "
        "the strategic value of each legal move.\n"
        "Respond with your analysis in <think> tags, then your move in <move> tags.\n"
        "Example: <think>Column 3 controls the center and creates a double threat.</think>"
        "<move>3</move>"
    ),
    "D": (
        "You are playing Connect Four.\n\n"
        "{board}\n\n"
        "Legal moves: {legal_moves}\n\n"
        "Analyze this position carefully. Consider threats, opportunities, and "
        "the strategic value of each legal move.\n"
        "After choosing your move, predict the board state that will result from "
        "your move. Show it as a text grid in the <future_state> tag.\n"
        "Respond with your analysis in <think> tags, the predicted board state in "
        "<future_state> tags, then your move in <move> tags.\n"
        "Example:\n"
        "<think>Column 3 builds toward a vertical connection.</think>"
        "<future_state>\n"
        ". . . . . . .\n"
        ". . . . . . .\n"
        ". . . . . . .\n"
        ". . . . . . .\n"
        ". . . X . . .\n"
        ". . O X O . .\n"
        "</future_state>"
        "<move>3</move>"
    ),
    "E": (
        "You are playing Connect Four.\n\n"
        "{board}\n\n"
        "Legal moves: {legal_moves}\n\n"
        "Analyze this position carefully. Consider threats, opportunities, and "
        "the strategic value of each legal move.\n"
        "Before choosing your move, predict how your opponent will respond to "
        "your top candidate moves. State your prediction in the "
        "<opponent_prediction> tag.\n"
        "Respond with your analysis in <think> tags, your opponent's predicted "
        "response column in <opponent_prediction> tags, then your move in <move> tags.\n"
        "Example:\n"
        "<think>If I play column 3, opponent likely responds column 4.</think>"
        "<opponent_prediction>4</opponent_prediction>"
        "<move>3</move>"
    ),
    "F": (
        "You are playing Connect Four.\n\n"
        "{board}\n\n"
        "Legal moves: {legal_moves}\n\n"
        "Analyze this position carefully. For each candidate move, think about:\n"
        "1. What threat or advantage does it create?\n"
        "2. How will your opponent most likely respond?\n"
        "3. Does the opponent's response negate your advantage?\n\n"
        "Before choosing your move, predict how your opponent will respond to "
        "your top candidate moves. State your prediction in the "
        "<opponent_prediction> tag.\n"
        "Respond with your analysis in <think> tags, your opponent's predicted "
        "response column in <opponent_prediction> tags, then your move in <move> tags.\n"
        "Example:\n"
        "<think>If I play column 3, opponent likely responds column 4 to block. "
        "If I play column 2 instead, opponent must respond column 2 or I get a "
        "double threat.</think>"
        "<opponent_prediction>2</opponent_prediction>"
        "<move>2</move>"
    ),
}

# Required tags per condition
REQUIRED_TAGS: Dict[str, List[str]] = {
    "A": ["think", "move"],
    "B": ["think", "move"],
    "C": ["think", "move"],
    "D": ["think", "future_state", "move"],
    "E": ["think", "opponent_prediction", "move"],
    "F": ["think", "opponent_prediction", "move"],
}


def format_prompt(condition: str, env: ConnectFourEnv) -> str:
    """Format a prompt template for the given condition and board state.

    Args:
        condition: One of 'A', 'B', 'C', 'D', 'E', 'F'.
        env: Current game environment.

    Returns:
        Formatted prompt string.

    Raises:
        KeyError: If condition is not recognized.
    """
    template = CONDITION_PROMPTS[condition]
    board = env.to_text_grid()
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
        "think": None,
        "move": None,
        "opponent_prediction": None,
        "future_state": None,
        "raw": response,
    }

    # Extract <think>...</think>
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()

    # Extract <move>...</move>
    move_match = re.search(r"<move>\s*(\d)\s*</move>", response)
    if move_match:
        result["move"] = int(move_match.group(1))

    # Extract <opponent_prediction>...</opponent_prediction> (conditions E, F)
    if condition in ("E", "F"):
        pred_match = re.search(r"<opponent_prediction>\s*(\d)\s*</opponent_prediction>", response)
        if pred_match:
            result["opponent_prediction"] = int(pred_match.group(1))

    # Extract <future_state>...</future_state> (condition D)
    if condition == "D":
        fs_match = re.search(r"<future_state>(.*?)</future_state>", response, re.DOTALL)
        if fs_match:
            result["future_state"] = fs_match.group(1).strip()

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
        if tag == "think" and parsed.get("think") is None:
            return False, f"Missing <think> tag for condition {condition}"
        elif tag == "move" and parsed.get("move") is None:
            return False, f"Missing <move> tag for condition {condition}"
        elif tag == "opponent_prediction" and parsed.get("opponent_prediction") is None:
            return False, f"Missing <opponent_prediction> tag for condition {condition}"
        elif tag == "future_state" and parsed.get("future_state") is None:
            return False, f"Missing <future_state> tag for condition {condition}"

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
    test_resp = "<think>I should play center.</think><move>3</move>"
    parsed = parse_response(test_resp, "A")
    valid, reason = validate_response(parsed, "A", env.legal_moves())
    print(f"Response: {test_resp}")
    print(f"Parsed: {parsed}")
    print(f"Valid: {valid}, Reason: '{reason}'")
