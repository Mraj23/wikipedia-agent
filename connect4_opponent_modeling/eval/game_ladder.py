"""Canonical game difficulty ladder evaluation.

This module is the source of truth for adversarial game evaluation. Baseline
calibration, prompt-only F, and post-training checkpoint evaluation should all
flow through these prompt templates, move parsing rules, and logging behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pyspiel
from training.prompts import extract_tag_text

OPENSPIEL_NAMES = {
    "breakthrough": "breakthrough(rows=6,columns=6)",
    "nim": "nim(pile_sizes=1;3;5;7)",
    "connect_four": "connect_four",
    "tic_tac_toe": "tic_tac_toe",
}

GAME_DESCRIPTIONS = {
    "breakthrough": (
        "Breakthrough on 6x6 grid. Move forward (straight/diagonal), capture "
        "diagonally, and reach the opponent's back rank to win."
    ),
    "nim": "Nim. Remove objects from one pile per turn. Take the last object to win.",
    "connect_four": "Connect Four. Drop a piece into column 0-6. Get 4 in a row to win.",
    "tic_tac_toe": "Tic-Tac-Toe on 3x3 grid. Get 3 in a row to win.",
}

PROMPT_STYLE_BASE = "base"
PROMPT_STYLE_OPPONENT_AWARE = "opponent_aware"

OPPONENT_SPECS = [
    {"type": "random", "label": "random"},
    {"type": "minimax", "depth": 1, "label": "minimax-1"},
    {"type": "minimax", "depth": 2, "label": "minimax-2"},
    {"type": "minimax", "depth": 4, "label": "minimax-4"},
    {"type": "mcts", "sims": 100, "label": "mcts-100"},
]


def clean_board(state: pyspiel.State, game_name: str) -> str:
    """Make board text easier for the model to read."""
    raw = str(state)
    if game_name == "connect_four":
        rows = raw.strip().split("\n")
        lines = []
        for row in rows:
            lines.append("  " + " ".join(list(row.upper())))
        lines.append("  0 1 2 3 4 5 6")
        lines.append("  X = current player, O = opponent")
        return "\n".join(lines)
    return raw


def make_prompt(
    state: pyspiel.State,
    game_name: str,
    legal_actions: Sequence[int],
    *,
    prompt_style: str = PROMPT_STYLE_BASE,
) -> str:
    """Create the canonical ladder prompt."""
    desc = GAME_DESCRIPTIONS.get(game_name, f"Playing {game_name}.")
    board = clean_board(state, game_name)
    if game_name == "connect_four":
        legal_strs = [str(a) for a in legal_actions]
    else:
        legal_strs = [state.action_to_string(state.current_player(), a) for a in legal_actions]

    if prompt_style == PROMPT_STYLE_OPPONENT_AWARE:
        reasoning_instruction = (
            "Analyze the best move in one or two short sentences. In your reasoning, explicitly "
            "consider the opponent's most likely reply before choosing your move."
        )
        response_format = """Respond in this exact format:
<reasoning>{reasoning_instruction}</reasoning>
<future_state>Predict the board after your move using the same board format.</future_state>
<opponent_prediction>Predict the opponent's most likely reply using the same move notation shown in Legal moves.</opponent_prediction>
<answer>your_move</answer>

/no_think"""
    else:
        reasoning_instruction = "Analyze threats and the best move in one or two short sentences."
        response_format = """Respond in this exact format:
<reasoning>{reasoning_instruction}</reasoning>
<answer>your_move</answer>

/no_think"""

    return f"""{desc}

Board:
{board}

Legal moves: {', '.join(legal_strs)}

{response_format.format(reasoning_instruction=reasoning_instruction)}"""


def parse_model_move(
    response: str,
    game_name: str,
    legal_actions: Sequence[int],
    state: pyspiel.State,
) -> Optional[int]:
    """Parse a legal action from model output with schema-aware rules."""
    import re

    answer_text = extract_tag_text(response, "answer")
    if answer_text is None:
        return None
    search_text = answer_text

    action_map = {}
    for action in legal_actions:
        action_str = state.action_to_string(state.current_player(), action)
        action_map[action_str] = action

    for action_str in sorted(action_map.keys(), key=len, reverse=True):
        if action_str in search_text:
            return action_map[action_str]

    clean = search_text.strip().rstrip(";").rstrip("*")

    for action_str, action in action_map.items():
        clean_action = action_str.rstrip(";").rstrip("*")
        if clean == clean_action:
            return action
        if clean == action_str.rstrip(";"):
            return action

    if game_name == "tic_tac_toe":
        player = None
        for action_str in action_map:
            if action_str and action_str[0] in ("x", "o"):
                player = action_str[0]
                break
        if player:
            prefixed = player + clean
            if prefixed in action_map:
                return action_map[prefixed]
            coord_match = re.search(r"\((\d),\s*(\d)\)", search_text)
            if coord_match:
                target = f"{player}({coord_match.group(1)},{coord_match.group(2)})"
                if target in action_map:
                    return action_map[target]

    if game_name == "connect_four":
        digits = re.findall(r"\b([0-6])\b", search_text)
        for digit in digits:
            move = int(digit)
            if move in legal_actions:
                return move

    if game_name == "nim":
        nim_match = re.search(
            r"pile[:\s]*(\d+)[,\s]*take[:\s]*(\d+)",
            search_text,
            re.IGNORECASE,
        )
        if nim_match:
            target = f"pile:{nim_match.group(1)}, take:{nim_match.group(2)};"
            if target in action_map:
                return action_map[target]

    if game_name == "breakthrough":
        bt_match = re.search(r"([a-f]\d[a-f]\d)", search_text)
        if bt_match:
            move = bt_match.group(1)
            if move in action_map:
                return action_map[move]
            if move + "*" in action_map:
                return action_map[move + "*"]

    return None


def validate_ladder_response(response: str, prompt_style: str) -> tuple[bool, str]:
    """Validate the schema expected by the canonical ladder prompt."""
    required_tags = ["reasoning", "answer"]
    if prompt_style == PROMPT_STYLE_OPPONENT_AWARE:
        required_tags.extend(["future_state", "opponent_prediction"])

    for tag in required_tags:
        if extract_tag_text(response, tag) is None:
            return False, f"Missing <{tag}> tag"
    return True, ""


def make_minimax_opponent(game_name: str, depth: int) -> Callable[[pyspiel.State], int]:
    """Create an OpenSpiel minimax opponent."""
    from open_spiel.python.algorithms import minimax

    game = pyspiel.load_game(OPENSPIEL_NAMES[game_name])

    def value_fn(state: pyspiel.State) -> float:
        if state.is_terminal():
            return state.returns()[state.current_player()]
        return 0.0

    def minimax_move(state: pyspiel.State) -> int:
        _, action = minimax.alpha_beta_search(
            game,
            state=state,
            maximum_depth=depth,
            value_function=value_fn,
        )
        return action

    return minimax_move


def make_mcts_opponent(game_name: str, simulations: int, seed: int):
    """Create an OpenSpiel MCTS opponent."""
    from open_spiel.python.algorithms import mcts

    game = pyspiel.load_game(OPENSPIEL_NAMES[game_name])
    rng = np.random.RandomState(seed)
    evaluator = mcts.RandomRolloutEvaluator(1, rng)
    return mcts.MCTSBot(
        game,
        2.0,
        simulations,
        evaluator,
        random_state=rng,
        solve=True,
        verbose=False,
    )


def play_game(
    model_fn: Callable[[str], str],
    game_name: str,
    *,
    model_player: int = 0,
    opponent_type: str = "random",
    opponent_depth: Optional[int] = None,
    opponent_sims: Optional[int] = None,
    prompt_style: str = PROMPT_STYLE_BASE,
    invalid_move_policy: str = "random_legal",
    seed: int = 42,
) -> Dict:
    """Play one game and return a detailed record."""
    game = pyspiel.load_game(OPENSPIEL_NAMES[game_name])
    state = game.new_initial_state()
    py_rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    model_moves = 0
    valid_moves = 0
    invalid_moves = 0
    game_log = []
    terminated_by_invalid = False

    minimax_fn = None
    mcts_bot = None
    if opponent_type == "minimax" and opponent_depth is not None:
        minimax_fn = make_minimax_opponent(game_name, opponent_depth)
    elif opponent_type == "mcts" and opponent_sims is not None:
        mcts_bot = make_mcts_opponent(game_name, opponent_sims, seed=seed)

    while not state.is_terminal():
        current = state.current_player()

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np_rng.choice(actions, p=probs)
            state.apply_action(action)
            continue

        if current == model_player:
            legal = state.legal_actions()
            prompt = make_prompt(
                state,
                game_name,
                legal,
                prompt_style=prompt_style,
            )
            response = model_fn(prompt)
            model_moves += 1

            schema_valid, invalid_reason = validate_ladder_response(response, prompt_style)
            parsed_action = parse_model_move(response, game_name, legal, state)
            used_fallback = (not schema_valid) or (parsed_action is None)
            if used_fallback:
                invalid_moves += 1
                if invalid_move_policy == "loss":
                    terminated_by_invalid = True
                    winner = "opponent"
                    break
                if invalid_move_policy == "random_legal":
                    action = py_rng.choice(legal)
                else:
                    action = legal[0]
            else:
                valid_moves += 1
                action = parsed_action

            action_str = state.action_to_string(current, action)
            game_log.append(
                {
                    "player": "model",
                    "move": action_str,
                    "valid": not used_fallback,
                    "used_fallback": used_fallback,
                    "invalid_reason": invalid_reason if used_fallback else "",
                    "response": response[:500],
                    "board": str(state),
                }
            )
            state.apply_action(action)
        else:
            legal = state.legal_actions()
            if opponent_type == "minimax" and minimax_fn is not None:
                action = minimax_fn(state)
            elif opponent_type == "mcts" and mcts_bot is not None:
                action = mcts_bot.step(state)
            else:
                action = py_rng.choice(legal)
            action_str = state.action_to_string(current, action)
            game_log.append({"player": "opponent", "move": action_str})
            state.apply_action(action)

    if terminated_by_invalid:
        returns = None
    else:
        returns = state.returns()
        if returns[model_player] > returns[1 - model_player]:
            winner = "model"
        elif returns[model_player] < returns[1 - model_player]:
            winner = "opponent"
        else:
            winner = "draw"

    return {
        "winner": winner,
        "model_moves": model_moves,
        "valid_moves": valid_moves,
        "invalid_moves": invalid_moves,
        "had_invalid_move": invalid_moves > 0,
        "terminated_by_invalid": terminated_by_invalid,
        "log": game_log,
    }


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> List[float]:
    """Compute a 95% Wilson interval for a binomial proportion."""
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2 * total)) / denom
    margin = (
        z
        * math.sqrt((p * (1 - p) + (z * z) / (4 * total)) / total)
        / denom
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def run_difficulty_ladder(
    model_fn: Callable[[str], str],
    *,
    games: Sequence[str] = ("connect_four", "breakthrough", "nim"),
    num_games: int = 50,
    prompt_style: str = PROMPT_STYLE_BASE,
    invalid_move_policy: str = "random_legal",
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """Run the canonical difficulty ladder across one or more games."""
    results: Dict[str, Dict] = {
        "prompt_style": prompt_style,
        "num_games": num_games,
        "invalid_move_policy": invalid_move_policy,
        "seed": seed,
        "games": {},
    }

    log_root = Path(output_dir) if output_dir else None
    if log_root is not None:
        log_root.mkdir(parents=True, exist_ok=True)

    root_rng = random.Random(seed)

    for game_name in games:
        game_results = []
        game_logs = []

        for spec in OPPONENT_SPECS:
            wins = 0
            losses = 0
            draws = 0
            total_valid = 0
            total_invalid = 0
            total_model_moves = 0
            games_with_invalid = 0
            clean_wins = 0
            clean_games = 0

            for game_index in range(num_games):
                game_seed = root_rng.randrange(0, 2**31)
                record = play_game(
                    model_fn,
                    game_name,
                    model_player=game_index % 2,
                    opponent_type=spec["type"],
                    opponent_depth=spec.get("depth"),
                    opponent_sims=spec.get("sims"),
                    prompt_style=prompt_style,
                    invalid_move_policy=invalid_move_policy,
                    seed=game_seed,
                )
                if record["winner"] == "model":
                    wins += 1
                elif record["winner"] == "opponent":
                    losses += 1
                else:
                    draws += 1

                total_valid += record["valid_moves"]
                total_invalid += record["invalid_moves"]
                total_model_moves += record["model_moves"]
                if record["had_invalid_move"]:
                    games_with_invalid += 1
                else:
                    clean_games += 1
                    if record["winner"] == "model":
                        clean_wins += 1

                game_logs.append(
                    {
                        "game": game_name,
                        "opponent": spec["label"],
                        "game_num": game_index + 1,
                        "result": record["winner"],
                        "valid_moves": record["valid_moves"],
                        "invalid_moves": record["invalid_moves"],
                        "had_invalid_move": record["had_invalid_move"],
                        "terminated_by_invalid": record["terminated_by_invalid"],
                        "total_model_moves": record["model_moves"],
                        "seed": game_seed,
                        "log": record["log"],
                    }
                )

            valid_rate = total_valid / max(total_model_moves, 1)
            invalid_rate = total_invalid / max(total_model_moves, 1)
            clean_game_rate = clean_games / num_games
            clean_game_win_rate = clean_wins / clean_games if clean_games > 0 else 0.0
            invalid_as_loss_win_rate = clean_wins / num_games
            game_results.append(
                {
                    "opponent": spec["label"],
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "total": num_games,
                    "win_rate": wins / num_games,
                    "win_rate_ci95": _wilson_ci(wins, num_games),
                    "valid_rate": valid_rate,
                    "invalid_rate": invalid_rate,
                    "games_with_invalid": games_with_invalid,
                    "clean_games": clean_games,
                    "clean_game_rate": clean_game_rate,
                    "clean_game_win_rate": clean_game_win_rate,
                    "invalid_as_loss_win_rate": invalid_as_loss_win_rate,
                }
            )

        game_result = {"opponents": game_results}
        if log_root is not None:
            log_path = log_root / f"{game_name}_logs.json"
            with open(log_path, "w") as handle:
                json.dump(game_logs, handle, indent=2, default=str)
            game_result["log_path"] = str(log_path)

        results["games"][game_name] = game_result

    return results


def summarize_game_results(ladder_results: Dict) -> Dict[str, float]:
    """Extract compact transfer metrics from ladder results."""
    summary: Dict[str, float] = {}

    for game_name, game_result in ladder_results.get("games", {}).items():
        opponents = {item["opponent"]: item for item in game_result.get("opponents", [])}
        for label, item in opponents.items():
            summary[f"{game_name}_{label}_win_rate"] = item.get("win_rate", 0.0)
            summary[f"{game_name}_{label}_valid_rate"] = item.get("valid_rate", 0.0)
            summary[f"{game_name}_{label}_clean_game_win_rate"] = item.get(
                "clean_game_win_rate", 0.0
            )
            summary[f"{game_name}_{label}_invalid_as_loss_win_rate"] = item.get(
                "invalid_as_loss_win_rate", 0.0
            )

        hard_labels = ["minimax-2", "minimax-4", "mcts-100"]
        hard_rates = [opponents[label]["win_rate"] for label in hard_labels if label in opponents]
        if hard_rates:
            summary[f"{game_name}_transfer_score"] = sum(hard_rates) / len(hard_rates)
        clean_hard_rates = [
            opponents[label]["clean_game_win_rate"]
            for label in hard_labels
            if label in opponents
        ]
        if clean_hard_rates:
            summary[f"{game_name}_clean_transfer_score"] = (
                sum(clean_hard_rates) / len(clean_hard_rates)
            )
        strict_hard_rates = [
            opponents[label]["invalid_as_loss_win_rate"]
            for label in hard_labels
            if label in opponents
        ]
        if strict_hard_rates:
            summary[f"{game_name}_invalid_as_loss_transfer_score"] = (
                sum(strict_hard_rates) / len(strict_hard_rates)
            )

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used by the wrapper script."""
    parser = argparse.ArgumentParser(description="Run the canonical game difficulty ladder")
    parser.add_argument("--model", required=True, help="Model checkpoint path or HuggingFace model ID")
    parser.add_argument(
        "--games",
        nargs="+",
        default=["connect_four", "breakthrough", "nim"],
        help="Games to evaluate",
    )
    parser.add_argument("--num_games", type=int, default=50, help="Games per opponent")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation RNG seed")
    parser.add_argument(
        "--prompt_style",
        choices=[PROMPT_STYLE_BASE, PROMPT_STYLE_OPPONENT_AWARE],
        default=PROMPT_STYLE_BASE,
        help="Prompting style for the ladder",
    )
    parser.add_argument(
        "--output_dir",
        default="results/calibration",
        help="Directory for summary and per-game logs",
    )
    parser.add_argument(
        "--invalid_move_policy",
        choices=["random_legal", "first_legal"],
        default="random_legal",
        help="Fallback policy when the model emits an invalid move",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    from eval.model_loader import create_model_fn

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    model_fn = create_model_fn(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ladder_results = run_difficulty_ladder(
        model_fn,
        games=args.games,
        num_games=args.num_games,
        prompt_style=args.prompt_style,
        invalid_move_policy=args.invalid_move_policy,
        output_dir=str(output_dir),
        seed=args.seed,
    )
    summary = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "prompt_style": args.prompt_style,
        "num_games": args.num_games,
        "seed": args.seed,
        "invalid_move_policy": args.invalid_move_policy,
        "difficulty_ladder": ladder_results,
        "transfer_summary": summarize_game_results(ladder_results),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
