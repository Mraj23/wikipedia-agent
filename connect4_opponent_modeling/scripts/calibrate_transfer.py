"""Quick calibration: can the model even play Breakthrough and Nim?

Loads the model ONCE, plays a few games against random opponent on each game.
Reports: valid move rate, win rate, game length.

Usage:
    python scripts/calibrate_transfer.py --model Qwen/Qwen3-4B --games breakthrough nim --num_games 3
    python scripts/calibrate_transfer.py --model checkpoints/condition_e/best --games breakthrough nim connect_four
"""

import argparse
import re
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyspiel
import numpy as np


# Game-specific prompts and move parsing (from GTBench format)
GAME_PROMPTS = {
    "breakthrough": {
        "system": "You are playing Breakthrough, a two-player board game on a 6x6 grid. "
                  "Each player starts with two rows of pieces. Pieces move forward one square "
                  "diagonally or straight. You capture by moving diagonally into an opponent's piece. "
                  "The first player to reach the opponent's home row wins.",
        "move_format": "Your move as <start->end>, e.g., <a2->a3> or <b2->c3>",
        "regex": r"([a-f][1-6]->[a-f][1-6])",
    },
    "nim": {
        "system": "You are playing Nim. There are 4 piles with 1, 3, 5, and 7 objects. "
                  "Players take turns removing any number of objects from a single pile. "
                  "The player who takes the last object wins.",
        "move_format": "Your move as <pile:X, take:Y>, e.g., <pile:1, take:1>",
        "regex": r"pile:(\d+),\s*take:(\d+)",
    },
    "connect_four": {
        "system": "You are playing Connect Four on a 7-column, 6-row board. "
                  "Players take turns dropping pieces into columns. "
                  "The first player to get 4 in a row (horizontal, vertical, or diagonal) wins.",
        "move_format": "Your move as a column number <C1> through <C7>",
        "regex": r"C(\d)",
    },
    "tic_tac_toe": {
        "system": "You are playing Tic-Tac-Toe on a 3x3 grid. "
                  "Players take turns marking a space. Get 3 in a row to win.",
        "move_format": "Your move as <CxRy>, e.g., <C1R1>",
        "regex": r"C(\d)R(\d)",
    },
}

OPENSPIEL_NAMES = {
    "breakthrough": "breakthrough(rows=6,columns=6)",
    "nim": "nim(pile_sizes=1;3;5;7)",
    "connect_four": "connect_four",
    "tic_tac_toe": "tic_tac_toe",
}


def board_to_text(state, game_name):
    """Convert OpenSpiel state to text description."""
    return str(state)


def action_to_text(state, action, game_name):
    """Convert OpenSpiel action to human-readable text."""
    return state.action_to_string(state.current_player(), action)


def parse_model_move(response, game_name, legal_actions, state):
    """Parse the model's response into an OpenSpiel action."""
    config = GAME_PROMPTS[game_name]
    matches = re.findall(config["regex"], response)
    if not matches:
        return None

    move_str = matches[-1]  # take last match
    if isinstance(move_str, tuple):
        move_str = ",".join(move_str)

    # Try to match against legal action strings
    for action in legal_actions:
        action_str = state.action_to_string(state.current_player(), action)
        # Flexible matching
        if game_name == "nim":
            # Parse pile:X, take:Y
            parts = move_str.split(",")
            if len(parts) == 2:
                try:
                    pile = int(parts[0].strip())
                    take = int(parts[1].strip())
                    if f"pile:{pile}" in action_str and f"take:{take}" in action_str:
                        return action
                except ValueError:
                    pass
        elif game_name == "connect_four":
            try:
                col = int(move_str) - 1  # GTBench uses 1-indexed
                if str(col) in action_str or action_str.strip() == str(col):
                    return action
            except ValueError:
                pass
        elif game_name == "breakthrough":
            # Model outputs "a2->a3", OpenSpiel uses "a2a3" (no arrow)
            clean_move = move_str.replace("->", "").replace(" ", "")
            clean_action = action_str.replace(" ", "")
            if clean_move == clean_action:
                return action
        elif game_name == "tic_tac_toe":
            parts = move_str.split(",") if "," in move_str else [move_str[:1], move_str[1:]]
            if len(parts) == 2:
                # Try matching
                if action_str and move_str in action_str:
                    return action

    return None


def make_prompt(state, game_name, legal_actions):
    """Create a prompt for the model."""
    config = GAME_PROMPTS[game_name]
    board = board_to_text(state, game_name)
    legal_strs = [state.action_to_string(state.current_player(), a) for a in legal_actions]

    # For breakthrough, convert "a5a4" to "a5->a4" for readability
    if game_name == "breakthrough":
        display_strs = [s[:2] + "->" + s[2:] for s in legal_strs]
    else:
        display_strs = legal_strs

    prompt = f"""{config['system']}

Current board state:
{board}

Legal moves: {', '.join(display_strs)}

Choose one legal move. {config['move_format']}
Respond with ONLY your move, no explanation."""

    return prompt


def make_minimax_opponent(game_name, depth):
    """Create a minimax opponent using OpenSpiel's minimax."""
    from open_spiel.python.algorithms import minimax
    game = pyspiel.load_game(OPENSPIEL_NAMES[game_name])

    def value_fn(state):
        """Simple heuristic: 0 for non-terminal states."""
        if state.is_terminal():
            return state.returns()[state.current_player()]
        return 0.0

    def minimax_move(state):
        _, action = minimax.alpha_beta_search(
            game, state=state, maximum_depth=depth,
            value_function=value_fn
        )
        return action

    return minimax_move


def make_mcts_opponent(game_name, simulations):
    """Create an MCTS opponent using OpenSpiel's MCTS."""
    from open_spiel.python.algorithms import mcts
    game = pyspiel.load_game(OPENSPIEL_NAMES[game_name])
    rng = np.random.RandomState(42)
    evaluator = mcts.RandomRolloutEvaluator(1, rng)
    bot = mcts.MCTSBot(game, 2.0, simulations, evaluator, random_state=rng,
                       solve=True, verbose=False)
    return bot


def play_game(model_fn, game_name, model_player=0, opponent="random",
              opponent_depth=None, opponent_sims=None, verbose=True):
    """Play one game, model vs opponent.

    opponent: "random", "minimax", or "mcts"

    Returns: (winner, num_moves, valid_move_count, total_model_moves)
    """
    game = pyspiel.load_game(OPENSPIEL_NAMES[game_name])
    state = game.new_initial_state()
    num_moves = 0
    valid_moves = 0
    model_moves = 0

    minimax_fn = None
    mcts_bot = None
    if opponent == "minimax" and opponent_depth:
        minimax_fn = make_minimax_opponent(game_name, opponent_depth)
    elif opponent == "mcts" and opponent_sims:
        mcts_bot = make_mcts_opponent(game_name, opponent_sims)

    while not state.is_terminal():
        current = state.current_player()

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
            num_moves += 1
            continue

        if current == model_player:
            # Model's turn
            legal = state.legal_actions()
            prompt = make_prompt(state, game_name, legal)
            response = model_fn(prompt)
            model_moves += 1

            action = parse_model_move(response, game_name, legal, state)
            if action is not None:
                valid_moves += 1
                if verbose:
                    action_str = state.action_to_string(current, action)
                    print(f"  Model plays: {action_str}")
            else:
                # Invalid move — play random
                action = random.choice(legal)
                if verbose:
                    print(f"  Model INVALID: '{response[:80]}' -> random")

            state.apply_action(action)
        else:
            # Opponent's turn
            legal = state.legal_actions()
            if opponent == "minimax" and minimax_fn:
                action = minimax_fn(state)
            elif opponent == "mcts" and mcts_bot:
                action = mcts_bot.step(state)
            else:
                action = random.choice(legal)
            state.apply_action(action)

        num_moves += 1

    returns = state.returns()
    if returns[model_player] > returns[1 - model_player]:
        winner = "model"
    elif returns[model_player] < returns[1 - model_player]:
        winner = "opponent"
    else:
        winner = "draw"

    return winner, num_moves, valid_moves, model_moves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--games", nargs="+", default=["breakthrough", "nim"])
    parser.add_argument("--num_games", type=int, default=3)
    args = parser.parse_args()

    # Load model once
    print(f"Loading model: {args.model}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype
    ).to(device)
    model.eval()
    print(f"Model loaded on {device}")

    def model_fn(prompt):
        # Use chat template if available
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = prompt
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128, do_sample=True,
                temperature=0.3, pad_token_id=tokenizer.pad_token_id
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Difficulty ladder
    OPPONENTS = [
        ("random", None, None),
        ("minimax", 1, None),
        ("minimax", 2, None),
        ("minimax", 4, None),
        ("mcts", None, 100),
    ]

    # Run calibration
    for game_name in args.games:
        if game_name not in OPENSPIEL_NAMES:
            print(f"Unknown game: {game_name}")
            continue

        print(f"\n{'='*60}")
        print(f"GAME: {game_name} — DIFFICULTY LADDER")
        print(f"{'='*60}")

        results = []
        for opp_type, depth, sims in OPPONENTS:
            label = opp_type
            if depth:
                label += f"-{depth}"
            if sims:
                label += f"-{sims}"

            wins = 0
            total_valid = 0
            total_model_moves = 0

            print(f"\n  --- vs {label} ({args.num_games} games) ---")
            for i in range(args.num_games):
                winner, moves, valid, model_m = play_game(
                    model_fn, game_name, model_player=i % 2,
                    opponent=opp_type, opponent_depth=depth,
                    opponent_sims=sims, verbose=False,
                )
                if winner == "model":
                    wins += 1
                total_valid += valid
                total_model_moves += model_m
                status = "W" if winner == "model" else "L" if winner == "opponent" else "D"
                print(f"    Game {i+1}: {status} (valid={valid}/{model_m})", flush=True)

            valid_rate = total_valid / max(total_model_moves, 1)
            win_rate = wins / args.num_games
            results.append((label, wins, args.num_games, valid_rate))
            print(f"  {label}: {wins}/{args.num_games} wins ({win_rate*100:.0f}%), "
                  f"valid={valid_rate*100:.0f}%")

        print(f"\n{'game_name'} LADDER SUMMARY:")
        print(f"  {'Opponent':<15} {'Wins':>5} {'Rate':>8} {'Valid':>8}")
        for label, w, n, vr in results:
            print(f"  {label:<15} {w}/{n:>3}  {w/n*100:>6.0f}%  {vr*100:>6.0f}%")


if __name__ == "__main__":
    main()
