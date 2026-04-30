"""Calibration: evaluate model on game difficulty ladder.

Tests win rate against opponents of increasing strength.
Uses /no_think + custom <think>/<answer> tags for structured reasoning.

Usage:
    python scripts/calibrate_transfer.py --model Qwen/Qwen3-4B --games connect_four breakthrough nim --num_games 10
"""

import argparse
import re
import sys
import random
import json
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyspiel
import numpy as np


OPENSPIEL_NAMES = {
    "breakthrough": "breakthrough(rows=6,columns=6)",
    "nim": "nim(pile_sizes=1;3;5;7)",
    "connect_four": "connect_four",
    "tic_tac_toe": "tic_tac_toe",
}

GAME_DESCRIPTIONS = {
    "breakthrough": "Breakthrough on 6x6 grid. Move forward (straight/diagonal). Capture diagonally. Reach opponent's row to win.",
    "nim": "Nim. Remove objects from one pile per turn. Take the last object to win.",
    "connect_four": "Connect Four. Drop a piece into column 0-6. Get 4 in a row to win.",
    "tic_tac_toe": "Tic-Tac-Toe on 3x3 grid. Get 3 in a row to win.",
}


def clean_board(state, game_name):
    """Make the board representation clearer for the model."""
    raw = str(state)
    if game_name == "connect_four":
        rows = raw.strip().split("\n")
        lines = []
        for row in rows:
            lines.append("  " + " ".join(list(row)))
        lines.append("  0 1 2 3 4 5 6")
        return "\n".join(lines)
    return raw


def make_prompt(state, game_name, legal_actions):
    """Create prompt with /no_think and custom think/answer tags."""
    desc = GAME_DESCRIPTIONS.get(game_name, f"Playing {game_name}.")
    board = clean_board(state, game_name)
    legal_strs = [state.action_to_string(state.current_player(), a) for a in legal_actions]

    prompt = f"""{desc}

Board:
{board}

Legal moves: {', '.join(legal_strs)}

Respond in this exact format:
<reasoning>Analyze threats and best move in 2-3 sentences</reasoning>
<answer>your_move</answer>

/no_think"""
    return prompt


def parse_model_move(response, game_name, legal_actions, state):
    """Parse move from model response. Checks <answer> tag first, then full response."""
    # Try <answer> tag first
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    search_text = answer_match.group(1).strip() if answer_match else response

    # Match against legal action strings
    action_map = {}
    for action in legal_actions:
        action_str = state.action_to_string(state.current_player(), action)
        action_map[action_str] = action

    # Try exact matches (longest first)
    for action_str in sorted(action_map.keys(), key=len, reverse=True):
        if action_str in search_text:
            return action_map[action_str]

    # Fallback for connect_four: bare column numbers
    if game_name == "connect_four":
        digits = re.findall(r"\b([0-6])\b", search_text)
        for d in digits:
            if int(d) in legal_actions:
                return int(d)

    return None


def make_minimax_opponent(game_name, depth):
    """Create a minimax opponent."""
    from open_spiel.python.algorithms import minimax
    game = pyspiel.load_game(OPENSPIEL_NAMES[game_name])

    def value_fn(state):
        if state.is_terminal():
            return state.returns()[state.current_player()]
        return 0.0

    def minimax_move(state):
        _, action = minimax.alpha_beta_search(
            game, state=state, maximum_depth=depth, value_function=value_fn
        )
        return action
    return minimax_move


def make_mcts_opponent(game_name, simulations):
    """Create an MCTS opponent."""
    from open_spiel.python.algorithms import mcts
    game = pyspiel.load_game(OPENSPIEL_NAMES[game_name])
    rng = np.random.RandomState(42)
    evaluator = mcts.RandomRolloutEvaluator(1, rng)
    bot = mcts.MCTSBot(game, 2.0, simulations, evaluator, random_state=rng,
                       solve=True, verbose=False)
    return bot


def play_game(model_fn, game_name, model_player=0, opponent="random",
              opponent_depth=None, opponent_sims=None, verbose=True, log_file=None):
    """Play one game. Returns (winner, num_moves, valid_count, model_moves, game_log)."""
    game = pyspiel.load_game(OPENSPIEL_NAMES[game_name])
    state = game.new_initial_state()
    num_moves = 0
    valid_moves = 0
    model_moves = 0
    game_log = []

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
            legal = state.legal_actions()
            prompt = make_prompt(state, game_name, legal)
            response = model_fn(prompt)
            model_moves += 1

            action = parse_model_move(response, game_name, legal, state)
            if action is not None:
                valid_moves += 1
                action_str = state.action_to_string(current, action)
                if verbose:
                    print(f"  Model plays: {action_str}")
            else:
                action = random.choice(legal)
                action_str = state.action_to_string(current, action)
                if verbose:
                    print(f"  Model INVALID -> random: {action_str}")

            game_log.append({
                "player": "model",
                "move": action_str,
                "valid": action is not None,
                "response": response[:500],
                "board": str(state),
            })
            state.apply_action(action)
        else:
            legal = state.legal_actions()
            if opponent == "minimax" and minimax_fn:
                action = minimax_fn(state)
            elif opponent == "mcts" and mcts_bot:
                action = mcts_bot.step(state)
            else:
                action = random.choice(legal)
            action_str = state.action_to_string(current, action)
            game_log.append({"player": "opponent", "move": action_str})
            state.apply_action(action)

        num_moves += 1

    returns = state.returns()
    if returns[model_player] > returns[1 - model_player]:
        winner = "model"
    elif returns[model_player] < returns[1 - model_player]:
        winner = "opponent"
    else:
        winner = "draw"

    return winner, num_moves, valid_moves, model_moves, game_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--games", nargs="+", default=["connect_four", "breakthrough", "nim"])
    parser.add_argument("--num_games", type=int, default=5)
    parser.add_argument("--output_dir", default="results/calibration")
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
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = prompt
        else:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, do_sample=True,
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

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for game_name in args.games:
        if game_name not in OPENSPIEL_NAMES:
            print(f"Unknown game: {game_name}")
            continue

        print(f"\n{'='*60}")
        print(f"GAME: {game_name} — DIFFICULTY LADDER")
        print(f"{'='*60}")

        game_results = []
        game_logs = []

        for opp_type, depth, sims in OPPONENTS:
            label = opp_type
            if depth: label += f"-{depth}"
            if sims: label += f"-{sims}"

            wins = 0
            total_valid = 0
            total_model_moves = 0

            print(f"\n  --- vs {label} ({args.num_games} games) ---")
            for i in range(args.num_games):
                winner, moves, valid, model_m, log = play_game(
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
                game_logs.append({
                    "game": game_name, "opponent": label, "game_num": i+1,
                    "result": status, "valid": valid, "total_moves": model_m,
                    "log": log,
                })

            valid_rate = total_valid / max(total_model_moves, 1)
            win_rate = wins / args.num_games
            game_results.append((label, wins, args.num_games, valid_rate))
            print(f"  {label}: {wins}/{args.num_games} wins ({win_rate*100:.0f}%), "
                  f"valid={valid_rate*100:.0f}%")

        print(f"\n  {game_name} LADDER SUMMARY:")
        print(f"  {'Opponent':<15} {'Wins':>5} {'Rate':>8} {'Valid':>8}")
        for label, w, n, vr in game_results:
            print(f"  {label:<15} {w}/{n:>3}  {w/n*100:>6.0f}%  {vr*100:>6.0f}%")

        all_results[game_name] = game_results

        # Save detailed logs
        log_path = os.path.join(args.output_dir, f"{game_name}_logs.json")
        with open(log_path, "w") as f:
            json.dump(game_logs, f, indent=2, default=str)
        print(f"  Logs saved to {log_path}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "num_games": args.num_games,
        "results": {
            game: [{"opponent": l, "wins": w, "total": n, "valid_rate": vr}
                   for l, w, n, vr in results]
            for game, results in all_results.items()
        }
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
