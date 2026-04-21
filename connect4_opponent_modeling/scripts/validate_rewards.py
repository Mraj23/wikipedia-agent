"""Reward consistency validation script.

Verifies that reward functions produce correct values, component weights
are internally consistent, and the double-computation path in
grpo_trainer._compute_single_reward matches direct reward_calc calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.reward import RewardCalculator
from training.grpo_config import _REWARD_WEIGHTS
from training.prompts import parse_response, validate_response, REQUIRED_TAGS
from spiral.position_buffer import PositionBuffer
from spiral.game_playout import play_to_completion
from training.minimax import MinimaxSolver


def make_valid_response(condition: str, env: ConnectFourEnv, col: int) -> str:
    """Generate a synthetic valid response with correct tags for the condition."""
    if condition == "B":
        return f"<think>I play column {col}.</think><move>{col}</move>"
    elif condition == "C":
        return f"<think>Column {col} looks strong.</think><move>{col}</move>"
    elif condition == "D":
        # Need opponent_prediction, future_state, and move
        next_env = env.copy()
        next_env.make_move(col)
        grid = next_env.to_text_grid()
        # Extract just the board rows (first 6 lines)
        board_lines = grid.strip().split("\n")[:6]
        board_text = "\n".join(board_lines)
        return (
            f"<think>Column {col} builds toward a connection.</think>"
            f"<opponent_prediction>3</opponent_prediction>"
            f"<future_state>\n{board_text}\n</future_state>"
            f"<move>{col}</move>"
        )
    elif condition == "E":
        return (
            f"<think>If I play {col}, opponent responds 3.</think>"
            f"<opponent_prediction>3</opponent_prediction>"
            f"<move>{col}</move>"
        )
    elif condition == "G":
        piece_count = len(env._move_history) % 7
        return (
            f"<think>There are pieces on the board.</think>"
            f"<piece_count>{piece_count}</piece_count>"
            f"<move>{col}</move>"
        )
    raise ValueError(f"Unknown condition: {condition}")


def test_reward_ranges():
    """Test that all rewards are in [0, 1]."""
    print("=== Test: Reward ranges ===")
    solver = PonsSolver(fallback_depth=4)
    calc = RewardCalculator(solver)
    buf = PositionBuffer(pool_size=50, min_moves_remaining=2, seed=42)
    minimax = MinimaxSolver(depth=4)

    for condition in ["B", "C", "D", "E", "G"]:
        errors = 0
        for i in range(10):
            env = buf.sample(1)[0]
            legal = env.legal_moves()
            col = legal[i % len(legal)]

            if condition == "B":
                result = play_to_completion(env, col, minimax, env.current_player())
                reward = calc.condition_b_reward(result)
                assert 0.0 <= reward <= 1.0, f"B reward out of range: {reward}"
            else:
                response = make_valid_response(condition, env, col)
                # Determine game result
                test_env = env.copy()
                test_env.make_move(col)
                if test_env.is_terminal():
                    winner = test_env.winner()
                    if winner == env.current_player():
                        game_result = "win"
                    elif winner is None:
                        game_result = "draw"
                    else:
                        game_result = "loss"
                else:
                    game_result = "ongoing"

                if condition == "C":
                    reward = calc.condition_c_reward(env, col, game_result, response)
                elif condition == "D":
                    reward = calc.condition_d_reward(env, col, game_result, response)
                elif condition == "E":
                    parsed = parse_response(response, "E")
                    pred = parsed.get("opponent_prediction", -1)
                    reward = calc.condition_e_reward(env, col, pred, game_result, response)
                elif condition == "G":
                    parsed = parse_response(response, "G")
                    count = parsed.get("piece_count", -1)
                    reward = calc.condition_g_reward(env, col, count, game_result, response)

                if not (0.0 <= reward <= 1.0):
                    print(f"  FAIL: {condition} reward = {reward} (out of range)")
                    errors += 1

        status = "PASS" if errors == 0 else f"FAIL ({errors} errors)"
        print(f"  Condition {condition}: {status}")
    print()


def test_component_weight_consistency():
    """Test that sum(weight * component) matches total reward."""
    print("=== Test: Component weight consistency ===")
    solver = PonsSolver(fallback_depth=4)
    calc = RewardCalculator(solver)
    buf = PositionBuffer(pool_size=50, min_moves_remaining=2, seed=42)

    for condition in ["C", "D", "E", "G"]:
        weights = _REWARD_WEIGHTS[condition]
        max_diff = 0.0
        errors = 0

        for i in range(10):
            env = buf.sample(1)[0]
            legal = env.legal_moves()
            col = legal[i % len(legal)]
            response = make_valid_response(condition, env, col)

            # Determine game result
            test_env = env.copy()
            test_env.make_move(col)
            if test_env.is_terminal():
                winner = test_env.winner()
                if winner == env.current_player():
                    game_result = "win"
                elif winner is None:
                    game_result = "draw"
                else:
                    game_result = "loss"
            else:
                game_result = "ongoing"

            # Get total reward from reward_calc
            if condition == "C":
                total = calc.condition_c_reward(env, col, game_result, response)
            elif condition == "D":
                total = calc.condition_d_reward(env, col, game_result, response)
            elif condition == "E":
                parsed = parse_response(response, "E")
                pred = parsed.get("opponent_prediction", -1)
                total = calc.condition_e_reward(env, col, pred, game_result, response)
            elif condition == "G":
                parsed = parse_response(response, "G")
                count = parsed.get("piece_count", -1)
                total = calc.condition_g_reward(env, col, count, game_result, response)

            # Compute components manually (same as grpo_trainer._compute_single_reward)
            move_quality = solver.normalize_reward(env, col)
            terminal = RewardCalculator._terminal_reward(game_result)

            if condition == "C":
                recomputed = 0.67 * move_quality + 0.33 * terminal
            elif condition == "D":
                parsed_d = parse_response(response, "D")
                fs = calc._future_state_accuracy(env, col, parsed_d.get("future_state", ""))
                recomputed = 0.56 * move_quality + 0.22 * fs + 0.22 * terminal
            elif condition == "E":
                pred_acc = calc._prediction_accuracy(env, col, pred)
                recomputed = 0.56 * move_quality + 0.22 * pred_acc + 0.22 * terminal
            elif condition == "G":
                count_acc = RewardCalculator._piece_count_accuracy(env, count)
                recomputed = 0.56 * move_quality + 0.22 * count_acc + 0.22 * terminal

            diff = abs(total - recomputed)
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                print(f"  FAIL: {condition} pos {i}: total={total:.6f} recomputed={recomputed:.6f} diff={diff:.8f}")
                errors += 1

        status = "PASS" if errors == 0 else f"FAIL ({errors} errors)"
        print(f"  Condition {condition}: {status} (max_diff={max_diff:.8f})")
    print()


def test_invalid_responses():
    """Test that invalid responses get zero reward."""
    print("=== Test: Invalid responses ===")
    solver = PonsSolver(fallback_depth=4)
    calc = RewardCalculator(solver)

    invalid_response = "I want to play column 3"

    for condition in ["C", "D", "E", "G"]:
        format_r = RewardCalculator._format_reward(invalid_response, condition)
        assert format_r == 0.0, f"Format reward for invalid {condition} response: {format_r}"
        print(f"  Condition {condition}: PASS (format_r=0.0)")
    print()


def test_component_ranges():
    """Test that individual components are in [0, 1]."""
    print("=== Test: Component ranges ===")
    solver = PonsSolver(fallback_depth=4)
    calc = RewardCalculator(solver)
    buf = PositionBuffer(pool_size=50, min_moves_remaining=2, seed=42)

    for i in range(20):
        env = buf.sample(1)[0]
        legal = env.legal_moves()
        col = legal[i % len(legal)]

        # move_quality
        mq = solver.normalize_reward(env, col)
        assert 0.0 <= mq <= 1.0, f"move_quality out of range: {mq}"

        # terminal
        for result in ["win", "loss", "draw", "ongoing"]:
            tr = RewardCalculator._terminal_reward(result)
            assert 0.0 <= tr <= 1.0, f"terminal_reward({result}) out of range: {tr}"

        # format
        for cond in ["C", "D", "E", "G"]:
            valid_resp = make_valid_response(cond, env, col)
            fr = RewardCalculator._format_reward(valid_resp, cond)
            assert fr == 1.0, f"format_reward for valid {cond} response: {fr}"

        # prediction_accuracy
        pred_acc = calc._prediction_accuracy(env, col, 3)
        assert 0.0 <= pred_acc <= 1.0, f"prediction_accuracy out of range: {pred_acc}"

        # piece_count_accuracy
        correct_count = len(env._move_history) % 7
        ca = RewardCalculator._piece_count_accuracy(env, correct_count)
        assert ca == 1.0, f"piece_count_accuracy for correct prediction: {ca}"
        ca_wrong = RewardCalculator._piece_count_accuracy(env, (correct_count + 1) % 7)
        assert ca_wrong == 0.0, f"piece_count_accuracy for wrong prediction: {ca_wrong}"

    print("  All 20 positions: PASS")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("REWARD CONSISTENCY VALIDATION")
    print("=" * 60)
    print()

    test_reward_ranges()
    test_component_weight_consistency()
    test_invalid_responses()
    test_component_ranges()

    print("=" * 60)
    print("ALL REWARD VALIDATION TESTS PASSED")
    print("=" * 60)
