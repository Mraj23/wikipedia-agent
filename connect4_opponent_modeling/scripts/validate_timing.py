"""Solver timing and bottleneck analysis.

Measures how long the solver takes per position and per group,
projecting total training time over 5000 steps.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.reward import RewardCalculator
from training.minimax import MinimaxSolver
from training.prompts import parse_response
from spiral.position_buffer import PositionBuffer


def time_solver_analyze(solver, positions, n_calls_per_pos=1):
    """Time solver.analyze() calls."""
    times = []
    for env in positions:
        start = time.time()
        for _ in range(n_calls_per_pos):
            solver.analyze(env)
        elapsed = time.time() - start
        times.append(elapsed)
    return times


def time_reward_computation(calc, solver, positions, group_size=8):
    """Simulate reward computation for a group of completions per position.

    This mirrors what _compute_rewards_batch does: for each completion,
    it calls solver.normalize_reward (which calls analyze) and potentially
    optimal_opponent_response (another analyze call).
    """
    times = []
    for env in positions:
        legal = env.legal_moves()
        start = time.time()

        for i in range(group_size):
            col = legal[i % len(legal)]
            # move_quality calls analyze once
            mq = solver.normalize_reward(env, col)
            # prediction_accuracy calls analyze on next_env
            calc._prediction_accuracy(env, col, 3)

        elapsed = time.time() - start
        times.append(elapsed)
    return times


def main():
    print("=" * 60)
    print("SOLVER TIMING & BOTTLENECK ANALYSIS")
    print("=" * 60)
    print()

    solver = PonsSolver()
    calc = RewardCalculator(solver)
    buf = PositionBuffer(pool_size=20, min_moves_remaining=2, seed=42)

    print(f"Pons binary available: {solver.is_available()}")
    print()

    # Sample 5 positions
    positions = buf.sample(5)

    # 1. Time single analyze call
    print("--- Single solver.analyze() call ---")
    single_times = time_solver_analyze(solver, positions, n_calls_per_pos=1)
    avg_single = sum(single_times) / len(single_times)
    print(f"  Per-position: {[f'{t:.4f}s' for t in single_times]}")
    print(f"  Average: {avg_single:.4f}s")
    print()

    # 2. Time group reward computation (simulating 8 completions)
    print("--- Reward computation for group_size=8 (current code, no caching) ---")
    group_times = time_reward_computation(calc, solver, positions, group_size=8)
    avg_group = sum(group_times) / len(group_times)
    print(f"  Per-position: {[f'{t:.3f}s' for t in group_times]}")
    print(f"  Average: {avg_group:.3f}s")
    print()

    # 3. Projections
    print("--- Projections for 5000 steps ---")
    total_current = avg_group * 5000
    total_cached = avg_single * 2 * 5000  # 2 analyze calls if cached (position + next)
    print(f"  Current (no cache): {total_current / 3600:.1f} hours of solver time")
    print(f"  With caching:       {total_cached / 3600:.1f} hours of solver time")
    print(f"  Speedup:            {total_current / total_cached:.1f}x")
    print()

    # 4. Decision
    if avg_group > 5.0:
        print("RECOMMENDATION: Solver caching is CRITICAL (>5s per step)")
    elif avg_group > 1.0:
        print("RECOMMENDATION: Solver caching recommended (>1s per step)")
    else:
        print("RECOMMENDATION: Solver is fast enough, caching optional")


if __name__ == "__main__":
    main()
