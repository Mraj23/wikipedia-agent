"""Position buffer for sampling diverse board states during RL training.

Generates positions via minimax-vs-minimax self-play (same approach as
sft_data_gen.py) but without labels — RL discovers the policy.
"""

import random
from typing import List, Optional

from env.connect_four_env import ConnectFourEnv
from training.minimax import MinimaxSolver


def _categorize_depth(moves_played: int) -> str:
    """Categorize a position by game phase.

    Args:
        moves_played: Number of moves played so far.

    Returns:
        One of 'beginning', 'middle', 'end'.
    """
    if moves_played <= 8:
        return "beginning"
    elif moves_played <= 20:
        return "middle"
    else:
        return "end"


class PositionBuffer:
    """Pool of diverse board positions for RL prompt sampling.

    Generates positions by playing minimax-vs-minimax games with randomized
    depth and 20% random move injection, then stores them as move sequence
    strings for memory efficiency.
    """

    def __init__(
        self,
        pool_size: int = 10000,
        min_moves_remaining: int = 6,
        seed: int = 42,
    ) -> None:
        """Initialize and populate the position buffer.

        Args:
            pool_size: Number of positions to generate.
            min_moves_remaining: Minimum moves to end (filters out near-terminal).
            seed: Random seed.
        """
        self._pool: List[str] = []  # move sequence strings
        self._rng = random.Random(seed)
        self._min_moves_remaining = min_moves_remaining
        self._fill(pool_size)

    def _fill(self, n: int) -> None:
        """Generate n positions via minimax self-play.

        Args:
            n: Target number of positions.
        """
        import time
        import sys

        # Track phase distribution for stratified coverage
        phase_counts = {"beginning": 0, "middle": 0, "end": 0}
        target_per_phase = n // 3
        games_played = 0
        start_time = time.time()
        last_log = 0

        while len(self._pool) < n:
            depth1 = self._rng.randint(2, 8)
            depth2 = self._rng.randint(2, 8)
            solver1 = MinimaxSolver(depth=depth1)
            solver2 = MinimaxSolver(depth=depth2)

            env = ConnectFourEnv()
            snapshots: List[str] = []

            while not env.is_terminal():
                moves_played = len(env._move_history)
                phase = _categorize_depth(moves_played)

                # Only collect if this phase isn't over-represented
                if not env.is_terminal() and phase_counts.get(phase, 0) < target_per_phase + n // 6:
                    move_seq = env.to_move_sequence()
                    snapshots.append((move_seq, phase))

                # Play the move
                current_solver = solver1 if env.current_player() == 1 else solver2
                if self._rng.random() < 0.2:
                    move = self._rng.choice(env.legal_moves())
                else:
                    move = current_solver.best_move(env)
                env.make_move(move)

            games_played += 1

            # Filter and add positions
            for move_seq, phase in snapshots:
                if len(self._pool) >= n:
                    break

                # Reconstruct to check validity
                test_env = ConnectFourEnv()
                if move_seq:
                    test_env.from_move_sequence([int(c) for c in move_seq])

                if test_env.is_terminal():
                    continue

                # Check minimum moves remaining (estimate via quick playout)
                remaining = self._estimate_remaining(test_env)
                if remaining < self._min_moves_remaining:
                    continue

                self._pool.append(move_seq)
                phase_counts[phase] = phase_counts.get(phase, 0) + 1

            # Progress logging every 10% or every 30 seconds
            current = len(self._pool)
            elapsed = time.time() - start_time
            if current - last_log >= max(1, n // 10) or elapsed - (last_log / max(1, n) * elapsed) > 30:
                pct = current / n * 100
                rate = current / elapsed if elapsed > 0 else 0
                eta = (n - current) / rate if rate > 0 else 0
                print(
                    f"  Buffer: {current}/{n} ({pct:.0f}%) | "
                    f"{games_played} games | "
                    f"{elapsed:.0f}s elapsed | "
                    f"~{eta:.0f}s remaining | "
                    f"phases: {dict(phase_counts)}",
                    flush=True,
                )
                last_log = current

    def _estimate_remaining(self, env: ConnectFourEnv) -> int:
        """Estimate moves remaining via quick minimax playout.

        Args:
            env: Current position.

        Returns:
            Estimated number of moves to terminal.
        """
        sim = env.copy()
        quick = MinimaxSolver(depth=2)
        count = 0
        while not sim.is_terminal() and count < 42:
            move = quick.best_move(sim)
            sim.make_move(move)
            count += 1
        return count

    def sample(self, batch_size: int = 1) -> List[ConnectFourEnv]:
        """Sample positions from the buffer.

        Args:
            batch_size: Number of positions to sample.

        Returns:
            List of ConnectFourEnv objects at the sampled positions.
        """
        seqs = self._rng.choices(self._pool, k=batch_size)
        envs = []
        for seq in seqs:
            env = ConnectFourEnv()
            if seq:
                env.from_move_sequence([int(c) for c in seq])
            envs.append(env)
        return envs

    def save(self, path: str) -> None:
        """Save the position pool to a JSON file.

        Args:
            path: Output file path.
        """
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._pool, f)

    @classmethod
    def load(cls, path: str, seed: int = 42) -> "PositionBuffer":
        """Load a pre-generated position pool from a JSON file.

        Skips the expensive minimax self-play generation.

        Args:
            path: Path to a JSON file saved by save().
            seed: Random seed for sampling.

        Returns:
            PositionBuffer with pre-loaded positions.
        """
        import json
        buf = cls.__new__(cls)
        buf._rng = random.Random(seed)
        with open(path) as f:
            buf._pool = json.load(f)
        buf._min_moves_remaining = 0  # Already filtered
        return buf

    def __len__(self) -> int:
        return len(self._pool)
