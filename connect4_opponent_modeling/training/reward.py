"""Reward functions for RL conditions B-E.

All reward functions return float in [0, 1].
Reward composition per condition:
  B: sparse win/loss only
  C: 0.6 * move_quality + 0.3 * terminal + 0.1 * format
  D: 0.5 * move_quality + 0.2 * future_state_accuracy + 0.2 * terminal + 0.1 * format
  E: 0.5 * move_quality + 0.2 * prediction_accuracy + 0.2 * terminal + 0.1 * format
"""

import re
from typing import Optional

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver
from training.prompts import parse_response, REQUIRED_TAGS


class RewardCalculator:
    """Computes condition-specific rewards for RL training."""

    def __init__(self, solver: PonsSolver) -> None:
        """Initialize with a Pons solver instance.

        Args:
            solver: PonsSolver for move quality and opponent response analysis.
        """
        self.solver = solver

    def condition_b_reward(self, game_result: str) -> float:
        """Sparse win/loss reward for condition B.

        Args:
            game_result: One of 'win', 'loss', 'draw', 'ongoing'.

        Returns:
            1.0 for win, 0.5 for draw, 0.0 for loss/ongoing.
        """
        if game_result == "win":
            return 1.0
        elif game_result == "draw":
            return 0.5
        else:
            return 0.0

    def condition_c_reward(
        self,
        env: ConnectFourEnv,
        played_col: int,
        game_result: str,
        response: str,
    ) -> float:
        """Composite reward for condition C: 0.6*move + 0.3*terminal + 0.1*format.

        Args:
            env: Game state BEFORE the move was played.
            played_col: Column that was played.
            game_result: One of 'win', 'loss', 'draw', 'ongoing'.
            response: Raw model response string.

        Returns:
            Float in [0, 1].
        """
        mq = self._move_quality(env, played_col)
        tr = self._terminal_reward(game_result)
        fr = self._format_reward(response, "C")
        return 0.6 * mq + 0.3 * tr + 0.1 * fr

    def condition_d_reward(
        self,
        env: ConnectFourEnv,
        played_col: int,
        game_result: str,
        response: str,
    ) -> float:
        """Composite reward for condition D: 0.5*move + 0.2*future + 0.2*terminal + 0.1*format.

        Args:
            env: Game state BEFORE the move was played.
            played_col: Column that was played.
            game_result: One of 'win', 'loss', 'draw', 'ongoing'.
            response: Raw model response string.

        Returns:
            Float in [0, 1].
        """
        parsed = parse_response(response, "D")
        mq = self._move_quality(env, played_col)
        fs = self._future_state_accuracy(env, played_col, parsed.get("future_state", ""))
        tr = self._terminal_reward(game_result)
        fr = self._format_reward(response, "D")
        return 0.5 * mq + 0.2 * fs + 0.2 * tr + 0.1 * fr

    def condition_e_reward(
        self,
        env: ConnectFourEnv,
        played_col: int,
        predicted_opp_col: int,
        game_result: str,
        response: str,
    ) -> float:
        """Composite reward for condition E: 0.5*move + 0.2*pred + 0.2*terminal + 0.1*format.

        Args:
            env: Game state BEFORE the move was played.
            played_col: Column that was played.
            predicted_opp_col: The opponent column predicted by the model.
            game_result: One of 'win', 'loss', 'draw', 'ongoing'.
            response: Raw model response string.

        Returns:
            Float in [0, 1].
        """
        mq = self._move_quality(env, played_col)
        pa = self._prediction_accuracy(env, played_col, predicted_opp_col)
        tr = self._terminal_reward(game_result)
        fr = self._format_reward(response, "E")
        return 0.5 * mq + 0.2 * pa + 0.2 * tr + 0.1 * fr

    def _move_quality(self, env: ConnectFourEnv, played_col: int) -> float:
        """Normalized [0,1] move quality from solver.

        Args:
            env: Game state BEFORE the move.
            played_col: Column played.

        Returns:
            Float in [0, 1].
        """
        return self.solver.normalize_reward(env, played_col)

    def _prediction_accuracy(
        self, env: ConnectFourEnv, played_col: int, predicted_col: int
    ) -> float:
        """Score the opponent prediction accuracy.

        1.0 if predicted_col matches solver's optimal opponent response.
        Partial credit based on score difference otherwise.

        Args:
            env: Game state BEFORE played_col.
            played_col: Column the current player played.
            predicted_col: Column the model predicted the opponent would play.

        Returns:
            Float in [0, 1].
        """
        optimal = self.solver.optimal_opponent_response(env, played_col)
        if optimal == -1:
            # Terminal after played_col — no opponent response possible
            return 1.0

        if predicted_col == optimal:
            return 1.0

        # Partial credit: check how good the predicted column is
        next_env = env.copy()
        next_env.make_move(played_col)
        if next_env.is_terminal():
            return 1.0

        scores = self.solver.analyze(next_env)
        if not scores:
            return 0.0

        values = list(scores.values())
        max_s = max(values)
        min_s = min(values)
        if max_s == min_s:
            return 1.0

        opt_score = scores.get(optimal, min_s)
        pred_score = scores.get(predicted_col, min_s)
        # Higher score = better for opponent, so partial credit = how close pred is to opt
        partial = 1.0 - (opt_score - pred_score) / (max_s - min_s)
        return max(0.0, min(1.0, partial))

    def _future_state_accuracy(
        self, env: ConnectFourEnv, played_col: int, stated_future: str
    ) -> float:
        """Score the future state prediction accuracy.

        Applies played_col to a copy of env, gets actual resulting text grid,
        then compares to stated_future.

        Args:
            env: Game state BEFORE played_col.
            played_col: Column played.
            stated_future: The model's predicted board state as text.

        Returns:
            1.0 for exact match, otherwise count matching cells / 42.
        """
        if not stated_future:
            return 0.0

        next_env = env.copy()
        next_env.make_move(played_col)
        actual_grid = next_env.to_text_grid()

        # Extract just the board rows (first 6 lines) from both grids
        actual_lines = actual_grid.strip().split("\n")[:6]
        stated_lines = stated_future.strip().split("\n")[:6]

        if len(actual_lines) != 6 or len(stated_lines) < 6:
            # If stated_future doesn't have 6 rows, do cell comparison with what we have
            pass

        # Extract cells
        actual_cells = []
        for line in actual_lines:
            actual_cells.extend(line.strip().split())

        stated_cells = []
        for line in stated_lines[:6]:
            stated_cells.extend(line.strip().split())

        if not stated_cells:
            return 0.0

        # Exact match check
        if actual_cells == stated_cells and len(actual_cells) == 42:
            return 1.0

        # Partial credit: matching cells / 42
        total = 42
        matches = sum(
            1
            for a, s in zip(actual_cells, stated_cells)
            if a == s
        )
        return matches / total

    @staticmethod
    def _terminal_reward(game_result: str) -> float:
        """Convert game result to terminal reward.

        Args:
            game_result: One of 'win', 'loss', 'draw', 'ongoing'.

        Returns:
            1.0 for win, 0.5 for draw, 0.0 for loss/ongoing.
        """
        if game_result == "win":
            return 1.0
        elif game_result == "draw":
            return 0.5
        return 0.0

    @staticmethod
    def _format_reward(response: str, condition: str) -> float:
        """Check if the response has all required tags and valid move.

        Args:
            response: Raw model response.
            condition: One of 'A'-'F'.

        Returns:
            1.0 if all required tags present and move is a valid integer, else 0.0.
        """
        required = REQUIRED_TAGS.get(condition, ["think", "move"])

        for tag in required:
            pattern = rf"<{tag}>.*?</{tag}>"
            if not re.search(pattern, response, re.DOTALL):
                return 0.0

        # Check move contains a valid integer
        move_match = re.search(r"<move>\s*(\d)\s*</move>", response)
        if not move_match:
            return 0.0

        return 1.0


if __name__ == "__main__":
    from env.pons_wrapper import PonsSolver

    print("=== Reward Calculator Demo ===\n")

    solver = PonsSolver()
    calc = RewardCalculator(solver)

    # Condition B
    print("Condition B:")
    print(f"  Win:  {calc.condition_b_reward('win')}")
    print(f"  Draw: {calc.condition_b_reward('draw')}")
    print(f"  Loss: {calc.condition_b_reward('loss')}")
    print()

    # Set up a position
    env = ConnectFourEnv()
    for col in [3, 3, 4, 2, 5]:
        env.make_move(col)

    # Condition C
    response_c = "<think>I should block.</think><move>3</move>"
    reward_c = calc.condition_c_reward(env, 3, "ongoing", response_c)
    print(f"Condition C reward: {reward_c:.3f}")

    # Format reward
    print(f"Format reward (good): {calc._format_reward(response_c, 'C')}")
    print(f"Format reward (bad):  {calc._format_reward('no tags here', 'C')}")
