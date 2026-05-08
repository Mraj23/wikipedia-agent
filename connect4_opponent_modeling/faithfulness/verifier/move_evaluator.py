"""Move-quality evaluation grounded in Pons solver scores.

Pons returns integer scores per legal column, where higher = better for the
current player. Concrete units depend on the binary version, but in
practice scores fall roughly in [-1000, +1000]. We compute regret in raw
Pons units (`solver_regret`) and a clipped-and-rescaled version for use as
an RL reward (`clipped_regret`).

REGRET_SCALE_DEFAULT divides raw regret so that:
    - a one-tempo blunder (~ 1 ply value swing in Pons units) maps to ~1.0
    - a terminal blunder (changing a draw or win into a loss) maps to ~2.0+

Pons units in this codebase: 1 ply ≈ 1 score unit; terminal ≈ 1000.
We use the raw difference and divide by `REGRET_SCALE_DEFAULT = 8` so that
small inaccuracies have small reward impact while terminal blunders are
fully clipped at 2.0.
"""

from dataclasses import dataclass
from typing import Optional

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver

REGRET_SCALE_DEFAULT = 8.0
REGRET_CLIP_DEFAULT = 2.0


@dataclass
class MoveEvaluation:
    legal: bool
    is_optimal: bool
    chosen_value: Optional[int]
    best_value: Optional[int]
    raw_regret: float
    clipped_regret: float


def evaluate_move(
    env: ConnectFourEnv,
    chosen_col: int,
    solver: PonsSolver,
    *,
    regret_scale: float = REGRET_SCALE_DEFAULT,
    clip: float = REGRET_CLIP_DEFAULT,
) -> MoveEvaluation:
    legal = env.legal_moves()
    if chosen_col not in legal:
        return MoveEvaluation(
            legal=False,
            is_optimal=False,
            chosen_value=None,
            best_value=None,
            raw_regret=float("inf"),
            clipped_regret=clip,
        )
    scores = solver.analyze(env)
    if not scores:
        return MoveEvaluation(
            legal=True,
            is_optimal=False,
            chosen_value=None,
            best_value=None,
            raw_regret=0.0,
            clipped_regret=0.0,
        )
    best_value = max(scores.values())
    chosen_value = scores.get(chosen_col)
    if chosen_value is None:
        return MoveEvaluation(
            legal=True,
            is_optimal=False,
            chosen_value=None,
            best_value=best_value,
            raw_regret=clip * regret_scale,
            clipped_regret=clip,
        )
    raw = float(best_value - chosen_value)
    if raw < 0:
        # Defensive: chosen value cannot exceed best; if it does, treat as 0.
        raw = 0.0
    scaled = raw / regret_scale
    clipped = max(0.0, min(clip, scaled))
    return MoveEvaluation(
        legal=True,
        is_optimal=(chosen_value == best_value),
        chosen_value=chosen_value,
        best_value=best_value,
        raw_regret=raw,
        clipped_regret=clipped,
    )


def solver_regret(env: ConnectFourEnv, chosen_col: int, solver: PonsSolver) -> float:
    """Raw Pons-unit regret (best_value - chosen_value), >= 0."""
    return evaluate_move(env, chosen_col, solver).raw_regret


def clipped_regret(
    env: ConnectFourEnv,
    chosen_col: int,
    solver: PonsSolver,
    *,
    regret_scale: float = REGRET_SCALE_DEFAULT,
    clip: float = REGRET_CLIP_DEFAULT,
) -> float:
    """Scaled regret in [0, clip]. Used as the negative term in the RL reward."""
    return evaluate_move(
        env, chosen_col, solver, regret_scale=regret_scale, clip=clip
    ).clipped_regret
