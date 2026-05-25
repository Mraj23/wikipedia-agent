"""GRPO with correctness + length penalty — plan §11 baseline.

Stub. Implement after RLTF-SD data generation works.

Reward
------
  reward = max(0, 1 - alpha * reasoning_tokens / token_budget)  if correct
  reward = 0                                                    otherwise

Loop
----
For each train prompt:
  - sample group_size completions at temp 0.7
  - grade each
  - compute group-relative advantage (per-prompt mean baseline)
  - build tinker.Datum with `padded_advantages` loss mask
    (pattern: connect4_opponent_modeling/experiments/opponent_next_move/
              tinker_train.py:187-212)
  - forward_backward_async with loss_fn="importance_sampling"
  - optim_step_async with AdamParams

Sweep
-----
  alpha in {0.05, 0.1, 0.2}
  token_budget in {32, 64, 96}
"""

import sys


def main():
    print(
        "train_grpo_length.py is a §11 stub. Implement after RLTF-SD data "
        "generation is validated end-to-end. See module docstring and "
        "connect4_opponent_modeling/faithfulness/rl/trainer.py for the GRPO "
        "loop pattern.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
