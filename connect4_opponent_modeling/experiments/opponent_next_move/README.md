# Narrow Experiment: Opponent Next Move

## Question

Does rewarding a model for predicting the opponent's best next move improve
strategic move choice beyond rewarding only the solver value of the model's
chosen move?

The confirmatory comparison is:

```text
OpponentNextMove > Value
```

on Pons-normalized Connect Four move quality.

## Conditions

| Condition | Training | Purpose |
|---|---|---|
| `BaseScaffold` | no training | Prompt/schema baseline |
| `Value` | `1.0 * move_quality` | Dense scalar solver-reward control |
| `OpponentNextMove` | `0.8 * move_quality + 0.2 * opponent_reply_quality` | Hypothesis condition |

`Value` and `OpponentNextMove` use the same response scaffold. The only
difference is which reward components are scored:

```text
<reasoning>...</reasoning>
<opponent_prediction>opponent's best next move</opponent_prediction>
<answer>your move</answer>
```

This is intentional for the narrow causal test. If the prompts differed, an
`OpponentNextMove` gain could come from the extra opponent-facing prompt text
rather than from the opponent-prediction reward itself.

## Deferred Baseline

Add `ValueAnswerOnly` after the narrow run if compute allows:

| Condition | Training | Purpose |
|---|---|---|
| `ValueAnswerOnly` | `1.0 * move_quality` with no opponent-prediction field | Practical value-only recipe baseline |

The answer-only scaffold should be:

```text
<reasoning>...</reasoning>
<answer>your move</answer>
```

This is not the primary causal comparison because it changes both the reward
and the prompt/schema. It answers a different but useful question: whether the
opponent-prediction scaffold plus reward is better than the simplest value-only
training recipe.

## Primary Metric

For a held-out state `s`, Pons scores every legal move. If the model chooses
`a`, the metric is:

```text
move_quality = (Q(s,a) - min_a Q(s,a)) / (max_a Q(s,a) - min_a Q(s,a))
```

If all moves are tied, a legal move scores `1.0`. Invalid or missing answers
score `0.0`.

The main result is the average of:

```text
Connect Four IID bank
Connect Four hard bank
```

The hard bank keeps positions where Pons scores differ meaningfully across
legal moves.

## Secondary Metric

The same evaluator also reports `opponent_reply_quality`, scored after applying
the model's chosen move and comparing the predicted opponent reply to Pons'
reply scores.

This is behavioral process evidence, not hidden-state mechanistic
interpretability.

## Run

Fresh GPU setup:

```bash
bash scripts/bootstrap_gpu.sh
source .venv/bin/activate
source scripts/gpu_env.sh
python scripts/verify_setup.py --expect-gpu --expect-vllm --expect-wandb --skip-probe-check
```

The run script sources `.venv` and `scripts/gpu_env.sh` automatically, sets
`C4_REQUIRE_PONS_SOLVER=1` by default, and runs `scripts/verify_setup.py` before
training. For CPU-only dry runs, use `EXPECT_GPU=0`; to skip setup verification,
use `VERIFY_SETUP=0`.

Smoke test with tiny training/eval:

```bash
GAME_STEPS=2 \
GROUP_SIZE=4 \
EVAL_EVERY=1 \
MAX_EVAL_PER_SPLIT=2 \
SEEDS="42" \
bash experiments/opponent_next_move/run_narrow_experiment.sh
```

Full narrow run:

```bash
GAME_STEPS=500 \
GROUP_SIZE=64 \
EVAL_EVERY=250 \
MAX_TOKENS=1024 \
SEEDS="42 43 44" \
USE_VLLM=1 \
WANDB=1 \
bash experiments/opponent_next_move/run_narrow_experiment.sh
```

Outputs:

```text
experiments/opponent_next_move/data/connect4_eval_banks.jsonl
experiments/opponent_next_move/logs/
experiments/opponent_next_move/checkpoints/
experiments/opponent_next_move/results/
experiments/opponent_next_move/results/summary.json
```

## Decision Rule

The clean positive result is:

```text
OpponentNextMove mean move_quality > Value mean move_quality
```

on both IID and hard banks, without a large validity advantage. The process
metric should also move in the expected direction:

```text
OpponentNextMove opponent_reply_quality > Value opponent_reply_quality
```

If `Value` improves over `BaseScaffold` but `OpponentNextMove` does not improve
over `Value`, the honest conclusion is that dense scalar Pons reward already
contains most of the useful strategic signal for this setup.

## What This Does Not Claim

- It does not claim principal-variation training.
- It does not claim hidden-state mechanistic interpretability.
- It does not use Breakthrough as a load-bearing result.
- It does not test sparse-vs-dense reward as the main contribution.
