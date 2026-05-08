# Tinker Backend: Opponent Next Move

This is the same Connect Four opponent-next-move experiment, but the model
updates run through the Tinker API as LoRA fine-tuning. The game engine, Pons
solver rewards, prompts, position buffer, and held-out move-quality evaluation
stay local.

## Why This Exists

The local GPU trainer currently does full-model updates and reloads vLLM around
rollouts. Tinker gives us a cleaner LoRA-based path:

1. Save current LoRA weights for a sampler.
2. Sample `group_size` completions for each board.
3. Score completions locally with Pons.
4. Build Tinker RL datums with sampled token logprobs and group-relative
   advantages.
5. Call `forward_backward(..., "importance_sampling")` and `optim_step(...)`.

The central causal comparison is unchanged:

```text
Value            = 1.0 * move_quality
OpponentNextMove = 0.8 * move_quality + 0.2 * opponent_reply_quality
```

Both RL conditions use the same scaffold:

```text
<reasoning>...</reasoning>
<opponent_prediction>...</opponent_prediction>
<answer>...</answer>
```

## Install

```bash
pip install -r requirements.txt
pip install -r requirements-tinker.txt
export TINKER_API_KEY="..."
# Optional, if billing is attached to a specific Tinker project:
export TINKER_PROJECT_ID="..."
```

You still need the Pons solver locally because rewards and evaluation are local:

```bash
bash scripts/bootstrap_gpu.sh
```

For better token efficiency, build a hard training buffer before longer runs:

```bash
python experiments/opponent_next_move/make_hard_training_buffer.py \
  --output experiments/opponent_next_move/data/hard_position_buffer_3000.json \
  --metadata_output experiments/opponent_next_move/data/hard_position_buffer_3000_meta.json \
  --n_positions 3000 \
  --min_spread 8 \
  --max_best_moves 2
```

## Smoke Test

Use tiny eval banks first so the API wiring is tested cheaply:

```bash
RL_STEPS=2 \
POSITIONS_PER_STEP=1 \
GROUP_SIZE=4 \
MAX_EVAL_PER_SPLIT=2 \
SEEDS="42" \
RUN_SFT=1 \
bash experiments/opponent_next_move/run_tinker_experiment.sh
```

## Narrow Run

```bash
MODEL="Qwen/Qwen3-4B-Instruct-2507" \
TINKER_RENDERER="qwen3" \
RL_STEPS=100 \
POSITIONS_PER_STEP=1 \
GROUP_SIZE=32 \
MAX_TOKENS=1024 \
EVAL_MAX_TOKENS=1024 \
TEMPERATURE=0.5 \
POSITION_BUFFER="experiments/opponent_next_move/data/hard_position_buffer_3000.json" \
LEARNING_RATE=4e-5 \
LORA_RANK=32 \
SEEDS="42" \
WANDB=1 \
bash experiments/opponent_next_move/run_tinker_experiment.sh
```

By default this runs:

| Condition | Backend | Purpose |
|---|---|---|
| `BaseSimple` | Tinker sampling | Answer-only prompt baseline |
| `BaseScaffold` | Tinker sampling | Opponent-prediction prompt baseline |
| `Value` | Tinker LoRA RL | Scalar Pons move-quality reward |
| `OpponentNextMove` | Tinker LoRA RL | Move-quality plus opponent-reply reward |

Set `RUN_SFT=1` to also run `SFTBestMove`.

## Outputs

```text
experiments/opponent_next_move/tinker_logs/
experiments/opponent_next_move/tinker_results/
experiments/opponent_next_move/tinker_results/summary.json
```

Each trained run writes:

```text
config.json
train_log.jsonl
checkpoint_paths.json
```

`checkpoint_paths.json` contains `final_sampler_path`, a `tinker://...` URI
that can be passed back into `tinker_eval_move_quality.py`.

## Notes From The Tinker Docs

- Tinker training is LoRA-based, so do not compare these trained conditions
  directly against the local full-finetune runs as if the backend were identical.
- RL examples in the docs use group-relative advantages and
  `importance_sampling` by default. This branch follows that pattern.
- Sampler checkpoints are the right checkpoint type for inference/evaluation.
  The runner gives intermediate rollout checkpoints a TTL and keeps the final
  sampler checkpoint unless `FINAL_TTL_SECONDS` is provided by editing the train
  command.
- Renderer choice matters. The default here is `qwen3`, which matches
  `Qwen/Qwen3-4B-Instruct-2507`.

Relevant docs:

- https://tinker-docs.thinkingmachines.ai/tinker/quickstart/
- https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-rl/
- https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/
- https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/weights/
- https://tinker-docs.thinkingmachines.ai/tinker/losses/
