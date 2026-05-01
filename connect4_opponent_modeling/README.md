# Connect Four Opponent Modeling

This repository studies a narrow question:

**Does explicit opponent-modeling reward in Connect Four RL produce adversarial transfer that exceeds value-only and future-state controls?**

As of the May 1, 2026 cleanup, the active experiment protocol is intentionally narrower and more defensible than the earlier mixed setup.

## Active Protocol

- **Base model:** `Qwen/Qwen3-4B` instruct
- **No SFT warmup** in the active protocol
- **Training domain:** Connect Four with OpenSpiel game logic
- **Oracle / reward source:** Pons solver when available, minimax fallback only where explicitly documented
- **Main trained conditions:** `C`, `D`, `E`
- **Prompt-only baseline:** `F`
- **Primary transfer evaluation:** canonical difficulty ladder, with **Breakthrough** as the main transfer target
- **Key mechanistic evaluation:** neutral opponent-response probe on locked Connect Four positions

If you see references elsewhere to `Qwen3-4B-Base`, SFT warmup as the active path, GTBench as the main transfer metric, or GameBench as a live benchmark, treat those as archived or superseded unless they were reintroduced explicitly after this cleanup.

## GPU Bring-Up

Fresh GPU machines should be bootstrapped from the repo itself, not by hand:

```bash
bash scripts/bootstrap_gpu.sh
source .venv/bin/activate
source scripts/gpu_env.sh
python scripts/verify_setup.py --expect-gpu --expect-vllm --expect-wandb
```

This bootstrap path installs Python dependencies, sets the stable GH200/Hopper `vLLM` defaults, downloads the Pascal Pons `7x6.book`, builds the `connect4_solver`, and verifies that the solver can actually execute from the repo root.

### External Artifact Policy

- Do **not** use Git LFS for the Pascal Pons opening book.
- Do **not** push model weights to GitHub.
- `7x6.book` is tracked in this repo for convenience, and `scripts/bootstrap_gpu.sh` will also download it if it is missing on a fresh machine.
- Checkpoints should live in `checkpoints/` during active work and be published to a model store or artifact system such as W&B artifacts or Hugging Face if you need persistence beyond the machine.

## Conditions

| Condition | Meaning | What changes |
|---|---|---|
| `A` | Instruct baseline | No RL, base prompt |
| `C` | Value-only RL | Reward for move quality and terminal outcome |
| `D` | Future-state RL | Shares the `D/E` output schema; rewards future-state accuracy |
| `E` | Opponent-modeling RL | Shares the `D/E` output schema; rewards opponent-response accuracy |
| `F` | Prompt-only baseline | No RL, opponent-aware prompting at inference time |

`D` and `E` are the critical comparison. They should use the same structured output contract and differ only in which auxiliary field is scored. The main claim should only be framed around `E > D`, not `E > base`.

## Canonical Evaluation

The source-of-truth evaluation paths are:

- [eval/baseline_eval.py](/Users/rajmehta/Documents/GitHub/wikipedia-agent/connect4_opponent_modeling/eval/baseline_eval.py:1)
- [eval/game_ladder.py](/Users/rajmehta/Documents/GitHub/wikipedia-agent/connect4_opponent_modeling/eval/game_ladder.py:1)
- [eval/model_loader.py](/Users/rajmehta/Documents/GitHub/wikipedia-agent/connect4_opponent_modeling/eval/model_loader.py:1)

These paths share:

- identical model loading behavior
- chat template handling
- completion-only decoding
- one ladder prompt/parsing policy
- deterministic eval seed control
- explicit prompt-style control for `A` vs `F`, with `F` using a richer opponent-aware structured response schema

Do not use ad hoc evaluation scripts to produce reportable numbers unless they call into these modules.

## Recommended Workflow

### 1. Train

```bash
bash scripts/bootstrap_gpu.sh
source .venv/bin/activate
source scripts/gpu_env.sh
python -m spiral.train --condition C --model Qwen/Qwen3-4B --game_steps 500 --group_size 64 --wandb
python -m spiral.train --condition D --model Qwen/Qwen3-4B --game_steps 500 --group_size 64 --wandb
python -m spiral.train --condition E --model Qwen/Qwen3-4B --game_steps 500 --group_size 64 --wandb
```

Or use:

```bash
bash scripts/run_preliminary.sh
```

### 2. Evaluate

Run the canonical suite:

```bash
bash scripts/run_all_evals.sh
```

This evaluates:

- `A` from the instruct checkpoint with base prompts
- `F` from the same instruct checkpoint with richer opponent-aware structured prompts
- any available trained checkpoints for `B/C/D/E`

### 3. Inspect Results

```bash
python -m analysis.correlation --results results/
python -m analysis.plot_curves --results results/ --output results/
```

## What Counts As Defensible Evidence

A good result should show all of the following:

1. `E` beats `D` on Breakthrough transfer under the canonical ladder.
2. `E` beats `D` on the neutral opponent-response probe.
3. `F` does not erase the need for training.
4. Improvements are not just validity improvements; raw win-rates, valid-move rates, clean-game win-rates, and invalid-as-loss views are all reported.
5. Optional controls such as GSM8K / MATH-500 do not suggest generic capability drift is the main story.

## Archived Material

Invalidated or non-canonical material from the pre-cleanup phase is stored under:

- `archive/invalidated_2026_05_01/`

That archive exists for forensic reference, not for reportable experiment claims.
