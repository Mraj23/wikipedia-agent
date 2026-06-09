# Caveman RLTF-SFT

**Question:** can text feedback teach a thinking model to internalize
Caveman-style terse *thinking*, so it spends fewer **thinking tokens** —
*without* a terse prompt at inference, and at no accuracy cost?

Models (thinking/hybrid, via Tinker LoRA):
- **`Qwen/Qwen3.6-35B-A3B`** — primary (cheap MoE), renderer `qwen3`.
- **`Qwen/Qwen3.5-9B`** — secondary dense check: rerun with
  `MODEL=Qwen/Qwen3.5-9B RENDERER=qwen3 ...`.

(Do not use Qwen2.5 or Qwen3.5-4B — too old / too small.)
Judge: Anthropic Claude. Tasks: one math (`gsm8k`) + two non-math BBH
(`tracking_shuffled_objects`, `logical_deduction`).

## Two honest framing notes (read first)

1. **This is RLTF-*SFT*, not RLTF-SD.** We do correctness-filtered supervised
   fine-tuning on the feedback-improved attempt. Upstream RLTF-SD is an
   advantage-weighted importance-sampling objective trained jointly with
   multi-turn GRPO — what we build is upstream's *SFT* distillation mode
   (≈ rejection-sampling / STaR). See [FAITHFULNESS.md](FAITHFULNESS.md).
2. **We extend Caveman beyond its stated scope, on purpose.** Caveman shrinks
   visible *output* prose, not hidden reasoning ("mouth, not brain"). We apply
   the terse style to the **thinking trace** and ask whether the brain can be
   shrunk without getting dumber. Because Qwen3.x is a real thinking model, we
   measure `<think>` tokens and answer tokens **separately**; the primary axis
   is **thinking tokens**.

## RLTF-SFT loop

```
x0     caveman prompt + question      (model thinks in <think>...</think>)
y0     first attempt (thinking + answer)
c0     critique from a stronger judge (given the gold answer)
x1     x0 + y0 + c0 + "re-think shorter"
y1     revised attempt
train  LoRA cross-entropy on (x0 -> y1), kept iff
       { y1 correct AND (y0 incorrect OR tokens(y1) <= 0.8*tokens(y0)) }
```

Feedback is present at training time but **removed at inference** — the model
sees only `x0 -> y1`, so it must internalize the compression.

## Conditions

Prompt-only arms (base model, no training):

| Prompt | Effect on thinking |
|---|---|
| `plain` | think normally |
| `concise` | "keep thinking brief" |
| `chain_of_draft` | minimal draft steps (Xu et al. 2025) |
| `caveman` | canonical caveman rules applied to thinking |

Trained arms, each evaluated under **both `plain` and `caveman`** prompts — the
`plain` eval is the internalization test:

| Arm | Data | Tests |
|---|---|---|
| `rltf_sft` | correct + shorter y1 | main result |
| `sft_caveman` | model's shortest correct y0 | do you even need the revision? |
| `sft_y1` | correct y1, no length gate | effect of the length filter |
| `grpo_length` | RL, reward = short-if-correct (`train_grpo_length.py`) | does feedback beat a length reward? |

Feedback ablations (rebuild y1 via `scripts/02` env vars, then retrain):

- `FEEDBACK_MODE=generic` — generic concise critique instead of caveman.
- `X1_ABLATION=--shuffle-feedback` — revise against *another* row's critique
  (does the critique *content* matter, or just the revise-shorter pressure?).
- `X1_ABLATION=--no-feedback` — revise-shorter with no critique (rejection
  -sampling lower bound).

## Setup

```
pip install -r requirements.txt
export TINKER_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Run order

```
scripts/00_prepare_data.sh      # tasks -> data/processed/{train,eval}.jsonl
scripts/01_generate_y0.sh       # caveman first attempts, n=4 @ temp 0.7
scripts/02_generate_feedback.sh # Claude critique + build x1
scripts/03_generate_y1.sh       # revisions, n=2 @ temp 0.7
scripts/04_build_sft_dataset.sh # filter -> data/train/rltf_sft.jsonl
scripts/05_train_rltf_sft.sh    # LoRA SFT on (x0 -> y1)
scripts/07_train_controls.sh    # sft_caveman, sft_y1, grpo_length (optional)
scripts/06_eval.sh              # eval matrix + summary.csv + Pareto plots
```

Add trained arms to the eval matrix:
```
RUN_NAMES='rltf_sft sft_caveman sft_y1 grpo_length' scripts/06_eval.sh
```

Rerun the whole thing on the 9B model:
```
MODEL=Qwen/Qwen3.5-9B RENDERER=qwen3 scripts/01_generate_y0.sh   # ...etc
```

## Metrics (`outputs/evals/summary.csv`)

accuracy (+ bootstrap 95% CI), **mean/median thinking tokens** (primary),
mean answer tokens, mean total output tokens, accuracy per 1k thinking tokens,
think-compression ratio vs the plain baseline, parse-success / has-thinking /
invalid-answer rates, and accuracy within {64,128,256}-thinking-token budgets.
`src/plot.py` draws accuracy-vs-thinking-token and accuracy-vs-answer-token
Pareto plots with accuracy error bars.

## Success bar

Report a win only with non-overlapping accuracy CIs (or clearly within noise on
accuracy while strictly fewer thinking tokens):

- `rltf_sft` under the **plain** prompt is terser (fewer thinking tokens) than
  `base/plain` at equal accuracy → terseness was internalized.
- `rltf_sft` ≥ `grpo_length` at equal thinking tokens → feedback earns its keep.
- `rltf_sft` ≥ `sft_caveman` and ≥ `sft_y1` → revision + length filter matter.
- Holds (or is neutral) across math and non-math, and on both models.

Run multiple `SEED`s before quoting any delta.

## Layout

```
configs/   model.yaml tasks.yaml prompts.yaml rltf_sft.yaml grpo_length.yaml
src/       load_bbh grade prompts generate_y0 feedback_judge build_x1
           generate_y1 build_sft_dataset train_sft train_grpo_length
           eval analyze plot _tinker
scripts/   00..07
outputs/   checkpoints/ (manifest.json per run) evals/ plots/
```
