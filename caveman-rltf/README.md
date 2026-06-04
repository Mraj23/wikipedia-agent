# Caveman RLTF-SFT

**Question:** can text feedback teach a model to internalize Caveman-style
terse reasoning, so it uses fewer output tokens *without* a terse prompt at
inference — at no accuracy cost?

Model: `Qwen/Qwen2.5-7B-Instruct` (Tinker LoRA). Judge: Anthropic Claude.
Tasks: one math (`gsm8k`) + two non-math BBH (`tracking_shuffled_objects`,
`logical_deduction`).

## Two honest framing notes (read first)

1. **This is RLTF-*SFT*, not RLTF-SD.** We do correctness-filtered supervised
   fine-tuning on the feedback-improved attempt. Upstream RLTF-SD is an
   advantage-weighted importance-sampling objective trained jointly with
   multi-turn GRPO. What we build is upstream's *SFT* distillation mode
   (≈ rejection-sampling / STaR). See [FAITHFULNESS.md](FAITHFULNESS.md).
2. **We extend Caveman beyond its stated scope.** The upstream Caveman skill
   is explicit that it shrinks *output* prose, not hidden reasoning ("make
   mouth smaller, not brain"). We deliberately apply the terse style to the
   visible reasoning trace. Qwen2.5-7B has no separate thinking channel, so
   "reasoning tokens" here are just visible prose before `Answer:`. The
   **primary length axis is total output tokens** (parser-independent, and the
   quantity Caveman actually targets); reasoning tokens are secondary.

## RLTF-SFT loop

```
x0     caveman prompt + question
y0     first answer (sampled under x0)
c0     critique from a stronger judge (given the gold answer)
x1     x0 + y0 + c0 + revise instruction
y1     revised answer (sampled under x1)
train  LoRA cross-entropy on (x0 -> y1), kept iff
       { y1 correct AND (y0 incorrect OR len(y1) <= 0.8*len(y0)) }
```

Feedback is present at training time but **removed at inference** — the model
sees only x0 -> y1, so it must internalize the compression.

## Conditions

Prompt-only arms (no training), all evaluated on the base model:

| Prompt condition | Prompt |
|---|---|
| `plain` | step-by-step baseline |
| `concise` | "brief reasoning, avoid unnecessary words" |
| `chain_of_draft` | dense minimal draft steps (Xu et al. 2025) |
| `caveman` | canonical caveman rules applied to reasoning |

Trained arms (`src/train_sft.py` unless noted), each evaluated under **both
`plain` and `caveman`** prompts — the `plain` eval is the internalization test:

| Arm | Data | Tests |
|---|---|---|
| `rltf_sft` | correct + shorter y1 | main result |
| `sft_caveman` | model's shortest correct y0 | "do you even need the revision?" |
| `sft_y1` | correct y1, no length gate | effect of the length filter |
| `grpo_length` | RL, reward = short-if-correct (`train_grpo_length.py`) | does feedback beat a length reward? |

Feedback ablations (rebuild y1 via `scripts/02` env vars, then retrain):

- `FEEDBACK_MODE=generic` — generic concise critique instead of caveman.
- `X1_ABLATION=--shuffle-feedback` — each y0 revised against *another* row's
  critique (does the critique *content* matter, or just revise-shorter?).
- `X1_ABLATION=--no-feedback` — revise-shorter with no critique (pure
  rejection-sampling lower bound).

## Stack

- **Inference + training:** Tinker (`ServiceClient`,
  `create_lora_training_client_async`, `forward_backward_async`,
  `optim_step_async`, `save_weights_for_sampler_async`).
- **Judge:** Anthropic Claude (`--judge-model`, default `claude-sonnet-4-6`).
- **Model:** `Qwen/Qwen2.5-7B-Instruct` + `qwen2_5_instruct` renderer.

## Setup

```
pip install -r requirements.txt
export TINKER_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Run order

```
scripts/00_prepare_data.sh      # tasks -> data/processed/{train,eval}.jsonl
scripts/01_generate_y0.sh       # caveman first answers, n=4 @ temp 0.7
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

## Metrics (`outputs/evals/summary.csv`)

accuracy (+ bootstrap 95% CI), mean/median **output tokens**, mean reasoning
tokens (parsed rows only), accuracy per 1k output tokens, compression ratio vs
the plain baseline, parse-success / invalid-answer rate, and accuracy within
{32,64,96}-output-token budgets. `src/plot.py` draws accuracy-vs-output-token
Pareto plots with accuracy error bars.

## Success bar

Report a win only if, with non-overlapping accuracy CIs (or clearly within
noise on accuracy while strictly fewer tokens):

- `rltf_sft` under the **plain** prompt is terser than `base/plain` at equal
  accuracy → terseness was internalized.
- `rltf_sft` ≥ `grpo_length` at equal output tokens → feedback earns its keep.
- `rltf_sft` ≥ `sft_caveman` and ≥ `sft_y1` → revision + length filter matter.
- Holds (or is neutral) across math and non-math; parse-success maintained.

Run multiple `SEED`s for the data split before quoting any delta.

## Layout

```
configs/   model.yaml tasks.yaml prompts.yaml rltf_sft.yaml grpo_length.yaml
src/       load_bbh grade prompts generate_y0 feedback_judge build_x1
           generate_y1 build_sft_dataset train_sft train_grpo_length
           eval analyze plot _tinker
scripts/   00..07
outputs/   checkpoints/ (manifest.json per run) evals/ plots/
```
