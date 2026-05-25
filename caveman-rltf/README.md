# Caveman RLTF Pipeline

Two-turn RLTF-SD (self-distillation) training pipeline to teach an 8B
instruct model to produce shorter caveman-style reasoning while preserving
accuracy on non-math BBH tasks.

## RLTF loop

```
x0     original prompt (caveman instructions + question)
y0     first answer
c0     structured critique from a stronger judge
x1     x0 + y0 + c0 + revise instruction
y1     revised answer
train  LoRA on (x0 -> y1) filtered for {y1 correct AND
       (y0 incorrect OR y1 reasoning <= 0.8 * y0 reasoning)}
```

Feedback is available at training time but NOT at inference. The model must
internalize the compression instruction.

## Stages and conditions

| Condition | Trainer | Data |
|---|---|---|
| SFT-caveman | `src/train_sft.py` | model's own short correct y0 |
| GRPO-length | `src/train_grpo_length.py` (stub) | correctness + length scalar |
| RLTF-SD-compression | `src/train_sft.py` | filtered (x0, y1) from §8 |
| RLTF-SD-random-feedback | `src/train_sft.py` | feedback shuffled across rows |

## Stack

- **Inference + training:** Tinker (`tinker.ServiceClient`,
  `create_lora_training_client_async`, `forward_backward_async`,
  `optim_step_async`, `save_weights_for_sampler_async`).
- **Judge:** Anthropic Claude (configurable via `--judge-model`), structured
  feedback per the §6.2 schema.
- **Model:** `Qwen/Qwen2.5-7B-Instruct` with `qwen2_5_instruct` renderer
  by default. Override via `--model` / `--renderer`.

## Setup

```
pip install -r requirements.txt
export TINKER_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Run order

```
scripts/00_prepare_data.sh        # BBH -> data/processed/{train,eval}.jsonl
scripts/01_generate_y0.sh         # n_samples=4 at temp=0.7
scripts/02_generate_feedback.sh   # Claude critique
scripts/03_generate_y1.sh         # n_samples=2 at temp=0.7
scripts/04_build_sd_dataset.sh    # filter + format SFT pairs
scripts/05_train_rltf_sd.sh       # LoRA SFT on (x0 -> y1)
scripts/06_eval.sh                # held-out eval, temp=0
```

Each script is thin — it reads env-var defaults and calls the matching
`src/*.py`.

## Layout

```
caveman-rltf/
  configs/
    model.yaml         # base model, renderer, LoRA rank
    tasks.yaml         # task list, train/eval sizes
    prompts.yaml       # human-readable mirror of src/prompts.py
    rltf_sd.yaml       # SFT hyperparams
    grpo_length.yaml   # GRPO sweep (used once train_grpo_length lands)
  data/
    raw/        processed/   rollouts/   feedback/   revised/   train/
  src/          # see above
  scripts/      # 00..06
  outputs/
    rollouts/   feedback/   revised/
    checkpoints/   # tinker://... sampler paths recorded as manifest.json
    evals/      plots/
```

## Success bar (plan §17)

Proceed only if:

- `RLTF-SD-compression` > `GRPO-length` at the same mean reasoning tokens
- `RLTF-SD-compression` > `SFT-caveman` and > `SFT-y1` (ablation C)
- Parse-success rate maintained
- Improves on at least one task without hurting the other

Strong target: ~71% / ~43 tokens vs base ~62% / ~90 tokens.
