# Caveman Reasoning Experiment

Prompt-only evaluation of "Caveman" reasoning vs. Concise CoT, Chain-of-Draft,
and Normal CoT on non-math reasoning tasks. No training — first prove whether
the format matters.

## Question

At the same visible reasoning-token budget, does Caveman reasoning outperform
Concise CoT and Chain-of-Draft on non-math reasoning tasks?

## Scope (first pass)

- **Benchmarks:** BBH Tracking Shuffled Objects, BBH Logical Deduction (n=250 each)
- **Model:** Qwen/Qwen2.5-7B-Instruct (temperature=0, max_new_tokens=512)
- **Conditions:** `answer_only`, `normal_cot`, `concise_cot`, `chain_of_draft`,
  `caveman_full`, plus matched-budget variants of `concise_cot` and
  `chain_of_draft` using the per-task mean reasoning-token count of
  `caveman_full`.

## Layout

```
caveman-reasoning/
  configs/         # experiment.yaml, prompts.yaml
  data/            # raw/, processed/{task}.jsonl
  src/             # load_bbh, prompts, run_inference, parse_outputs,
                   # grade, token_count, analyze, plot
  outputs/         # raw_generations/, parsed/, metrics/, plots/
  scripts/         # run_first_pass.sh, run_matched_budget.sh
```

## Pass criteria

Strong positive: `caveman_full` matches or beats `concise_cot` and
`chain_of_draft` on accuracy, uses fewer reasoning tokens, and still wins
against the matched-budget controls on at least one task.

## First-pass Results

_To be filled in after the run._
