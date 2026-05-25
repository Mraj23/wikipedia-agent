# Caveman Reasoning Experiment

Prompt-only evaluation of "Caveman" reasoning vs. Concise CoT, Chain-of-Draft,
and Normal CoT on non-math reasoning tasks. No training — first prove whether
the format matters.

## Question

At the same visible reasoning-token budget, does Caveman reasoning outperform
Concise CoT and Chain-of-Draft on non-math reasoning tasks?

## Scope (first pass)

- **Benchmarks:** BBH Tracking Shuffled Objects, BBH Logical Deduction (n=250 each)
- **Inference:** Tinker (`tinker.ServiceClient` + `tinker_cookbook.renderers`),
  rank-1 LoRA training client used as a base-model sampler (same pattern as
  `connect4_opponent_modeling/faithfulness_v2/generate_pool.py`)
- **Model:** `Qwen/Qwen3-8B` with `qwen3_instruct` renderer (temperature=0,
  max_new_tokens=512). Swap via `--model` / `--renderer` or the `MODEL` /
  `RENDERER` env vars in the run scripts.
- **Conditions:** `answer_only`, `normal_cot`, `concise_cot`, `chain_of_draft`,
  `caveman_full`, plus matched-budget variants of `concise_cot` and
  `chain_of_draft` using the per-task mean reasoning-token count of
  `caveman_full`.

Set `TINKER_API_KEY` before running the scripts.

## Layout

```
caveman-reasoning/
  configs/         # experiment.yaml, prompts.yaml
  data/            # raw/, processed/{task}.jsonl
  src/             # load_bbh, prompts, run_inference, parse_outputs,
                   # grade, token_count, analyze, plot,
                   # error_analysis, write_report
  outputs/         # raw_generations/, parsed/, metrics/, plots/,
                   # error_analysis/, RESULTS.md
  scripts/         # run_first_pass.sh, run_matched_budget.sh
```

## How to run

```
pip install -r requirements.txt
export TINKER_API_KEY=...
./scripts/run_first_pass.sh         # 5 base conditions
./scripts/run_matched_budget.sh     # adds the two matched-budget controls
                                    # and writes outputs/RESULTS.md
```

`run_matched_budget.sh` computes the per-task budget from `caveman_full`'s
mean reasoning tokens, reruns the two control conditions under that budget,
re-aggregates metrics, samples errors via `src/error_analysis.py`, and writes
the final report via `src/write_report.py`.

## Pass criteria

Strong positive: `caveman_full` matches or beats `concise_cot` and
`chain_of_draft` on accuracy, uses fewer reasoning tokens, and still wins
against the matched-budget controls on at least one task.

## First-pass Results

_To be filled in after the run._
