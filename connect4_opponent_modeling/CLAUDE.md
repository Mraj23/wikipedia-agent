# CLAUDE.md — Active Experiment Contract

## Core Question

Does explicit opponent-modeling reward in Connect Four RL improve adversarial transfer beyond value-only and future-state controls?

## Active Protocol

- **Base model:** `Qwen/Qwen3-4B` instruct
- **No SFT warmup** in the active protocol
- **Game engine:** OpenSpiel
- **In-domain training game:** Connect Four
- **Main trained conditions:** `C`, `D`, `E`
- **Prompt-only comparison:** `F`
- **Primary transfer benchmark:** canonical difficulty ladder, centered on Breakthrough
- **Mechanistic benchmark:** neutral opponent-response probe on locked positions

## Claims We Are Allowed To Make

The strongest intended claim is:

`E > D` on adversarial transfer and on the neutral probe.

Everything else is secondary. Avoid expanding the claim beyond that unless the code and results explicitly support it.

## Critical Invariants

1. All reportable evaluations must use the canonical model loader and ladder harness.
2. `data/probe_positions_locked.jsonl` is immutable once created.
3. Conditions `C`, `D`, and `E` must start from the same base checkpoint.
4. Prompt-only `F` is evaluated from the same base checkpoint as `A`, not from a trained `condition_f` checkpoint.
5. Breakthrough transfer results are more trustworthy than any archived GTBench stub path.
6. Invalid moves must remain analytically visible through valid-rate reporting.

## Canonical Code Paths

- [eval/model_loader.py](/Users/rajmehta/Documents/GitHub/wikipedia-agent/connect4_opponent_modeling/eval/model_loader.py:1)
- [eval/game_ladder.py](/Users/rajmehta/Documents/GitHub/wikipedia-agent/connect4_opponent_modeling/eval/game_ladder.py:1)
- [eval/baseline_eval.py](/Users/rajmehta/Documents/GitHub/wikipedia-agent/connect4_opponent_modeling/eval/baseline_eval.py:1)
- [eval/probe.py](/Users/rajmehta/Documents/GitHub/wikipedia-agent/connect4_opponent_modeling/eval/probe.py:1)
- [eval/pons_benchmark.py](/Users/rajmehta/Documents/GitHub/wikipedia-agent/connect4_opponent_modeling/eval/pons_benchmark.py:1)

## Canonical Comparisons

- `A` vs `F`: prompting-only effect
- `C` vs `D`: future-state auxiliary effect
- `D` vs `E`: opponent-modeling effect

If resources are limited, prioritize `D` vs `E` before anything else.

## What Is Archived / Non-Canonical

Pre-cleanup benchmark stubs, mixed-protocol docs, and exploratory narrative result files are archived under:

- `archive/invalidated_2026_05_01/`

They may still be useful for debugging or project memory, but not for new claims.
