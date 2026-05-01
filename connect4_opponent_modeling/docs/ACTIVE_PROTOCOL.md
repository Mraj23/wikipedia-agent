# Active Protocol

Date frozen: **May 1, 2026**

## Summary

The repository now treats **`Qwen/Qwen3-4B` instruct with no SFT warmup** as the only active protocol.

This freeze was made because the working prompts, recent RL runs, and usable artifacts all came from the instruct path, while the older `Qwen3-4B-Base + SFT warmup` story remained partially documented but not operationally consistent.

## Reportable Experiment Path

1. Train `C`, `D`, and `E` from the same instruct checkpoint.
2. Evaluate `A` and `F` from that same instruct checkpoint without RL.
3. Use the canonical difficulty ladder for adversarial transfer.
4. Treat Breakthrough as the primary transfer target.
5. Use the neutral opponent-response probe as the mechanistic test.
6. Keep `D` and `E` output-matched: same structured response schema, different scored auxiliary signal.

## Explicitly Deprecated As Primary Evidence

- archived GTBench wrapper paths that do not run the intended opponent
- GameBench stub paths
- mixed-protocol comparisons across instruct and base checkpoints
- narrative markdown summaries not backed by canonical eval JSONs

## Practical Rule

If a number was not produced by the canonical evaluation suite after this freeze, it should be treated as exploratory only.

## Bring-Up Rule

Fresh GPU instances should start from `scripts/bootstrap_gpu.sh`, then `source .venv/bin/activate` and `source scripts/gpu_env.sh`. The active protocol assumes the Pascal Pons solver, `7x6.book`, and the canonical eval stack are all present before any training run is treated as real.
