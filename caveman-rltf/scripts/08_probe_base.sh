#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# EXPERIMENT 0 — base operating-point probe (NO training).
# Measures, per task, the base model's natural accuracy and thinking length
# under the PLAIN prompt. We keep tasks that land in the sweet spot for the
# accuracy-vs-thinking-length "hook" curve:
#   accuracy ~30-75%  (room for the expand phase)
#   median thinking tokens > ~150 and low truncation (room to compress)
# Tasks that are saturated (>90%) or floored (~0%) get dropped.

[[ -z "${TINKER_API_KEY:-}" ]] && { echo "TINKER_API_KEY not set" >&2; exit 1; }

MODEL=${MODEL:-Qwen/Qwen3.5-9B}     # smaller model = more failure headroom
RENDERER=${RENDERER:-qwen3}          # thinking ON
EVAL_N=${EVAL_N:-100}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}   # high, so we measure TRUE thinking length
CONCURRENCY=${CONCURRENCY:-16}
TASKS=${TASKS:-"gsm8k date_understanding temporal_sequences tracking_shuffled_objects logical_deduction"}

# Prepare a held-out probe set (we don't train, so train_n is minimal).
TRAIN_N=1 EVAL_N="$EVAL_N" TASKS="$TASKS" scripts/00_prepare_data.sh

python src/eval.py \
  --model "$MODEL" --renderer "$RENDERER" \
  --eval-data data/processed/eval.jsonl \
  --output outputs/evals/probe/base_plain.jsonl \
  --condition base --prompt-condition plain \
  --max-new-tokens "$MAX_NEW_TOKENS" --concurrency "$CONCURRENCY"

python src/analyze.py \
  --input outputs/evals/probe/base_plain.jsonl \
  --output outputs/evals/probe

echo
echo "=== operating point (keep tasks at ~30-75% acc, median think > ~150) ==="
python - <<'PY'
import pandas as pd
d = pd.read_csv("outputs/evals/probe/summary.csv")
cols = ["task", "n_examples", "accuracy", "median_thinking_tokens",
        "mean_thinking_tokens", "has_thinking_rate", "parse_success_rate"]
print(d[cols].sort_values("accuracy").to_string(index=False))
print()
keep = d[(d.accuracy >= 0.30) & (d.accuracy <= 0.75) & (d.median_thinking_tokens > 150)]
print("SWEET SPOT tasks:", list(keep.task) or "NONE — adjust model/tasks/EVAL_N")
PY
