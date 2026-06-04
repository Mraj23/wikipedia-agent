#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -z "${TINKER_API_KEY:-}" ]] && { echo "TINKER_API_KEY not set" >&2; exit 1; }

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
RENDERER=${RENDERER:-qwen2_5_instruct}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
CONCURRENCY=${CONCURRENCY:-16}
EVAL_DATA=${EVAL_DATA:-data/processed/eval.jsonl}
# Trained checkpoints to evaluate (must have outputs/checkpoints/<name>/manifest.json)
RUN_NAMES=${RUN_NAMES:-"rltf_sft"}

run_eval () {  # cond prompt_cond out [extra-args...]
  local cond="$1" pc="$2" out="$3"; shift 3
  python src/eval.py \
    --model "$MODEL" --renderer "$RENDERER" \
    --eval-data "$EVAL_DATA" \
    --output "outputs/evals/${out}.jsonl" \
    --condition "$cond" --prompt-condition "$pc" \
    --max-new-tokens "$MAX_NEW_TOKENS" --concurrency "$CONCURRENCY" "$@"
}

# Base model across all four prompt conditions (gives the prompt-only Pareto:
# plain baseline / concise / chain_of_draft / caveman).
for pc in plain concise chain_of_draft caveman; do
  run_eval base "$pc" "base_${pc}"
done

# Each trained checkpoint under PLAIN (internalization test) and CAVEMAN.
for run in $RUN_NAMES; do
  manifest="outputs/checkpoints/${run}/manifest.json"
  [[ -f "$manifest" ]] || { echo "skip ${run}: no ${manifest}" >&2; continue; }
  for pc in plain caveman; do
    run_eval "$run" "$pc" "${run}_${pc}" --manifest "$manifest"
  done
done

python src/analyze.py --input outputs/evals --output outputs/evals
python src/plot.py --input outputs/evals/summary.csv --output-dir outputs/plots
