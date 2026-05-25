#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is not set. Inference will fail. Exiting." >&2
  exit 1
fi

MODEL=${MODEL:-Qwen/Qwen3-8B}
RENDERER=${RENDERER:-qwen3_instruct}
MODEL_TAG=${MODEL_TAG:-qwen3_8b}
N=${N:-250}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0}
CONCURRENCY=${CONCURRENCY:-16}

python src/load_bbh.py \
  --tasks tracking_shuffled_objects logical_deduction \
  --n "$N"

for task in tracking_shuffled_objects logical_deduction; do
  for cond in answer_only normal_cot concise_cot chain_of_draft caveman_full; do
    python src/run_inference.py \
      --model "$MODEL" \
      --renderer "$RENDERER" \
      --task "$task" \
      --condition "$cond" \
      --input "data/processed/${task}.jsonl" \
      --output "outputs/raw_generations/${MODEL_TAG}/${task}/${cond}.jsonl" \
      --temperature "$TEMPERATURE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --concurrency "$CONCURRENCY"
  done
done

python src/parse_outputs.py \
  --input outputs/raw_generations \
  --output outputs/parsed

python src/grade.py \
  --input outputs/parsed \
  --output outputs/metrics/graded.jsonl

python src/token_count.py \
  --input outputs/metrics/graded.jsonl \
  --output outputs/metrics/tokenized.jsonl

python src/analyze.py \
  --input outputs/metrics/tokenized.jsonl \
  --output outputs/metrics/summary.csv

python src/plot.py \
  --input outputs/metrics/summary.csv \
  --output outputs/plots/pareto.png

python src/error_analysis.py \
  --input outputs/metrics/tokenized.jsonl \
  --output-dir outputs/error_analysis \
  --conditions normal_cot chain_of_draft caveman_full

python src/write_report.py \
  --input outputs/metrics/summary.csv \
  --output outputs/RESULTS.md \
  --pareto-plot plots/pareto.png \
  --error-analysis-dir error_analysis
