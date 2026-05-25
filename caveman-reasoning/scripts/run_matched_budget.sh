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
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0}
CONCURRENCY=${CONCURRENCY:-16}

python - <<'PY'
import json, pathlib, statistics

budgets = {}
with open("outputs/metrics/tokenized.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["condition"] != "caveman_full":
            continue
        budgets.setdefault(r["task"], []).append(r["reasoning_tokens"])

out = {t: max(1, round(statistics.mean(v))) for t, v in budgets.items()}
pathlib.Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
with open("outputs/metrics/budgets.json", "w") as f:
    json.dump(out, f, indent=2)
print("per-task budgets:", out)
PY

BUDGETS_FILE=outputs/metrics/budgets.json

for task in tracking_shuffled_objects logical_deduction; do
  BUDGET=$(python -c "import json; print(json.load(open('${BUDGETS_FILE}'))['${task}'])")
  for cond in concise_cot_matched_budget chain_of_draft_matched_budget; do
    python src/run_inference.py \
      --model "$MODEL" \
      --renderer "$RENDERER" \
      --task "$task" \
      --condition "$cond" \
      --input "data/processed/${task}.jsonl" \
      --output "outputs/raw_generations/${MODEL_TAG}/${task}/${cond}.jsonl" \
      --budget "$BUDGET" \
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
  --output-dir outputs/error_analysis

python src/write_report.py \
  --input outputs/metrics/summary.csv \
  --output outputs/RESULTS.md \
  --budgets outputs/metrics/budgets.json \
  --pareto-plot plots/pareto.png \
  --error-analysis-dir error_analysis
