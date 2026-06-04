#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -z "${ANTHROPIC_API_KEY:-}" ]] && { echo "ANTHROPIC_API_KEY not set" >&2; exit 1; }

JUDGE_MODEL=${JUDGE_MODEL:-claude-sonnet-4-6}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-1024}
JUDGE_CONCURRENCY=${JUDGE_CONCURRENCY:-8}
# caveman (main) | generic (feedback-content ablation)
FEEDBACK_MODE=${FEEDBACK_MODE:-caveman}
# build_x1 ablation flag: "" | "--shuffle-feedback" | "--no-feedback"
X1_ABLATION=${X1_ABLATION:-}

python src/feedback_judge.py \
  --input data/rollouts/y0_graded.jsonl \
  --output data/feedback/c0.jsonl \
  --judge-model "$JUDGE_MODEL" \
  --feedback-mode "$FEEDBACK_MODE" \
  --max-tokens "$JUDGE_MAX_TOKENS" \
  --concurrency "$JUDGE_CONCURRENCY"

python src/build_x1.py \
  --input data/feedback/c0.jsonl \
  --output data/feedback/x1.jsonl \
  $X1_ABLATION
