#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -z "${ANTHROPIC_API_KEY:-}" ]] && { echo "ANTHROPIC_API_KEY not set" >&2; exit 1; }

JUDGE_MODEL=${JUDGE_MODEL:-claude-sonnet-4-6}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-1024}
JUDGE_CONCURRENCY=${JUDGE_CONCURRENCY:-8}

python src/feedback_judge.py \
  --input data/rollouts/y0_graded.jsonl \
  --output data/feedback/c0.jsonl \
  --judge-model "$JUDGE_MODEL" \
  --max-tokens "$JUDGE_MAX_TOKENS" \
  --concurrency "$JUDGE_CONCURRENCY"

python src/build_x1.py \
  --input data/feedback/c0.jsonl \
  --output data/feedback/x1.jsonl
