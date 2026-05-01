#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export C4_PROJECT_ROOT="${C4_PROJECT_ROOT:-$ROOT_DIR}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/logs/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# GH200 / Hopper-friendly default for vLLM. Override explicitly if you know
# your environment supports DeepGEMM reliably.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

# Helps reduce allocator fragmentation on long RL runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
