#!/usr/bin/env bash
set -euo pipefail

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed in the active environment; run scripts/deploy_hpc.sh first." >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/.cache/huggingface}"
mkdir -p "$HF_HOME"

# The minimum image size is lowered to 128x128 so the C3-M mask costs ~16 tokens
# (see MM_PROCESSOR_KWARGS in src/smvqa_gaze/hpc/config.py; keep them equal).
MM_KWARGS='{"size": {"shortest_edge": 16384, "longest_edge": 16777216}}'
# JSON defaults live in variables: "${VAR:-{...}}" would append a stray "}"
# whenever VAR is set.
MM_LIMIT='{"image": 70}'

# shellcheck disable=SC2086
exec vllm serve "${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}" \
  --dtype "${VLLM_DTYPE:-bfloat16}" \
  --host "${VLLM_HOST:-0.0.0.0}" \
  --port "${VLLM_PORT:-8000}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-49152}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}" \
  --limit-mm-per-prompt "${VLLM_LIMIT_MM_PER_PROMPT:-$MM_LIMIT}" \
  --mm-processor-kwargs "${VLLM_MM_PROCESSOR_KWARGS:-$MM_KWARGS}" \
  --enable-prefix-caching \
  --seed 0 \
  ${VLLM_EXTRA_ARGS:-}
