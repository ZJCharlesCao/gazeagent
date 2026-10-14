#!/usr/bin/env bash
# Shared settings for the HPC job scripts. Defaults: code in this repository,
# data and outputs under $SCRATCH/smvqa_gaze_data (override by exporting the
# variables before sbatch, or edit the defaults below once).
set -euo pipefail

: "${SMVQA_REPO:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${SMVQA_TRANSFER:=${SCRATCH:-$HOME}/smvqa_gaze_data/transfer_ready_v2}"  # uploaded data package
: "${SMVQA_WORK:=${SCRATCH:-$HOME}/smvqa_gaze_data/work}"                     # caches, memory, preds, results
: "${SMVQA_ENV:=smvqa-gaze-hpc}"
: "${HF_HOME:=${SCRATCH:-$HOME}/.cache/huggingface}"
export SMVQA_REPO SMVQA_TRANSFER SMVQA_WORK HF_HOME
# Compute nodes often have no internet: use the models cached by hpc-fetch-models.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export SMVQA_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
mkdir -p "$SMVQA_WORK/logs"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$SMVQA_ENV"
fi
cd "$SMVQA_REPO"

smvqa() { python -m smvqa_gaze "$@" --data "$SMVQA_TRANSFER" --work "$SMVQA_WORK"; }

start_vllm() {
  # Arguments: extra environment overrides, e.g. VLLM_MAX_NUM_SEQS=32
  env "$@" bash scripts/start_vllm.sh > "$SMVQA_WORK/logs/vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
  smvqa hpc-wait-server --base-url "$SMVQA_BASE_URL" --timeout 1800
}

account_job() {
  # Count this job's whole wall time (one GPU) in the shared GPU-hour ledger;
  # inner commands then only read the ledger instead of double counting.
  export SMVQA_JOB_ACCOUNTED=1
  smvqa hpc-budget --label "$1" --heartbeat-until-pid $$ > /dev/null 2>&1 &
}
