#!/usr/bin/env bash
# Shared settings for the HPC job scripts. Defaults: code in this repository,
# data and outputs under $SCRATCH/smvqa_gaze_data (override by exporting the
# variables before sbatch, or put cluster-specific values in scripts/hpc_site.sh,
# which is sourced first when it exists and is not tracked by Git).
set -euo pipefail

: "${SMVQA_REPO:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
[ -f "$SMVQA_REPO/scripts/hpc_site.sh" ] && source "$SMVQA_REPO/scripts/hpc_site.sh"
: "${SMVQA_TRANSFER:=${SCRATCH:-$HOME}/smvqa_gaze_data/transfer_ready_v2}"  # uploaded data package
: "${SMVQA_WORK:=${SCRATCH:-$HOME}/smvqa_gaze_data/work}"                     # caches, memory, preds, results
: "${SMVQA_ENV:=smvqa-gaze-hpc}"
: "${HF_HOME:=${SCRATCH:-$HOME}/.cache/huggingface}"
export SMVQA_REPO SMVQA_TRANSFER SMVQA_WORK HF_HOME
# Compute nodes often have no internet: use the models cached by hpc-fetch-models.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Each parallel task otherwise starts one thread per core in every library
# pool; job 1076207 hit "can't start new thread".
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export SMVQA_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
mkdir -p "$SMVQA_WORK/logs"

# Batch shells do not run ~/.bashrc: load conda from SMVQA_CONDA_ROOT if needed.
if ! command -v conda >/dev/null 2>&1 && [ -n "${SMVQA_CONDA_ROOT:-}" ]; then
  # shellcheck disable=SC1091
  source "$SMVQA_CONDA_ROOT/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$SMVQA_ENV"
fi
cd "$SMVQA_REPO"
# Fail now rather than on every later command if the environment is missing.
python -c "import smvqa_gaze" || { echo "smvqa_gaze not importable: check SMVQA_CONDA_ROOT and SMVQA_ENV=$SMVQA_ENV" >&2; exit 1; }

smvqa() { python -m smvqa_gaze "$@" --data "$SMVQA_TRANSFER" --work "$SMVQA_WORK"; }

start_vllm() {
  # Arguments: extra environment overrides, e.g. VLLM_MAX_NUM_SEQS=32
  # Append: a restart within the same job keeps the log of the crash.
  env "$@" bash scripts/start_vllm.sh >> "$SMVQA_WORK/logs/vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
  # Wait for the server, but fail at once if vLLM exits during startup.
  smvqa hpc-wait-server --base-url "$SMVQA_BASE_URL" --timeout 1800 &
  local waiter=$!
  while kill -0 "$waiter" 2>/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      kill "$waiter" 2>/dev/null || true
      echo "vLLM exited during startup; see $SMVQA_WORK/logs/vllm_${SLURM_JOB_ID:-local}.log" >&2
      return 1
    fi
    sleep 5
  done
  wait "$waiter"
}

stop_vllm() {
  [ -n "${VLLM_PID:-}" ] || return 0
  kill "$VLLM_PID" 2>/dev/null || true
  for _ in $(seq 60); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  # Children such as the EngineCore can outlive the API server and hold the GPU.
  pkill -9 -f "vllm serve" 2>/dev/null || true
  VLLM_PID=
}

account_job() {
  # Count this job's whole wall time (one GPU) in the shared GPU-hour ledger;
  # inner commands then only read the ledger instead of double counting.
  export SMVQA_JOB_ACCOUNTED=1
  smvqa hpc-budget --label "$1" --heartbeat-until-pid $$ > /dev/null 2>&1 &
}
