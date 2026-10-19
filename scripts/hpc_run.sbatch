#!/usr/bin/env bash
#SBATCH --job-name=smvqa-h3
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=zccao@utexas.edu
# Step 3 (H3 + H4): run the confirmed COMPUTE_PLAN. Safe to resubmit after a
# time-limit kill: finished tasks are skipped and predictions resume per question.
# Can run on its own without the dev check and benchmark jobs: the first
# submission then caches frames and gaze checks (hpc-prepare) and writes a plan
# of all P0-P2 tasks with default server settings (hpc-plan --no-bench).
# The group runner stops launching at 22 GPU h and stops everything at 23.5 h.
# SLURM runs a copy of this file from its spool directory, so locate the
# repository from the submission directory (the repo root or its scripts/).
set -euo pipefail
for SMVQA_REPO in "${SMVQA_REPO:-}" "${SLURM_SUBMIT_DIR:-$PWD}" "${SLURM_SUBMIT_DIR:-$PWD}/.."; do
  [ -f "$SMVQA_REPO/scripts/hpc_common.sh" ] && break
done
[ -f "$SMVQA_REPO/scripts/hpc_common.sh" ] || { echo "Repository not found: submit from its root or export SMVQA_REPO" >&2; exit 1; }
SMVQA_REPO="$(cd -- "$SMVQA_REPO" && pwd)"
source "$SMVQA_REPO/scripts/hpc_common.sh"
account_job h3-run
PLAN="$SMVQA_WORK/results/compute_plan.json"
# The dev check normally runs hpc-prepare; its manifest marks completion.
[ -f "$SMVQA_WORK/cache/manifest.json" ] || smvqa hpc-prepare --skip-verify --workers "$(( $(nproc) < 16 ? $(nproc) : 16 ))"
[ -f "$PLAN" ] || smvqa hpc-plan --no-bench
SERVER=$(python -c "import json;s=json.load(open('$PLAN'))['server'];print(f\"VLLM_MAX_NUM_SEQS={s['max_num_seqs']} VLLM_MAX_NUM_BATCHED_TOKENS={s['max_num_batched_tokens']}\")")
# Other GPU models (GroundingDINO, CLIP, EasyOCR, Contriever) share the GPU
# with vLLM in up to 4 parallel tasks. At 0.80 vLLM grew to ~88 of 95 GB and
# the tasks' allocations crashed it (job 1076207).
VLLM_ENV=($SERVER VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}")
start_vllm "${VLLM_ENV[@]}"

# Parallelism: at most SMVQA_MAX_PARALLEL tasks, and a task starts only while
# free CPU RAM and GPU memory are above the limits below (one start per poll).
GROUP=(--plan "$PLAN" --base-url "$SMVQA_BASE_URL"
  --max-parallel "${SMVQA_MAX_PARALLEL:-4}"
  --min-free-ram-gb "${SMVQA_MIN_FREE_RAM_GB:-24}"
  --min-free-gpu-gb "${SMVQA_MIN_FREE_GPU_GB:-10}"
  --max-consecutive-failures "${SMVQA_MAX_CONSECUTIVE_FAILURES:-4}")
# hpc-group aborts with exit code 3 (vLLM stopped answering for ~90 s),
# 4 (that many tasks failed in a row) or 5 (memory stayed critical). The job
# then ends at once with that code instead of running on. Finished tasks keep
# their DONE markers, so a resubmission resumes. Set SMVQA_MAX_VLLM_RESTARTS>0
# to restart vLLM after code 3 instead of ending.
run_group() {
  local restarts=0 code
  while true; do
    code=0
    smvqa hpc-group "${GROUP[@]}" "$@" || code=$?
    [ "$code" -eq 0 ] && return 0
    case "$code" in
      3) reason="vLLM server stopped answering; see $SMVQA_WORK/logs/vllm_${SLURM_JOB_ID:-local}.log" ;;
      4) reason="too many tasks failed in a row; see $SMVQA_WORK/logs/tasks/" ;;
      5) reason="memory stayed critical; see $SMVQA_WORK/logs/group_events.jsonl" ;;
      *) reason="hpc-group crashed" ;;
    esac
    if [ "$code" -eq 3 ] && [ "$restarts" -lt "${SMVQA_MAX_VLLM_RESTARTS:-0}" ]; then
      restarts=$((restarts + 1))
      echo "ABORT CHECK: $reason; restarting vLLM ($restarts)" >&2
      stop_vllm
      start_vllm "${VLLM_ENV[@]}"
      continue
    fi
    echo "JOB STOPPED (exit $code): $reason" >&2
    stop_vllm
    exit "$code"
  done
}
run_group --priorities P0
if python -c "import json,sys;sys.exit(0 if json.load(open('$PLAN'))['variant_gate'] else 1)"; then
  [ -f "$SMVQA_WORK/cache/best_c3_variant.json" ] || smvqa hpc-select-variant --base-url "$SMVQA_BASE_URL"
fi
run_group --priorities P1,P2,P3
smvqa hpc-evaluate
smvqa hpc-analyze
smvqa hpc-budget
