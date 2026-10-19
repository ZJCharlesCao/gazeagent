#!/usr/bin/env bash
#SBATCH --job-name=smvqa-smoke
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=zccao@utexas.edu
#SBATCH --output=%x-%j.out
# Smoke test of the main experiment (hpc-run as used by H3): both pipelines,
# baseline and gaze conditions, on a few dev questions with memory built only
# near each question. Writes to its own work directory so partial smoke
# memory is never reused by the real run. Ends with "SMOKE TEST PASSED".
# Submit from the repository root: sbatch -A <account> scripts/hpc_smoke.sbatch
# (the job also runs directly with bash inside an interactive GPU session).
set -euo pipefail
export SMVQA_WORK="${SMVQA_WORK_SMOKE:-${SCRATCH:-$HOME}/gazeagent/outputs_smoke}"
# Run with bash: the script's own directory. Under sbatch (spooled copy): the
# submission directory (the repo root or its scripts/).
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for candidate in "${SMVQA_REPO:-}" "${SLURM_SUBMIT_DIR:-$PWD}" "${SLURM_SUBMIT_DIR:-$PWD}/.."; do
  [ -f "$HERE/hpc_common.sh" ] || HERE="$candidate/scripts"
done
[ -f "$HERE/hpc_common.sh" ] || { echo "Repository not found: submit from its root or export SMVQA_REPO" >&2; exit 1; }
source "$HERE/hpc_common.sh"
# The smoke test is not part of the experiment's GPU-hour ledger.
export SMVQA_JOB_ACCOUNTED=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "data=$SMVQA_TRANSFER work=$SMVQA_WORK"

# Video-RAG loads CLIP, GroundingDINO, EasyOCR and Contriever on the same GPU.
start_vllm VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
SMOKE=(--split dev --limit "${SMOKE_LIMIT:-4}" --smoke-window-s 300 --base-url "$SMVQA_BASE_URL")
for pipeline in egobutler videorag; do
  for condition in C0 C3-V C5; do
    smvqa hpc-run --pipeline "$pipeline" --tier B1 --condition "$condition" "${SMOKE[@]}" --beta 0.5
  done
  smvqa hpc-run --pipeline "$pipeline" --tier B2 --condition C0 "${SMOKE[@]}"
done
smvqa hpc-evaluate
echo "SMOKE TEST PASSED"
