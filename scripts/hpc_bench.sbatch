#!/usr/bin/env bash
#SBATCH --job-name=smvqa-h2
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=zccao@utexas.edu
# Step 2 (H2): measure throughput for a small server grid (vLLM is started and
# stopped by the benchmark itself), precompute Video-RAG OCR/CLIP features with
# vLLM stopped, then write results/COMPUTE_PLAN.md for the user to confirm.
# SLURM runs a copy of this file from its spool directory, so locate the
# repository from the submission directory (the repo root or its scripts/).
set -euo pipefail
for SMVQA_REPO in "${SMVQA_REPO:-}" "${SLURM_SUBMIT_DIR:-$PWD}" "${SLURM_SUBMIT_DIR:-$PWD}/.."; do
  [ -f "$SMVQA_REPO/scripts/hpc_common.sh" ] && break
done
[ -f "$SMVQA_REPO/scripts/hpc_common.sh" ] || { echo "Repository not found: submit from its root or export SMVQA_REPO" >&2; exit 1; }
SMVQA_REPO="$(cd -- "$SMVQA_REPO" && pwd)"
source "$SMVQA_REPO/scripts/hpc_common.sh"
account_job h2-bench

smvqa hpc-bench --launch-server --port "$VLLM_PORT" --base-url "$SMVQA_BASE_URL"
smvqa hpc-features --variants full,fovea-own-B1,fovea-own-B2
smvqa hpc-plan
echo "Review $SMVQA_WORK/results/COMPUTE_PLAN.md and confirm it before submitting hpc_run.sbatch"
