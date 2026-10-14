#!/usr/bin/env bash
#SBATCH --job-name=smvqa-h2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
# Step 2 (H2): measure throughput for a small server grid (vLLM is started and
# stopped by the benchmark itself), precompute Video-RAG OCR/CLIP features with
# vLLM stopped, then write results/COMPUTE_PLAN.md for the user to confirm.
# SLURM runs a copy of this file from its spool directory, so locate the
# repository from the submission directory: submit from the repo root.
source "${SMVQA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/hpc_common.sh"
account_job h2-bench

smvqa hpc-bench --launch-server --port "$VLLM_PORT" --base-url "$SMVQA_BASE_URL"
smvqa hpc-features --variants full,fovea-own-B1,fovea-own-B2
smvqa hpc-plan
echo "Review $SMVQA_WORK/results/COMPUTE_PLAN.md and confirm it before submitting hpc_run.sbatch"
