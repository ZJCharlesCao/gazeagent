#!/usr/bin/env bash
#SBATCH --job-name=smvqa-dev
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=zccao@utexas.edu
# Step 1 on HPC: verify data, prepare caches, audit tokens and run every
# condition on a few dev questions through both pipelines. Must finish with
# "DEV CHECK PASSED" before submitting the benchmark job.
# SLURM runs a copy of this file from its spool directory, so locate the
# repository from the submission directory (the repo root or its scripts/).
set -euo pipefail
for SMVQA_REPO in "${SMVQA_REPO:-}" "${SLURM_SUBMIT_DIR:-$PWD}" "${SLURM_SUBMIT_DIR:-$PWD}/.."; do
  [ -f "$SMVQA_REPO/scripts/hpc_common.sh" ] && break
done
[ -f "$SMVQA_REPO/scripts/hpc_common.sh" ] || { echo "Repository not found: submit from its root or export SMVQA_REPO" >&2; exit 1; }
SMVQA_REPO="$(cd -- "$SMVQA_REPO" && pwd)"
source "$SMVQA_REPO/scripts/hpc_common.sh"
account_job h0-dev-check

smvqa hpc-prepare --workers "${SLURM_CPUS_PER_TASK:-8}"
smvqa hpc-token-audit
start_vllm
SMOKE=(--split dev --limit 4 --smoke-window-s 300 --base-url "$SMVQA_BASE_URL")
for pipeline in videorag egobutler; do
  for condition in C0 C1 C3-T C3-V C3-M C4 C5 C6a0.5 C8shift-V C8cross-M C9-T; do
    smvqa hpc-run --pipeline "$pipeline" --tier B1 --condition "$condition" "${SMOKE[@]}" --beta 0.5
  done
  smvqa hpc-run --pipeline "$pipeline" --tier B2 --condition C0 "${SMOKE[@]}"
done
smvqa hpc-evaluate
echo "DEV CHECK PASSED"
