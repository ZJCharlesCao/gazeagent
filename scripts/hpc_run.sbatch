#!/usr/bin/env bash
#SBATCH --job-name=smvqa-h3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
# Step 3 (H3 + H4): run the confirmed COMPUTE_PLAN. Safe to resubmit after a
# time-limit kill: finished tasks are skipped and predictions resume per question.
# The group runner stops launching at 22 GPU h and stops everything at 23.5 h.
# SLURM runs a copy of this file from its spool directory, so locate the
# repository from the submission directory: submit from the repo root.
source "${SMVQA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/hpc_common.sh"
account_job h3-run
PLAN="$SMVQA_WORK/results/compute_plan.json"
SERVER=$(python -c "import json;s=json.load(open('$PLAN'))['server'];print(f\"VLLM_MAX_NUM_SEQS={s['max_num_seqs']} VLLM_MAX_NUM_BATCHED_TOKENS={s['max_num_batched_tokens']}\")")
# Other GPU models (GroundingDINO, CLIP, Contriever) share the GPU with vLLM.
start_vllm $SERVER VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"

GROUP=(--plan "$PLAN" --base-url "$SMVQA_BASE_URL")
smvqa hpc-group "${GROUP[@]}" --priorities P0
if python -c "import json,sys;sys.exit(0 if json.load(open('$PLAN'))['variant_gate'] else 1)"; then
  [ -f "$SMVQA_WORK/cache/best_c3_variant.json" ] || smvqa hpc-select-variant --base-url "$SMVQA_BASE_URL"
fi
smvqa hpc-group "${GROUP[@]}" --priorities P1,P2,P3
smvqa hpc-evaluate
smvqa hpc-analyze
smvqa hpc-budget
