#!/usr/bin/env bash
#SBATCH --job-name=smvqa-rehearsal
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Parallel rehearsal of the H3 run: every task of the compute plan through the
# same group runner (4 tasks at a time next to one vLLM server, as in
# hpc_run.sbatch), but on a few dev questions with memory built only near each
# question. Catches GPU/host memory and server crashes that the serial smoke
# test cannot. Writes to its own work directory (cache files are copied, frames
# are shared read-mostly). Ends with "REHEARSAL PASSED".
# Submit from the repo root (sbatch -A <account> scripts/hpc_rehearsal.sbatch)
# or run with bash inside an interactive GPU session.
set -euo pipefail
REAL_WORK="${SMVQA_WORK:-${SCRATCH:-$HOME}/gazeagent/outputs}"
export SMVQA_WORK="${SMVQA_WORK_REHEARSAL:-${SCRATCH:-$HOME}/gazeagent/outputs_rehearsal}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for candidate in "${SMVQA_REPO:-}" "${SLURM_SUBMIT_DIR:-$PWD}" "${SLURM_SUBMIT_DIR:-$PWD}/.."; do
  [ -f "$HERE/hpc_common.sh" ] || HERE="$candidate/scripts"
done
[ -f "$HERE/hpc_common.sh" ] || { echo "Repository not found: submit from its root or export SMVQA_REPO" >&2; exit 1; }
source "$HERE/hpc_common.sh"
export SMVQA_JOB_ACCOUNTED=1  # not part of the experiment's GPU-hour ledger
LIMIT="${REHEARSAL_LIMIT:-6}"
PLAN="$REAL_WORK/results/compute_plan.json"
[ -f "$PLAN" ] || { echo "No plan at $PLAN: run hpc-plan first" >&2; exit 1; }

# Fresh work directory with the real job's small prepared caches.
rm -rf "$SMVQA_WORK/memory" "$SMVQA_WORK/preds" "$SMVQA_WORK/logs/tasks" \
  "$SMVQA_WORK/logs/group_events.jsonl" "$SMVQA_WORK/logs/group_status.json"
mkdir -p "$SMVQA_WORK/cache" "$SMVQA_WORK/logs"
for item in manifest.json label_index.json shift_check.json saliency_boxes; do
  [ -e "$REAL_WORK/cache/$item" ] && cp -r "$REAL_WORK/cache/$item" "$SMVQA_WORK/cache/"
done
[ -e "$SMVQA_WORK/cache/frames_1fps" ] || ln -s "$REAL_WORK/cache/frames_1fps" "$SMVQA_WORK/cache/frames_1fps"

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "data=$SMVQA_TRANSFER work=$SMVQA_WORK limit=$LIMIT"
# Record GPU memory during the rehearsal (peak is printed at the end).
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 10 \
  > "$SMVQA_WORK/logs/gpu_mem.csv" 2>/dev/null &
SMI_PID=$!

VLLM_ENV=(VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}")
start_vllm "${VLLM_ENV[@]}"
trap 'kill $VLLM_PID $SMI_PID 2>/dev/null || true' EXIT

started=$(date +%s)
# No automatic vLLM restart here: a server crash must fail the rehearsal.
smvqa hpc-group --plan "$PLAN" --base-url "$SMVQA_BASE_URL" --split dev \
  --limit "$LIMIT" --smoke-window-s 300 --max-parallel "${SMVQA_MAX_PARALLEL:-4}" \
  --min-free-ram-gb "${SMVQA_MIN_FREE_RAM_GB:-24}" --min-free-gpu-gb "${SMVQA_MIN_FREE_GPU_GB:-10}" > "$SMVQA_WORK/logs/group_output.log" \
  || echo "hpc-group exited with code $?"
smvqa hpc-evaluate > "$SMVQA_WORK/logs/evaluate.json"
echo "group runtime: $(( ($(date +%s) - started) / 60 )) min"
awk -F', ' '{ if ($2 > max) max = $2 } END { printf "peak GPU memory: %.1f GiB\n", max / 1024 }' "$SMVQA_WORK/logs/gpu_mem.csv"

python - "$SMVQA_WORK/logs/group_status.json" <<'EOF'
import json, sys
summary = json.load(open(sys.argv[1]))
status = summary["status"]
bad = {name: state for name, state in status.items() if state != "done"}
print(f"{len(status) - len(bad)}/{len(status)} tasks done; server_down={summary.get('server_down')}")
for name, state in sorted(bad.items()):
    print(f"  NOT DONE {name}: {state}")
sys.exit(1 if bad or summary.get("server_down") else 0)
EOF
echo "REHEARSAL PASSED"
