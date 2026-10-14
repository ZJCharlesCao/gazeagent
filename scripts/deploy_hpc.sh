#!/usr/bin/env bash
# Create/update the smvqa-gaze-hpc conda env and start vLLM (README section 5.1).
# Use --no-serve on a login node to only build the environment.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="smvqa-gaze-hpc"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; initialize Conda before running this script." >&2
  exit 1
fi

cd "$ROOT_DIR"
if conda run --no-capture-output --name "$ENV_NAME" python --version >/dev/null 2>&1; then
  conda env update --name "$ENV_NAME" --file "$ROOT_DIR/env/environment_hpc.yml"
else
  conda env create --file "$ROOT_DIR/env/environment_hpc.yml"
fi

if [[ "${1:-}" == "--no-serve" ]]; then
  echo "Environment $ENV_NAME is ready."
  exit 0
fi
exec conda run --no-capture-output --name "$ENV_NAME" bash "$ROOT_DIR/scripts/start_vllm.sh"
