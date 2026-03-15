#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export_project_env

python -m ipykernel install \
  --sys-prefix \
  --name deeplearning-py311 \
  --display-name "Python (deeplearning-py311)"
