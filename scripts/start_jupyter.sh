#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export_project_env

jupyter lab --ip=127.0.0.1 --port="${JUPYTER_PORT:-8888}" --no-browser
