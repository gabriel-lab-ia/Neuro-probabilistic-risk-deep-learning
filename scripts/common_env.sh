#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT

ensure_project_runtime_dirs() {
  mkdir -p \
    "$PROJECT_ROOT/.cache/pip" \
    "$PROJECT_ROOT/.cache/pycache" \
    "$PROJECT_ROOT/.cache/matplotlib" \
    "$PROJECT_ROOT/.ipython" \
    "$PROJECT_ROOT/.jupyter/data" \
    "$PROJECT_ROOT/.jupyter/runtime"
}

export_project_env() {
  ensure_project_runtime_dirs

  export JUPYTER_CONFIG_DIR="$PROJECT_ROOT/.jupyter"
  export JUPYTER_DATA_DIR="$PROJECT_ROOT/.jupyter/data"
  export JUPYTER_RUNTIME_DIR="$PROJECT_ROOT/.jupyter/runtime"
  export IPYTHONDIR="$PROJECT_ROOT/.ipython"
  export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
  export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
  export PYTHONPYCACHEPREFIX="$PROJECT_ROOT/.cache/pycache"
  export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
}
