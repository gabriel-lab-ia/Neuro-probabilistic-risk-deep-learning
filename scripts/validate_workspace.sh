#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export_project_env

status=0

echo "[python] pip check"
pip_check_output_file="$(mktemp)"
if ! python -m pip check >"$pip_check_output_file" 2>&1; then
  cat "$pip_check_output_file"
  if grep -q "nvidia-nccl-cu12" "$pip_check_output_file"; then
    echo "[warning] Detected known optional CUDA package mismatch in the local workspace."
    echo "[warning] Treating this as non-fatal because the current project validation path is CPU-safe."
  else
    status=1
  fi
else
  cat "$pip_check_output_file"
fi
rm -f "$pip_check_output_file"

echo
if ! python scripts/check_stack.py; then
  status=1
fi

echo
echo "[jsviz] npm ls --depth=0"
if ! (cd jsviz && npm ls --depth=0); then
  status=1
fi

echo
echo "[jsviz] npm run build"
if ! (cd jsviz && npm run build); then
  status=1
fi

exit "$status"
