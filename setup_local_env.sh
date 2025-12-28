#!/usr/bin/env bash
# Set up local (no-OBS) rendering dependencies for BG-Soft.
# Installs: onnxruntime (CPU), numpy, ffmpeg-python (and existing deps from environment.yml).

set -euo pipefail

ENV_NAME="BG-Soft"
ENV_FILE="$(dirname "$0")/environment.yml"

echo "==> Checking for conda..."
if command -v conda >/dev/null 2>&1; then
  echo "==> conda found. Updating/creating environment '${ENV_NAME}' using ${ENV_FILE}"
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune || conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
  echo "==> Done. Activate with: conda activate ${ENV_NAME}"
  exit 0
fi

echo "==> conda not found. Falling back to pip in current interpreter."
REQS=(
  "onnxruntime-gpu>=1.19,<1.20"
  "numpy>=1.26"
  "ffmpeg-python>=0.2.0"
)
pip install "${REQS[@]}"
echo "==> Done. If ffmpeg CLI is missing, install it via your package manager (apt/yum/brew)."
