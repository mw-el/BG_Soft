#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] Conda wurde nicht gefunden. Bitte Miniconda/Anaconda installieren." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate BG-Soft

python bg_soft_gui.py "$@"
