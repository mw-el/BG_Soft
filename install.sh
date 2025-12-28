#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$REPO_ROOT/environment.yml"
REQ_FILE="$REPO_ROOT/requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] Conda wurde nicht gefunden. Bitte Miniconda/Anaconda installieren und erneut versuchen." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] environment.yml wurde nicht gefunden: $ENV_FILE" >&2
  exit 1
fi

ENV_NAME="$(awk '/^name:/ {print $2; exit}' "$ENV_FILE")"
if [[ -z "$ENV_NAME" ]]; then
  echo "[ERROR] Konnte den Envnamen aus environment.yml nicht lesen." >&2
  exit 1
fi

echo "[INFO] Verwende Conda-Environment: $ENV_NAME"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[INFO] Environment existiert bereits -> Update mit environment.yml"
  conda env update --file "$ENV_FILE" --prune
else
  echo "[INFO] Environment existiert noch nicht -> wird erstellt"
  conda env create --file "$ENV_FILE"
fi

echo "[INFO] Installiere/aktualisiere Python-Abhängigkeiten via pip"
conda run -n "$ENV_NAME" python -m pip install -r "$REQ_FILE"

# Desktop file installation
echo "[INFO] Richte Desktop-Integration ein..."
DESKTOP_TEMPLATE="$REPO_ROOT/bgsoft.desktop.template"
DESKTOP_DIR="$HOME/.local/share/applications"
DESKTOP_FILE="$DESKTOP_DIR/bgsoft.desktop"

if [[ ! -f "$DESKTOP_TEMPLATE" ]]; then
  echo "[WARNING] Desktop-Template nicht gefunden: $DESKTOP_TEMPLATE"
else
  mkdir -p "$DESKTOP_DIR"

  # Get the conda python path for this environment
  CONDA_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"

  # Generate desktop file from template
  export CONDA_PREFIX
  export APP_DIR="$REPO_ROOT"
  envsubst < "$DESKTOP_TEMPLATE" > "$DESKTOP_FILE"

  if command -v desktop-file-validate >/dev/null 2>&1; then
    desktop-file-validate "$DESKTOP_FILE" && echo "[INFO] Desktop-Datei installiert: $DESKTOP_FILE" || echo "[WARNING] Desktop-Datei hat Validierungsfehler"
  else
    echo "[INFO] Desktop-Datei installiert: $DESKTOP_FILE"
  fi

  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$DESKTOP_DIR" && echo "[INFO] Desktop-Datenbank aktualisiert"
  fi
fi

# Ensure GUI script is not executable (to prevent shebang issues)
chmod -x "$REPO_ROOT/bg_soft_gui.py"

cat <<MSG

Installation abgeschlossen.

CLI-Verwendung:
  conda activate $ENV_NAME
  python render_with_obs.py /pfad/zur/datei.mp4

GUI-Starten:
  python bg_soft_gui.py
  oder über Desktop-Icon "BG-Soft"

MSG
