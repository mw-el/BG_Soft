#!/bin/bash
# Launch BG-Soft with automatic OBS Automation profile startup
# This is used by the desktop file to ensure OBS is running properly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cleanup function to stop OBS when BG-Soft exits
cleanup() {
    echo "[→] Cleaning up OBS..."
    pkill -f "obsproject.Studio" || true
    sleep 1
}

# Register cleanup to run on exit
trap cleanup EXIT

# Check if Automation profile exists, if not set it up
if [[ ! -d "$HOME/.config/obs-studio/basic/profiles/Automation" ]]; then
    echo "[!] Automation profile not found"
    echo "[→] Setting up OBS Automation profile..."
    "$SCRIPT_DIR/setup_obs_automation.sh"
fi

# Check if OBS (flatpak) is running
if ! pgrep -f "obsproject.Studio" > /dev/null 2>&1; then
    echo "[→] Starting OBS..."
    "$SCRIPT_DIR/start_obs.sh" &
    sleep 15  # Give OBS time to start (wait longer than start_obs.sh's 10s + overhead)
else
    echo "[✓] OBS is already running"
fi

# Launch the GUI
echo "[→] Launching BG-Soft GUI..."
exec python "$SCRIPT_DIR/bg_soft_gui.py"
