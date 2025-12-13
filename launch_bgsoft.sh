#!/bin/bash
# Launch BG-Soft with automatic OBS Automation profile startup
# This is used by the desktop file to ensure OBS is running properly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Automation profile exists, if not set it up
if [[ ! -d "$HOME/.config/obs-studio/basic/profiles/Automation" ]]; then
    echo "[!] Automation profile not found"
    echo "[→] Setting up OBS Automation profile..."
    "$SCRIPT_DIR/setup_obs_automation.sh"
fi

# Check if OBS is running with Automation profile
if ! pgrep -f "obs.*--profile.*Automation" > /dev/null 2>&1; then
    echo "[→] Starting OBS with Automation profile..."
    "$SCRIPT_DIR/start_obs.sh" &
    sleep 3  # Give OBS time to start
fi

# Launch the GUI
echo "[→] Launching BG-Soft GUI..."
exec python "$SCRIPT_DIR/bg_soft_gui.py"
