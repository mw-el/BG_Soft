#!/bin/bash
# Launch BG-Soft with automatic OBS Automation profile startup
# This is used by the desktop file to ensure OBS is running properly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Flag to track if we started OBS
OBS_STARTED_BY_US=0

# Cleanup function to stop OBS when BG-Soft exits
cleanup() {
    if [[ $OBS_STARTED_BY_US -eq 1 ]]; then
        echo "[→] Cleaning up OBS (started by BG-Soft)..."
        # Kill all OBS instances that we started
        pkill -9 -f "bwrap.*obs" 2>/dev/null || true
        pkill -9 obs 2>/dev/null || true
        pkill -9 -f "com.obsproject.Studio" 2>/dev/null || true
        sleep 2
        echo "[✓] OBS cleaned up"
    else
        echo "[→] OBS was already running before BG-Soft started, leaving it running"
    fi
}

# Register cleanup to run on exit
trap cleanup EXIT

# Check if Automation profile exists, if not set it up
if [[ ! -d "$HOME/.config/obs-studio/basic/profiles/Automation" ]]; then
    echo "[!] Automation profile not found"
    echo "[→] Setting up OBS Automation profile..."
    "$SCRIPT_DIR/setup_obs_automation.sh"
fi

# Check if OBS (native) is running, exclude zombies
if ! (ps aux | grep -v grep | grep /usr/bin/obs | grep -v "Z" > /dev/null 2>&1); then
    echo "[→] Starting OBS..."
    OBS_STARTED_BY_US=1
    "$SCRIPT_DIR/start_obs.sh" &
    sleep 15  # Give OBS time to start (wait longer than start_obs.sh's 10s + overhead)
else
    echo "[✓] OBS is already running (started externally)"
fi

# Launch the GUI
echo "[→] Launching BG-Soft GUI..."
python "$SCRIPT_DIR/bg_soft_gui.py"
