#!/bin/bash
# Launch BG-Soft with automatic OBS Automation profile startup
# This is used by the desktop file to ensure OBS is running properly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cleanup function to stop OBS when BG-Soft exits
cleanup() {
    echo "[→] Cleaning up OBS..."
    # Try graceful shutdown first
    pkill -TERM obs 2>/dev/null || true
    sleep 2
    # Force kill if still running
    pkill -9 -f "bwrap.*obs" 2>/dev/null || true
    pkill -9 obs 2>/dev/null || true
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

# Check if OBS is already running
if ! pgrep obs > /dev/null 2>&1; then
    echo "[→] Starting OBS..."
    "$SCRIPT_DIR/start_obs.sh"
    WAIT_TIME=3  # OBS just started, quick wait to ensure WebSocket is ready
else
    echo "[✓] OBS is already running"
    WAIT_TIME=1  # Already running, minimal wait
fi

# Wait for OBS to fully initialize and WebSocket server to be ready
echo "[→] Waiting for OBS to be ready..."
sleep $WAIT_TIME

# Launch the GUI (without exec to allow trap to trigger)
echo "[→] Launching BG-Soft GUI..."
python "$SCRIPT_DIR/bg_soft_gui.py"
