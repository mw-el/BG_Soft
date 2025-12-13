#!/bin/bash
# Launch BG-Soft with automatic OBS Automation profile startup
# This is used by the desktop file to ensure OBS is running properly

set -euo pipefail

# Ensure we have a proper environment when launched from desktop
export PATH="$HOME/miniconda3/bin:$HOME/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cleanup function to stop OBS when BG-Soft exits
cleanup() {
    local exit_code=$?
    echo "[→] Cleaning up OBS..."

    # Try graceful shutdown first
    if pgrep obs > /dev/null 2>&1; then
        echo "[→] Sending graceful shutdown signal to OBS..."
        pkill -TERM obs 2>/dev/null || true
        sleep 2
    fi

    # Force kill any remaining OBS processes (native)
    if pgrep obs > /dev/null 2>&1; then
        echo "[!] OBS still running, force killing..."
        pkill -9 obs 2>/dev/null || true
    fi

    # Force kill any remaining Flatpak OBS processes
    if pgrep -f "bwrap.*obs" > /dev/null 2>&1; then
        echo "[!] Flatpak OBS still running, force killing..."
        pkill -9 -f "bwrap.*obs" 2>/dev/null || true
    fi

    sleep 1

    # Verify cleanup succeeded
    if pgrep obs > /dev/null 2>&1; then
        echo "[✗] Warning: OBS processes still running after cleanup"
        pgrep -l obs
    else
        echo "[✓] OBS cleaned up successfully"
    fi

    return $exit_code
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

# Log to file when launched from desktop (no terminal)
if [[ ! -t 1 ]]; then
    exec >> "$HOME/.local/share/bgsoft/launch.log" 2>&1
    mkdir -p "$HOME/.local/share/bgsoft"
fi

python "$SCRIPT_DIR/bg_soft_gui.py"
