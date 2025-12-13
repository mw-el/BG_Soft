#!/bin/bash
# Start OBS with Automation profile for BG-Soft

set -euo pipefail

export PATH=~/bin:$PATH

echo "Starting OBS Studio with Automation profile..."
echo

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if setup has been run
if [[ ! -d "$HOME/.config/obs-studio/basic/profiles/Automation" ]]; then
    echo "[!] Automation profile not found!"
    echo "[→] Run this first:"
    echo "    $SCRIPT_DIR/setup_obs_automation.sh"
    echo
    exit 1
fi

# Check if OBS is already running
if pgrep -f "obs.*--profile.*Automation" > /dev/null 2>&1; then
    echo "[✓] OBS Automation profile is already running"
    sleep 1
else
    echo "[→] Launching OBS with Automation profile..."

    # Try native OBS first, fall back to flatpak
    if command -v obs >/dev/null 2>&1; then
        obs --profile "Automation" --collection "Automation" > /tmp/obs.log 2>&1 &
    else
        flatpak run com.obsproject.Studio --profile "Automation" --collection "Automation" > /tmp/obs.log 2>&1 &
    fi

    OBS_PID=$!
    echo "[→] OBS PID: $OBS_PID"
    echo "[→] Waiting for OBS to start..."
    sleep 5

    if pgrep -f "obs.*--profile.*Automation" > /dev/null 2>&1; then
        echo "[✓] OBS started successfully with Automation profile"
    else
        echo "[✗] OBS failed to start"
        cat /tmp/obs.log
        exit 1
    fi
fi

echo
echo "============================================"
echo "OBS Automation Profile is running"
echo "============================================"
echo
echo "Configuration status:"
echo
echo "1. WebSocket Server"
if grep -q "websocket_server_enabled" "$HOME/.config/obs-studio/global.conf" 2>/dev/null; then
    echo "   [✓] WebSocket Server is enabled"
else
    echo "   [!] WebSocket Server needs to be enabled"
    echo "      Go to: Tools → WebSocket Server Settings → Enable"
fi
echo
echo "2. Scene Configuration"
echo "   [→] Edit the 'BR-Render' scene:"
echo "      - Select 'Media Source 2' in the Sources panel"
echo "      - Add your Background-Removal filter in Properties"
echo "      - Save when done"
echo
echo "3. Start BG-Soft"
echo "   [→] Run in another terminal:"
echo "      cd $SCRIPT_DIR"
echo "      python bg_soft_gui.py"
echo
echo "============================================"
echo
