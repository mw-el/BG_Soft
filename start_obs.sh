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
if pgrep obs > /dev/null 2>&1; then
    echo "[✓] OBS is already running"
    sleep 1
else
    echo "[→] Configuring Automation profile to use Automation scene collection..."

    # Ensure the Automation profile's basic.ini has the correct scene collection
    AUTOMATION_PROFILE_INI="$HOME/.config/obs-studio/basic/profiles/Automation/basic.ini"
    if [[ -f "$AUTOMATION_PROFILE_INI" ]]; then
        sed -i '/^SceneCollection=/d' "$AUTOMATION_PROFILE_INI"
    fi
    echo "SceneCollection=Automation" >> "$AUTOMATION_PROFILE_INI"

    echo "[→] Launching native OBS Studio..."

    obs > /tmp/obs.log 2>&1 &

    OBS_PID=$!
    echo "[→] OBS PID: $OBS_PID"
    echo "[→] Waiting for OBS to start..."
    sleep 5

    if pgrep obs > /dev/null 2>&1; then
        echo "[✓] OBS started successfully"
    else
        echo "[✗] OBS failed to start"
        cat /tmp/obs.log
        exit 1
    fi
fi

echo
echo "============================================"
echo "OBS is running"
echo "============================================"
echo
echo "Configuration status:"
echo
echo "1. WebSocket Server"
if grep -q "websocket_server_enabled" "$HOME/.config/obs-studio/global.ini" 2>/dev/null; then
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
