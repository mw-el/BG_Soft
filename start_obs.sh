#!/bin/bash
# Start OBS with proper configuration

export PATH=~/bin:$PATH

echo "Starting OBS Studio..."
echo

# Check if OBS is already running
if pgrep obs > /dev/null; then
    echo "[✓] OBS is already running"
    sleep 1
else
    echo "[→] Launching OBS..."
    flatpak run com.obsproject.Studio > /tmp/obs.log 2>&1 &
    OBS_PID=$!
    echo "[→] OBS PID: $OBS_PID"
    echo "[→] Waiting for OBS to start..."
    sleep 5

    if pgrep obs > /dev/null; then
        echo "[✓] OBS started successfully"
    else
        echo "[✗] OBS failed to start"
        cat /tmp/obs.log
        exit 1
    fi
fi

echo
echo "OBS is running and ready"
echo
echo "Next steps:"
echo "1. Enable WebSocket: Tools → WebSocket Server Settings → Enable"
echo "2. Create scene 'BR-Render' with 'Media Source' input"
echo "3. Run: bash quick_start.sh"
echo
