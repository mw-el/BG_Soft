#!/bin/bash
# Launch BG-Soft with automatic OBS Automation profile startup
# This is used by the desktop file to ensure OBS is running properly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use explicit Python from BG-Soft conda environment
# Hardcode home directory for this user (required for desktop launcher)
HOME="/home/matthias"
export PYTHON="$HOME/miniconda3/envs/BG-Soft/bin/python"

# Verify conda environment exists and Python is available
if [ ! -x "$PYTHON" ]; then
    echo "[✗] Error: Python not found in BG-Soft conda environment"
    echo "[✗] Expected: $PYTHON"
    exit 1
fi

# Set PATH to use BG-Soft conda environment
export PATH="$HOME/miniconda3/envs/BG-Soft/bin:$HOME/miniconda3/bin:$HOME/bin:$PATH"

# Ensure CUDA/cuDNN libs are discoverable (Phase 1 GPU acceleration)
# Priority: conda-installed libs (cuDNN 9.x via conda) > pip-installed libs
CUDA_LIBS=(
    "$HOME/miniconda3/envs/BG-Soft/lib"           # Conda-installed cuDNN, CUDA libs
    "$HOME/miniconda3/envs/BG-Soft/lib64"         # 64-bit CUDA libs
    "$HOME/miniconda3/envs/BG-Soft/lib/python3.11/site-packages/nvidia/cudnn/lib"
    "$HOME/miniconda3/envs/BG-Soft/lib/python3.11/site-packages/nvidia/cublas/lib"
)
for d in "${CUDA_LIBS[@]}"; do
    if [[ -d "$d" ]]; then
        export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    fi
done

# Verify GPU acceleration is available
echo "[→] Checking GPU acceleration availability..."
"$PYTHON" << 'PYEOF'
import onnxruntime as ort
providers = ort.get_available_providers()
has_gpu = any("CUDA" in p for p in providers)
status = "✓ GPU (CUDA) acceleration ENABLED" if has_gpu else "⚠ GPU acceleration not available"
print(f"[✓] ONNX providers: {', '.join(providers)}")
print(f"[{('✓' if has_gpu else '!')}] Status: {status}")
PYEOF
echo ""

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
else
    echo "[✓] OBS is already running"
fi

# Launch the GUI immediately - WebSocket connection retries will handle delays
echo "[→] Launching BG-Soft GUI..."

"$PYTHON" "$SCRIPT_DIR/bg_soft_gui.py"
