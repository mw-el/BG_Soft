#!/usr/bin/env bash
# Cleanup OBS configuration to prevent missing files dialog at startup
# Run this if OBS shows a "Missing Files" dialog when starting

set -euo pipefail

OBS_CONFIG_DIR="$HOME/.config/obs-studio"
SCENES_JSON="$OBS_CONFIG_DIR/basic/scenes.json"

if [[ ! -d "$OBS_CONFIG_DIR" ]]; then
  echo "[!] OBS config directory not found: $OBS_CONFIG_DIR"
  echo "[!] OBS may not have been configured yet"
  exit 1
fi

if [[ ! -f "$SCENES_JSON" ]]; then
  echo "[!] OBS scenes configuration not found: $SCENES_JSON"
  exit 1
fi

echo "[→] Backing up OBS scenes configuration..."
cp "$SCENES_JSON" "$SCENES_JSON.backup"
echo "[✓] Backup created: $SCENES_JSON.backup"

echo "[→] Clearing file paths from media sources..."
# Use jq to remove/empty the 'local_file' settings in all media sources
jq '
  .sources[] |= (
    if .type == "ffmpeg_source" then
      .settings.local_file = ""
    else
      .
    end
  )
' "$SCENES_JSON" > "$SCENES_JSON.tmp"

mv "$SCENES_JSON.tmp" "$SCENES_JSON"
echo "[✓] Cleared media file paths from OBS configuration"

echo ""
echo "[✓] OBS configuration cleaned up!"
echo "You can now start OBS without the 'Missing Files' dialog."
echo ""
echo "If you want to restore the original configuration:"
echo "  cp $SCENES_JSON.backup $SCENES_JSON"
echo ""
