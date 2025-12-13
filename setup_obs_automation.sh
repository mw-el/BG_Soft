#!/usr/bin/env bash
# Setup dedicated OBS Automation scene collection for BG-Soft
# This creates a clean collection without file references that trigger the missing files dialog

set -euo pipefail

OBS_CONFIG_DIR="$HOME/.config/obs-studio"
OBS_PROFILES_DIR="$OBS_CONFIG_DIR/basic/profiles"
OBS_SCENES_DIR="$OBS_CONFIG_DIR/basic/scenes"

# Validate OBS is installed
if ! command -v obs >/dev/null 2>&1; then
  echo "[!] OBS Studio is not installed or not in PATH"
  exit 1
fi

# Create necessary directories
mkdir -p "$OBS_PROFILES_DIR"
mkdir -p "$OBS_SCENES_DIR"

AUTOMATION_PROFILE="Automation"
AUTOMATION_COLLECTION="Automation"

echo "[→] Setting up OBS Automation configuration..."

# Create Automation profile if it doesn't exist
if [[ ! -d "$OBS_PROFILES_DIR/$AUTOMATION_PROFILE" ]]; then
  echo "[→] Creating profile: $AUTOMATION_PROFILE"
  mkdir -p "$OBS_PROFILES_DIR/$AUTOMATION_PROFILE"

  # Create minimal profile config
  cat > "$OBS_PROFILES_DIR/$AUTOMATION_PROFILE/basic.ini" << 'EOF'
[General]
Name=Automation

[Output]
FilenameFormatting=%CCYY-%MM-%DD %hh-%mm-%ss

[SimpleOutput]
RecFormat=mkv
RecEncoder=libx264
EOF

  echo "[✓] Profile created: $AUTOMATION_PROFILE"
else
  echo "[✓] Profile already exists: $AUTOMATION_PROFILE"
fi

# Create Automation scene collection with clean media source
if [[ ! -f "$OBS_SCENES_DIR/$AUTOMATION_COLLECTION.json" ]]; then
  echo "[→] Creating scene collection: $AUTOMATION_COLLECTION"

  cat > "$OBS_SCENES_DIR/$AUTOMATION_COLLECTION.json" << 'EOF'
{
  "AuxAudioDevice1": null,
  "AuxAudioDevice2": null,
  "AuxAudioDevice3": null,
  "DesktopAudioDevice": "default",
  "DesktopAudioDevice2": null,
  "DesktopAudioDevice3": null,
  "PPTAEOSDeinterlaceFilter": "Yadif2x",
  "VirtualCamAudioDevice": null,
  "groups": [],
  "name": "Automation",
  "scene-order": [
    {
      "name": "BR-Render"
    }
  ],
  "scenes": [
    {
      "auxAudioDeices": [],
      "customSize": false,
      "desktopAudioDevice1": "default",
      "desktopAudioDevice2": null,
      "desktopAudioDevice3": null,
      "height": 1080,
      "id": "1",
      "lockAspectRatio": true,
      "name": "BR-Render",
      "overflowHidden": true,
      "sceneUuid": "00000000-0000-0000-0000-000000000001",
      "sources": [
        {
          "bloom_threshold_i": 0,
          "color_range": 0,
          "deinterlace_field_order": 0,
          "deinterlace_mode": 0,
          "enabled": true,
          "flags": 0,
          "hotkeys": {
            "OBS_HOTKEY_PUSH_TO_MUTE": [],
            "OBS_HOTKEY_PUSH_TO_TALK": []
          },
          "id": "1",
          "mixers": 0,
          "muted": false,
          "name": "Media Source 2",
          "private_settings": {},
          "settings": {
            "local_file": ""
          },
          "sync": 0,
          "versioned_id": "ffmpeg_source",
          "volume": 1.0
        }
      ],
      "uuid": "00000000-0000-0000-0000-000000000001",
      "virtualCamAudioDevice": null,
      "width": 1920
    }
  ],
  "sources": [],
  "version": 1
}
EOF

  echo "[✓] Scene collection created: $AUTOMATION_COLLECTION"
else
  echo "[✓] Scene collection already exists: $AUTOMATION_COLLECTION"
fi

cat << 'INSTRUCTIONS'

[✓] OBS Automation setup complete!

=== NEXT STEPS ===

1. Close OBS completely if it's running

2. Start OBS with the Automation profile and collection:
   obs --profile "Automation" --collection "Automation"

3. Configure the Automation scene:
   - Go to OBS: Scene 'BR-Render' → Sources → 'Media Source 2'
   - In the Properties panel, set up your Background-Removal filter
   - Save the scene

4. Enable WebSocket Server:
   - Tools → WebSocket Server Settings
   - Enable the server
   - Set port (default 4455)
   - Set password (default: obsstudio)

5. Now you can use BG-Soft:
   python bg_soft_gui.py

=== BENEFITS ===

✓ No missing files dialog on startup
✓ Clean, isolated configuration for automation
✓ Separate from manual OBS usage
✓ Prevents file reference corruption

=== BACKING UP YOUR CURRENT CONFIG ===

Your current scene collection is still intact. If you want to go back:
- In OBS, switch to your original collection
- Use the default profile

To restore a backup, you can access the original scenes in:
  ~/.config/obs-studio/basic/scenes/

INSTRUCTIONS

echo ""
echo "[✓] Setup complete!"
