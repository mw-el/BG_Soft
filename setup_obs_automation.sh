#!/usr/bin/env bash
# Setup dedicated OBS Automation scene collection for BG-Soft
# This creates a clean collection without file references that trigger the missing files dialog

set -euo pipefail

OBS_CONFIG_DIR="$HOME/.config/obs-studio"
OBS_PROFILES_DIR="$OBS_CONFIG_DIR/basic/profiles"
OBS_SCENES_DIR="$OBS_CONFIG_DIR/basic/scenes"

# Validate OBS is installed (check for Flatpak or native binary)
if ! command -v flatpak >/dev/null 2>&1 || ! flatpak list --app 2>/dev/null | grep -q "com.obsproject.Studio"; then
  if ! command -v obs >/dev/null 2>&1; then
    echo "[!] OBS Studio is not installed (neither Flatpak nor native binary found)"
    exit 1
  fi
fi

# Create necessary directories
mkdir -p "$OBS_PROFILES_DIR"
mkdir -p "$OBS_SCENES_DIR"

AUTOMATION_PROFILE="Automation"
AUTOMATION_COLLECTION="Automation"

echo "[→] Setting up OBS Automation configuration..."

# Enable WebSocket server for automation
WEBSOCKET_CONFIG="$HOME/.config/obs-studio/plugin_config/obs-websocket"
mkdir -p "$WEBSOCKET_CONFIG"
echo "[→] Configuring WebSocket server..."
cat > "$WEBSOCKET_CONFIG/config.json" << 'WEBSOCKET_EOF'
{
  "alerts_enabled": false,
  "auth_required": true,
  "first_load": false,
  "server_enabled": true,
  "server_password": "obsstudio",
  "server_port": 4455
}
WEBSOCKET_EOF
echo "[✓] WebSocket server configured"

# Create Automation profile if it doesn't exist
if [[ ! -d "$OBS_PROFILES_DIR/$AUTOMATION_PROFILE" ]]; then
  echo "[→] Creating profile: $AUTOMATION_PROFILE"
  mkdir -p "$OBS_PROFILES_DIR/$AUTOMATION_PROFILE"

  # Create minimal profile config that defaults to Automation scene collection
  cat > "$OBS_PROFILES_DIR/$AUTOMATION_PROFILE/basic.ini" << 'EOF'
[General]
Name=Automation
SceneCollection=Automation

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
    "current_scene": "BR-Render",
    "current_program_scene": "BR-Render",
    "scene_order": [
        {
            "name": "BR-Render"
        }
    ],
    "name": "Automation",
    "sources": [
        {
            "prev_ver": 536870914,
            "name": "BR-Render",
            "uuid": "00000000-0000-0000-0000-000000000001",
            "id": "scene",
            "versioned_id": "scene",
            "settings": {
                "id_counter": 1,
                "custom_size": false,
                "items": [
                    {
                        "name": "Media Source 2",
                        "source_uuid": "00000000-0000-0000-0000-000000000002"
                    }
                ]
            },
            "mixers": 0,
            "sync": 0,
            "flags": 0,
            "volume": 1.0,
            "balance": 0.5,
            "enabled": true,
            "muted": false,
            "push-to-mute": false,
            "push-to-mute-delay": 0,
            "push-to-talk": false,
            "push-to-talk-delay": 0,
            "hotkeys": {
                "OBSBasic.SelectScene": []
            },
            "deinterlace_mode": 0,
            "deinterlace_field_order": 0,
            "monitoring_type": 0,
            "canvas_uuid": "6c69626f-6273-4c00-9d88-c5136d61696e",
            "private_settings": {}
        },
        {
            "prev_ver": 536870914,
            "name": "Media Source 2",
            "uuid": "00000000-0000-0000-0000-000000000002",
            "id": "ffmpeg_source",
            "versioned_id": "ffmpeg_source",
            "settings": {
                "local_file": ""
            },
            "mixers": 0,
            "sync": 0,
            "flags": 0,
            "volume": 1.0,
            "balance": 0.5,
            "enabled": true,
            "muted": false,
            "push-to-mute": false,
            "push-to-mute-delay": 0,
            "push-to-talk": false,
            "push-to-talk-delay": 0,
            "hotkeys": {
                "OBS_HOTKEY_PUSH_TO_MUTE": [],
                "OBS_HOTKEY_PUSH_TO_TALK": []
            },
            "deinterlace_mode": 0,
            "deinterlace_field_order": 0,
            "monitoring_type": 0,
            "canvas_uuid": "6c69626f-6273-4c00-9d88-c5136d61696e",
            "private_settings": {}
        }
    ],
    "groups": [],
    "quick_transitions": [
        {
            "name": "Cut",
            "duration": 300,
            "hotkeys": [],
            "id": 1,
            "fade_to_black": false
        },
        {
            "name": "Fade",
            "duration": 300,
            "hotkeys": [],
            "id": 2,
            "fade_to_black": false
        },
        {
            "name": "Fade",
            "duration": 300,
            "hotkeys": [],
            "id": 3,
            "fade_to_black": true
        }
    ],
    "transitions": [],
    "saved_projectors": [],
    "canvases": [],
    "current_transition": "Fade",
    "transition_duration": 300,
    "preview_locked": false,
    "scaling_enabled": false,
    "scaling_level": -17,
    "scaling_off_x": 0.0,
    "scaling_off_y": 0.0,
    "virtual-camera": {
        "type2": 3
    },
    "modules": {
        "scripts-tool": [],
        "output-timer": {
            "streamTimerHours": 0,
            "streamTimerMinutes": 0,
            "streamTimerSeconds": 30,
            "recordTimerHours": 0,
            "recordTimerMinutes": 0,
            "recordTimerSeconds": 30,
            "autoStartStreamTimer": false,
            "autoStartRecordTimer": false,
            "pauseRecordTimer": true
        },
        "auto-scene-switcher": {
            "interval": 300,
            "non_matching_scene": "",
            "switch_if_not_matching": false,
            "active": false,
            "switches": []
        }
    },
    "version": 2
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

4. WebSocket Server:
   - WebSocket Server is now pre-configured and enabled
   - Default port: 4455
   - Default password: obsstudio

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
