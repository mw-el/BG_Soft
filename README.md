# BG-Soft: OBS Background Removal Automation

Automatisierung BG-Soften mit OBS - A Python tool that automates video background removal using OBS Studio with the Background Removal plugin.

## Features

- **Automated Recording**: Automatically starts/stops recording when video playback ends
- **Batch Processing**: Process multiple videos in a single session
- **GUI Interface**: User-friendly PyQt5 interface for easy configuration
- **Filter Control**: Adjust background removal and sharpening parameters
- **Smart Cleanup**: Automatically cleans up media sources between renders
- **Desktop Integration**: Launch via desktop icon for quick access

## Quick Start

### 1. Prerequisites

- Linux system with OBS Studio (Flatpak or native)
- Python 3.11+
- Conda/Miniconda installed

### 2. Installation

```bash
cd /path/to/BG_Soft
./install.sh
```

The installer will:
- Create a `BG-Soft` conda environment
- Install all Python dependencies (obsws-python, PyQt5)
- Set up the OBS Automation profile and scene collection
- Configure the desktop launcher

### 3. Initial OBS Setup (One-time)

After installation, start OBS with the Automation profile:

```bash
./start_obs.sh
```

Then:
1. Go to **Tools → WebSocket Server Settings**
2. Enable the WebSocket Server
3. Set port to `4455` (default)
4. Set password to `obsstudio` (default, changeable in GUI)

### 4. Configure Filters (One-time)

In OBS:
1. Select the `BR-Render` scene
2. Select the `bg-soft` media source
3. Add or configure your background removal filters:
   - **Background Removal** - Main filter (required)
   - **Sharpen** - Optional, for detail enhancement

## Usage

### Desktop Launcher

```bash
# Click the "BG-Soft" icon in your applications menu
# Or run:
./launch_bgsoft.sh
```

The GUI will:
- Automatically start OBS if needed
- Launch the user interface
- Ready to process videos

### GUI Interface

1. **Connection Settings**
   - Host: `localhost` (default)
   - Port: `4455` (default)
   - Password: `obsstudio` (default)
   - Scene: `BR-Render` (default)
   - Media Source: `bg-soft` (default)

2. **Background Removal Settings**
   - Segmentation model selection (SINet recommended)
   - Threshold, contour, and silhouette smoothing
   - Mask expansion and temporal smoothing
   - Focal blur configuration

3. **Sharpening Settings**
   - Adjustable sharpness factor (default: 10.0)

4. **Batch Processing**
   - Add multiple video files
   - Set common filter parameters
   - Process all files with one click
   - Monitor progress in real-time

### CLI Usage

```bash
conda activate BG-Soft
python render_with_obs.py /path/to/video.mp4
```

Optional flags:
```bash
--host localhost          # OBS WebSocket host
--port 4455              # OBS WebSocket port
--password obsstudio     # OBS WebSocket password
--scene BR-Render        # Scene name in OBS
--input bg-soft          # Media source name
--poll 0.5               # Status polling interval (seconds)
```

## Output Files

Processed videos are saved with the pattern:
```
<original_name>_soft_<YYYYMMDD-HHMMSS>.<original_extension>
```

Example: `video.mp4` → `video_soft_20231213-103015.mkv`

Output location: Same directory as input file

## Configuration Files

### Environment
- `environment.yml` - Conda environment specification
- `requirements.txt` - Python dependencies

### OBS Setup
- `setup_obs_automation.sh` - Creates clean Automation profile/scene
- `bgsoft.desktop.template` - Desktop launcher template
- `cleanup_obs_config.sh` - Utility to clean old OBS references

### Application
- `bg_soft_gui.py` - PyQt5 GUI application
- `obs_controller.py` - OBS WebSocket API wrapper
- `render_with_obs.py` - CLI rendering script
- `launch_bgsoft.sh` - Desktop launcher script
- `start_obs.sh` - OBS startup helper
- `dummy.mp4` - Placeholder video for media source cleanup

## Default Filter Settings

### Background Removal
- **Model**: SINet (better quality than Selfie Segmentation)
- **Threshold**: 0.65 (enabled)
- **Contour Filter**: 1.0
- **Smooth Silhouette**: 0.05
- **Mask Expansion**: -5 (tighter mask)
- **Temporal Smooth Factor**: 0.5
- **Focal Blur**: Enabled
  - Focus Point: 0.05
  - Focus Depth: 0.16

### Sharpening
- **Sharpness**: 10.0 (strong, can be adjusted in GUI)

These defaults are optimized for high-quality background removal with proper edge definition and visual clarity.

## Troubleshooting

### OBS won't start
- Ensure OBS is closed completely before running `start_obs.sh`
- Check if port 4455 is already in use: `lsof -i :4455`

### Media file not loading
- Verify the media source name matches exactly (usually `bg-soft`)
- Check WebSocket connection is enabled in OBS
- Look at GUI output for detailed error messages

### No GPU support (for CUDA acceleration)
- Currently using Flatpak OBS, which is CPU-only
- See **GPU Acceleration** section below for native build instructions

## GPU Acceleration (Advanced)

The default Flatpak OBS installation uses CPU-only inference. For GPU acceleration with NVIDIA GPUs:

**See GPU_ACCELERATION.md for detailed setup instructions.**

Quick summary:
1. Install CUDA 12 + cuDNN 9 runtime
2. Build obs-backgroundremoval from source with CUDA support
3. Install as system package (not Flatpak)
4. Configure OBS to use native installation

## System Requirements

- **Minimum**: CPU inference (any Linux system with OBS)
- **Recommended**: NVIDIA GPU (RTX 30-series or newer)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: ~2GB for conda environment

## File Structure

```
BG_Soft/
├── README.md                      # This file
├── GPU_ACCELERATION.md            # GPU setup guide
├── environment.yml                # Conda environment
├── requirements.txt               # Python dependencies
│
├── bg_soft_gui.py                 # Main GUI application
├── obs_controller.py              # OBS WebSocket API
├── render_with_obs.py             # CLI script
│
├── launch_bgsoft.sh               # Desktop launcher
├── start_obs.sh                   # OBS startup helper
├── install.sh                     # Installation script
│
├── setup_obs_automation.sh        # OBS profile setup
├── cleanup_obs_config.sh          # Config cleanup utility
│
├── bgsoft.desktop.template        # Desktop file template
├── bgsoft.png                     # Application icon
├── dummy.mp4                      # Placeholder video
│
└── .gitignore                     # Git ignore rules
```

## License

This project integrates with OBS Studio and the obs-backgroundremoval plugin.

## Development

### Making Changes

1. Update code in the repo
2. Test with: `python bg_soft_gui.py`
3. Commit changes: `git commit -m "Description"`
4. Push: `git push origin main`

### Building from Source

See GPU_ACCELERATION.md for native build instructions.

## Support

For issues with:
- **BG-Soft**: Check the GitHub issues
- **OBS Studio**: https://obsproject.com/
- **Background Removal Plugin**: https://github.com/royshil/obs-backgroundremoval
