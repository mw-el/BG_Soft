#!/usr/bin/env python3
"""Shared OBS automation utilities for BG-Soft CLI + GUI."""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import obsws_python as obs
from obsws_python.error import OBSSDKRequestError

DEFAULT_MODEL_SELFIE = "models/selfie_segmentation.onnx"


def load_settings(settings_file: str = "settings.json") -> Dict[str, Any]:
    """Load settings from JSON file. Returns dict with defaults if file not found."""
    file_path = pathlib.Path(settings_file)
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load settings file: {e}")
    return {}


class RenderError(RuntimeError):
    """Raised when OBS reports a problem while processing a clip."""


@dataclass
class ConnectionSettings:
    host: str = "localhost"
    port: int = 4455
    password: str = "obsstudio"
    scene_name: str = "BR-Render"
    input_name: str = "bg-soft"
    background_filter_name: str = "Background Removal"
    sharpen_filter_name: str = "Sharpen"


def _get_background_removal_defaults() -> Dict[str, Any]:
    """Get background removal defaults from settings file or hardcoded values."""
    settings = load_settings()
    bg_settings = settings.get("background_removal", {})
    return {
        "advanced": bg_settings.get("advanced", True),
        "enable_threshold": bg_settings.get("enable_threshold", True),
        "threshold": bg_settings.get("threshold", 0.65),
        "contour_filter": bg_settings.get("contour_filter", 1.0),
        "smooth_contour": bg_settings.get("smooth_contour", 0.05),
        "mask_expansion": bg_settings.get("mask_expansion", -5),
        "use_gpu": bg_settings.get("use_gpu", "cpu"),
        "mask_every_x_frames": bg_settings.get("mask_every_x_frames", 1),
        "num_threads": bg_settings.get("num_threads", 8),
        "model_select": bg_settings.get("model_select", "models/SINet_Softmax_simple.onnx"),
        "temporal_smooth_factor": bg_settings.get("temporal_smooth_factor", 0.5),
        "enable_image_similarity": bg_settings.get("enable_image_similarity", True),
        "image_similarity_threshold": bg_settings.get("image_similarity_threshold", 100.0),
        "blur_background": bg_settings.get("blur_background", 3),
        "enable_focal_blur": bg_settings.get("enable_focal_blur", True),
        "blur_focus_point": bg_settings.get("blur_focus_point", 0.05),
        "blur_focus_depth": bg_settings.get("blur_focus_depth", 0.16),
        "feather": bg_settings.get("feather", 0.0),
    }


@dataclass
class BackgroundRemovalSettings:
    advanced: bool = True
    enable_threshold: bool = True
    threshold: float = 0.65
    contour_filter: float = 1.0
    smooth_contour: float = 0.05
    mask_expansion: int = -5
    use_gpu: str = "cpu"
    mask_every_x_frames: int = 1
    num_threads: int = 8
    model_select: str = "models/SINet_Softmax_simple.onnx"
    temporal_smooth_factor: float = 0.5
    enable_image_similarity: bool = True
    image_similarity_threshold: float = 100.0
    blur_background: int = 3
    enable_focal_blur: bool = True
    blur_focus_point: float = 0.05
    blur_focus_depth: float = 0.16
    feather: float = 0.0

    def to_filter_payload(self) -> Dict[str, object]:
        return {
            "advanced": self.advanced,
            "enable_threshold": self.enable_threshold,
            "threshold": float(self.threshold),
            "contour_filter": float(self.contour_filter),
            "smooth_contour": float(self.smooth_contour),
            "mask_expansion": int(self.mask_expansion),
            "useGPU": self.use_gpu,
            "mask_every_x_frames": int(self.mask_every_x_frames),
            "numThreads": int(self.num_threads),
            "model_select": self.model_select,
            "temporal_smooth_factor": float(self.temporal_smooth_factor),
            "enable_image_similarity": self.enable_image_similarity,
            "image_similarity_threshold": float(self.image_similarity_threshold),
            "blur_background": int(self.blur_background),
            "enable_focal_blur": self.enable_focal_blur,
            "blur_focus_point": float(self.blur_focus_point),
            "blur_focus_depth": float(self.blur_focus_depth),
            "feather": float(self.feather),
        }


@dataclass
class SharpenSettings:
    sharpness: float = 10.0

    def to_filter_payload(self) -> Dict[str, float]:
        return {"sharpness": float(self.sharpness)}


class ObsRenderer:
    """High-level helper that talks to OBS over obs-websocket."""

    def __init__(self, conn: ConnectionSettings, poll_interval: float = 0.5):
        self.conn = conn
        self.poll_interval = poll_interval
        self._client: Optional[obs.ReqClient] = None
        self._validated = False

    @property
    def client(self) -> obs.ReqClient:
        if self._client is None:
            self._client = self._connect_with_retry()
        return self._client

    def _connect_with_retry(self) -> obs.ReqClient:
        """Connect to OBS, with automatic startup attempt if needed."""
        max_retries = 20
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                client = obs.ReqClient(
                    host=self.conn.host,
                    port=self.conn.port,
                    password=self.conn.password,
                    timeout=5,
                )
                print(f"[✓] Connected to OBS at {self.conn.host}:{self.conn.port}")
                return client
            except Exception as e:
                if attempt == 0:
                    print(f"[!] OBS connection failed on first attempt: {e}")
                    print("[!] Attempting to auto-start OBS...")
                    self._try_start_obs()
                    print(f"[!] Waiting {max_retries * retry_delay} seconds for OBS to start...")

                if attempt < max_retries - 1:
                    print(f"[!] Retry attempt {attempt + 1}/{max_retries - 1}... (waiting {retry_delay}s)")
                    time.sleep(retry_delay)
                else:
                    raise RenderError(
                        f"Could not connect to OBS at {self.conn.host}:{self.conn.port} after {max_retries} attempts. "
                        f"Please ensure:\n"
                        f"  1. OBS Studio is installed and running\n"
                        f"  2. WebSocket server is enabled (Tools → WebSocket Server Settings)\n"
                        f"  3. Port {self.conn.port} is correct\n"
                        f"  4. Password is correct\n"
                        f"Error: {e}"
                    ) from e

    @staticmethod
    def _try_start_obs() -> None:
        """Attempt to start OBS automatically based on OS."""
        try:
            if sys.platform.startswith("linux"):
                # Try common OBS installation locations on Linux
                try:
                    subprocess.Popen(["obs", "--profile", "Automation"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print("[✓] OBS startup command sent (Linux)")
                except FileNotFoundError:
                    print("[!] obs command not found in PATH")
                    print("[!] Please start OBS manually")

            elif sys.platform.startswith("darwin"):
                # macOS - use open command
                subprocess.Popen(["open", "-a", "OBS"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("[✓] OBS startup command sent (macOS)")

            elif sys.platform.startswith("win"):
                # Windows - try to find OBS installation
                try:
                    subprocess.Popen(["obs.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print("[✓] OBS startup command sent (Windows)")
                except FileNotFoundError:
                    subprocess.Popen("obs", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print("[✓] OBS startup command sent (Windows)")
            else:
                print(f"[!] Unsupported platform: {sys.platform}")

        except Exception as e:
            print(f"[!] Could not auto-start OBS: {e}")
            print("[!] Please start OBS manually and enable WebSocket server")

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.disconnect()
            self._client = None

    # ---------------- OBS operations ---------------- #
    def ensure_media_finished(self, input_name: str) -> None:
        """Block until OBS finishes playing the media source."""
        last_state: Optional[str] = None
        while True:
            status = self.client.get_media_input_status(input_name)
            state = getattr(status, "media_state", None)
            if state and state != last_state:
                print(f"Media state: {state}")
                last_state = state
            if state == "OBS_MEDIA_STATE_ENDED":
                return
            if state == "OBS_MEDIA_STATE_ERROR":
                raise RenderError("OBS reported OBS_MEDIA_STATE_ERROR while playing the clip.")
            time.sleep(self.poll_interval)

    def finalize_record(self) -> pathlib.Path:
        """Ensure recording stopped and return the finished file path."""
        # Retry logic for OBS readiness
        max_retries = 5
        for attempt in range(max_retries):
            try:
                status = self.client.get_record_status()
                output_active = getattr(status, "output_active", False)
                output_path = getattr(status, "output_path", None)
                break
            except OBSSDKRequestError as e:
                if "not ready" in str(e).lower() and attempt < max_retries - 1:
                    print(f"[!] OBS not ready, retrying in 0.5s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(0.5)
                else:
                    raise

        if output_active:
            try:
                response = self.client.stop_record()
            except OBSSDKRequestError as exc:
                raise RenderError(f"OBS konnte die Aufnahme nicht stoppen: {exc}") from exc
            output_path = getattr(response, "output_path", output_path)

        if not output_path:
            raise RenderError("OBS meldete keinen Speicherpfad für die Aufnahme.")

        recorded = pathlib.Path(output_path)
        deadline = time.time() + 30
        while not recorded.exists():
            if time.time() > deadline:
                raise RenderError(f"Recorded file was not created: {recorded}")
            time.sleep(self.poll_interval)
        return recorded

    def cleanup_media_source(self) -> None:
        """Clear the media source and load dummy file to prevent missing file errors on next run."""
        try:
            dummy_file = pathlib.Path(__file__).parent / "dummy.mp4"
            if not dummy_file.exists():
                print(f"[!] Dummy file not found: {dummy_file}")
                return

            print("[→] Cleaning up media source...")

            # Stop playback
            try:
                self.client.trigger_media_input_action(
                    self.conn.input_name,
                    "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_PAUSE",
                )
                time.sleep(0.5)
            except Exception:
                pass  # Media might already be stopped

            # Load dummy file (always available, prevents "file not found" dialog)
            self.client.set_input_settings(
                self.conn.input_name,
                {
                    "local_file": str(dummy_file.resolve()),
                    "is_local_file": True,
                },
                overlay=True,
            )
            print("[✓] Media source reset to dummy file")
        except Exception as e:
            print(f"[!] Could not cleanup media source: {e}")

    @staticmethod
    def move_output(recorded: pathlib.Path, original: pathlib.Path) -> pathlib.Path:
        """Move the recorded file next to the original clip with the desired suffix."""
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        target_name = f"{original.stem}_soft_{timestamp}{recorded.suffix}"
        target = original.with_name(target_name)
        shutil.move(str(recorded), target)
        return target

    def apply_filter_settings(
        self,
        background_settings: Optional[BackgroundRemovalSettings] = None,
        sharpen_settings: Optional[SharpenSettings] = None,
    ) -> None:
        """Push filter settings to OBS, creating filters if they don't exist."""
        if background_settings is not None:
            self._ensure_and_update_filter(
                source_name=self.conn.input_name,
                filter_name=self.conn.background_filter_name,
                filter_kind="background_removal",
                settings=background_settings.to_filter_payload(),
            )

        if sharpen_settings is not None:
            self._ensure_and_update_filter(
                source_name=self.conn.input_name,
                filter_name=self.conn.sharpen_filter_name,
                filter_kind="sharpness_filter_v2",
                settings=sharpen_settings.to_filter_payload(),
            )

    def _ensure_and_update_filter(
        self,
        source_name: str,
        filter_name: str,
        filter_kind: str,
        settings: Dict[str, object],
    ) -> None:
        """Ensure a filter exists (create if needed) and update its settings."""
        try:
            # Try to get existing filter
            existing = self.client.get_source_filter(
                source_name=source_name,
                filter_name=filter_name,
            )
            print(f"[✓] Filter '{filter_name}' exists, updating settings...")
            # Filter exists, update settings
            self.client.set_source_filter_settings(
                source_name=source_name,
                filter_name=filter_name,
                settings=settings,
                overlay=True,
            )
            # Enable the filter
            self.client.set_source_filter_enabled(
                source_name=source_name,
                filter_name=filter_name,
                enabled=True,
            )
            print(f"[✓] Updated filter '{filter_name}' settings and enabled it")
        except OBSSDKRequestError:
            # Filter doesn't exist, create it
            print(f"[!] Filter '{filter_name}' not found, creating...")
            try:
                self.client.create_source_filter(
                    source_name=source_name,
                    filter_name=filter_name,
                    filter_kind=filter_kind,
                    filter_settings=settings,
                )
                # Enable the newly created filter
                self.client.set_source_filter_enabled(
                    source_name=source_name,
                    filter_name=filter_name,
                    enabled=True,
                )
                print(f"[✓] Created and enabled filter '{filter_name}'")
            except OBSSDKRequestError as exc:
                raise RenderError(
                    f"Could not create filter '{filter_name}' of type '{filter_kind}': {exc}"
                ) from exc

    @staticmethod
    def _get_video_duration(source: pathlib.Path) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1:nokey=1",
                    str(source),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            print(f"[!] Could not get video duration: {e}")
        return 0.0

    def _wait_for_playback_started(self) -> None:
        """Wait until media is actually playing and rendering frames."""
        max_wait = 10  # Maximum 10 seconds to wait for playback to start
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                status = self.client.get_media_input_status(self.conn.input_name)
                media_state = getattr(status, "media_state", None)

                if media_state == "OBS_MEDIA_STATE_PLAYING":
                    print("[✓] Media is playing, recording started capture")
                    time.sleep(0.5)  # Small extra buffer to ensure first frame is captured
                    return
                elif media_state == "OBS_MEDIA_STATE_ERROR":
                    raise RenderError("Media playback error detected")
            except OBSSDKRequestError:
                # OBS might not be ready yet, that's okay
                pass
            except Exception as e:
                print(f"[!] Error waiting for playback: {e}")

            time.sleep(0.2)  # Check frequently

        print("[!] Warning: Playback didn't reach 'playing' state within timeout, continuing anyway")

    def _monitor_and_auto_stop(self, duration_sec: float) -> None:
        """Monitor playback and auto-stop recording when video finishes or duration elapses."""
        if duration_sec <= 0:
            # If we couldn't determine duration, fall back to monitoring media state
            self.ensure_media_finished(self.conn.input_name)
            return

        # Wait for video to finish based on duration + buffer
        deadline = time.time() + duration_sec + 2  # 2 second buffer for safety
        print(f"[→] Will auto-stop recording in ~{duration_sec:.1f} seconds")

        consecutive_errors = 0
        while time.time() < deadline:
            try:
                status = self.client.get_media_input_status(self.conn.input_name)
                media_state = getattr(status, "media_state", None)
                consecutive_errors = 0  # Reset error counter on success

                if media_state == "OBS_MEDIA_STATE_ENDED":
                    print("Media playback finished")
                    break
                if media_state == "OBS_MEDIA_STATE_ERROR":
                    print("Media error detected")
                    break
            except OBSSDKRequestError as e:
                # "OBS not ready" is expected during early playback startup
                if "not ready" in str(e).lower():
                    consecutive_errors += 1
                    if consecutive_errors % 5 == 1:  # Log every 5th error
                        print(f"[!] OBS warming up... ({consecutive_errors})")
                    if consecutive_errors > 30:  # Fail after 15 seconds of errors
                        raise
                else:
                    raise
            except Exception as e:
                print(f"[!] Error checking media state: {e}")
                consecutive_errors += 1
                if consecutive_errors > 30:
                    raise

            time.sleep(self.poll_interval)

    def _clear_media_input_file(self) -> None:
        """Clear the file path on the media input to prevent OBS trying to load missing files."""
        try:
            print("[→] Clearing media input to prevent OBS loading stale files...")
            # This should prevent the "missing files" dialog from appearing
            self.client.set_input_settings(
                self.conn.input_name,
                {"local_file": ""},
                overlay=True,
            )
            time.sleep(1)  # Let OBS process the empty file setting
            print("[✓] Media input cleared")
        except Exception as e:
            print(f"[!] Could not clear media input: {e}")

    def validate_obs_setup(self) -> None:
        """Ensure the configured scene and input exist. Filters are created automatically."""
        # Clear stale media file references to prevent "missing files" dialog
        self._clear_media_input_file()

        scenes_response = self.client.get_scene_list()
        scene_names = [scene.get("sceneName") for scene in getattr(scenes_response, "scenes", [])]
        if self.conn.scene_name not in scene_names:
            print(f"[!] Scene '{self.conn.scene_name}' not found. Creating it...")
            try:
                self.client.create_scene(self.conn.scene_name)
                print(f"[✓] Scene '{self.conn.scene_name}' created")
            except Exception as e:
                raise RenderError(
                    f"Could not create scene '{self.conn.scene_name}': {e}"
                ) from e

        inputs_response = self.client.get_input_list()
        input_names = [item.get("inputName") for item in getattr(inputs_response, "inputs", [])]
        if self.conn.input_name not in input_names:
            print(f"[!] Media input '{self.conn.input_name}' not found. Creating it in scene '{self.conn.scene_name}'...")
            try:
                self.client.create_input(
                    sceneName=self.conn.scene_name,
                    inputName=self.conn.input_name,
                    inputKind="ffmpeg_source",
                    inputSettings={"local_file": ""},
                    sceneItemEnabled=True,
                )
                print(f"[✓] Media input '{self.conn.input_name}' created")
            except Exception as e:
                raise RenderError(
                    f"Could not create media input '{self.conn.input_name}': {e}"
                ) from e

        # Note: We no longer require filters to exist beforehand.
        # They will be created automatically when apply_filter_settings() is called.
        filters_response = self.client.get_source_filter_list(self.conn.input_name)
        filters = [f.get("filterName") for f in getattr(filters_response, "filters", [])]
        if filters:
            print(f"[✓] Existing filters on '{self.conn.input_name}': {', '.join(filters)}")
        else:
            print(f"[!] No filters found on '{self.conn.input_name}'. They will be created automatically.")

    def render_file(
        self,
        source: pathlib.Path,
        background_settings: Optional[BackgroundRemovalSettings] = None,
        sharpen_settings: Optional[SharpenSettings] = None,
    ) -> pathlib.Path:
        """Render a single media file through OBS."""
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input clip not found: {source}")

        if not self._validated:
            self.validate_obs_setup()
            self._validated = True

        # Get video duration upfront
        duration_sec = self._get_video_duration(source)
        print(f"[→] Video duration: {duration_sec:.2f} seconds")

        record_status = self.client.get_record_status()
        if getattr(record_status, "output_active", False):
            print("OBS is currently recording. Stopping active recording...")
            try:
                self.client.stop_record()
                time.sleep(0.5)  # Give OBS time to finalize the recording
                print("Active recording stopped successfully.")
            except OBSSDKRequestError as exc:
                raise RenderError(f"Could not stop active OBS recording: {exc}") from exc

        try:
            self.client.set_current_program_scene(self.conn.scene_name)
        except OBSSDKRequestError as exc:
            raise RenderError(
                f"OBS konnte die Szene '{self.conn.scene_name}' nicht aktivieren: {exc}"
            ) from exc

        # Clean OBS state: stop any currently playing media and clear old file reference
        print("[→] Clearing OBS media state...")
        try:
            # First, pause any playing media
            self.client.trigger_media_input_action(
                self.conn.input_name,
                "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_PAUSE",
            )
            print("[→] Stopped any previously playing media...")
            time.sleep(0.5)
        except Exception as e:
            print(f"[!] Could not pause media (might be okay): {e}")

        # Clear the old file path completely to ensure clean slate
        try:
            self.client.set_input_settings(
                self.conn.input_name,
                {"local_file": ""},  # Clear the file path first
                overlay=True,
            )
            print("[→] Cleared old media file reference...")
            time.sleep(1)  # Let OBS fully release the old file
        except Exception as e:
            print(f"[!] Could not clear old file (might be okay): {e}")

        # Now load the new file with proper settings
        print(f"[→] Setting file path for input '{self.conn.input_name}'")
        print(f"    File: {source}")

        self.client.set_input_settings(
            self.conn.input_name,
            {
                "local_file": str(source),
                "is_local_file": True,  # Critical: Tell OBS this is a local file
                "restart_on_activate": False,  # We'll control playback manually
            },
            overlay=True,
        )
        print("[→] Settings sent to OBS, waiting for decoder initialization...")
        time.sleep(5)  # Let media source fully load and decode first frame

        # Verify the file loaded without errors
        try:
            status = self.client.get_media_input_status(self.conn.input_name)
            media_state = getattr(status, "media_state", None)
            print(f"[→] Media input status check:")
            print(f"    Input: {self.conn.input_name}")
            print(f"    State: {media_state}")

            if media_state == "OBS_MEDIA_STATE_ERROR":
                raise RenderError(f"Failed to load media file: {source}")
            print(f"[✓] Media file ready")
        except OBSSDKRequestError as e:
            print(f"[!] Could not verify media state: {e}")
        except Exception as e:
            print(f"[!] Error during media verification: {e}")

        if background_settings or sharpen_settings:
            self.apply_filter_settings(background_settings, sharpen_settings)
            print("[→] Filters applied, waiting for filter initialization...")
            time.sleep(5)  # Let filters initialize (especially background removal)

        # Ensure scene item is visible
        try:
            scene_items = self.client.get_scene_item_list(self.conn.scene_name)
            items = getattr(scene_items, "scene_items", [])
            for item in items:
                if item.get("sourceName") == self.conn.input_name:
                    item_id = item.get("sceneItemId")
                    self.client.set_scene_item_enabled(
                        self.conn.scene_name,
                        item_id,
                        True
                    )
                    print(f"[✓] Scene item '{self.conn.input_name}' enabled")
                    break
        except Exception as e:
            print(f"[!] Could not enable scene item: {e}")

        # Start recording BEFORE triggering playback to capture from the beginning
        self.client.start_record()
        print("Recording started...")
        time.sleep(1)  # Minimal delay for encoder initialization

        # Now trigger playback immediately
        self.client.trigger_media_input_action(
            self.conn.input_name,
            "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART",
        )
        print("Media playback triggered...")

        # Wait for media to actually be playing and rendering
        # This ensures frames are being captured to the recording
        self._wait_for_playback_started()

        # Monitor playback and auto-stop recording when done
        try:
            self._monitor_and_auto_stop(duration_sec)
        finally:
            # Always attempt to stop recording so OBS finalizes the file even if playback failed.
            recorded = self.finalize_record()
            # Clean up media source to prevent missing file errors next time
            self.cleanup_media_source()

        target = self.move_output(recorded, source)
        return target


def render_batch(
    files: Iterable[pathlib.Path],
    renderer: ObsRenderer,
    background_settings: Optional[BackgroundRemovalSettings] = None,
    sharpen_settings: Optional[SharpenSettings] = None,
) -> List[pathlib.Path]:
    """Render multiple files sequentially."""
    outputs: List[pathlib.Path] = []
    try:
        for video in files:
            outputs.append(renderer.render_file(video, background_settings, sharpen_settings))
    finally:
        renderer.disconnect()
    return outputs


def open_with_system_handler(path: pathlib.Path) -> None:
    """Open a file in the default system handler/video player."""
    absolute = str(path.expanduser().resolve())
    if sys.platform.startswith("darwin"):
        subprocess.Popen(["open", absolute], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif os.name == "nt":
        os.startfile(absolute)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", absolute], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
