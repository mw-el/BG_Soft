#!/usr/bin/env python3
"""
Local renderer that performs selfie segmentation without OBS.

Pipeline:
- Probe video (dimensions, rotation)
- Decode frames via ffmpeg pipe
- Run ONNX Selfie Segmentation model
- Composite foreground over black background (with optional blur/expansion)
- Encode via ffmpeg with audio passthrough

Dependencies: onnxruntime, numpy, ffmpeg CLI.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter


@dataclass
class VideoProbe:
    width: int
    height: int
    rotation_deg: int  # clockwise metadata rotation
    fps: float
    duration: float


def probe_video(path: Path) -> VideoProbe:
    """Probe video dimensions/rotation/fps/duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,side_data_list:stream_tags=rotate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(subprocess.check_output(cmd))
    streams = data.get("streams", [])
    width = height = 0
    rotate = 0
    fps = 30.0
    if streams:
        s = streams[0]
        width = int(s.get("width", 0))
        height = int(s.get("height", 0))
        rotate_tag = s.get("tags", {}).get("rotate")
        if rotate_tag:
            rotate = int(float(rotate_tag)) % 360
        for sd in s.get("side_data_list", []) or []:
            if str(sd.get("side_data_type", "")).lower().startswith("display matrix"):
                rot = sd.get("rotation")
                if rot is not None:
                    rotate = int(float(rot)) % 360
        afr = s.get("avg_frame_rate", "30/1")
        try:
            num, den = afr.split("/")
            fps = float(num) / float(den)
        except Exception:
            fps = 30.0
    duration = float(data.get("format", {}).get("duration", 0.0))
    return VideoProbe(width=width, height=height, rotation_deg=rotate, fps=fps, duration=duration)


def effective_dimensions(width: int, height: int, rotation_deg: int) -> Tuple[int, int]:
    """Return dimensions after applying rotation (clockwise)."""
    if abs(rotation_deg) % 180 == 90:
        return height, width
    return width, height


def iter_frames(
    path: Path,
    target_width: int,
    target_height: int,
    log_stream: Optional[object] = None,
    threads: int = 8,
    use_hwaccel: bool = True,
) -> Generator[np.ndarray, None, None]:
    """Decode frames as RGB via ffmpeg pipe, scaled to target size."""
    # CUDA hwaccel only works with video codecs, not images. Disable for image files.
    if use_hwaccel and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}:
        if log_stream:
            try:
                log_stream.write(f"Image file detected ({path.suffix}). Disabling CUDA hwaccel for image decoding.\n")
                log_stream.flush()
            except Exception:
                pass
        use_hwaccel = False

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    cmd += [
        "-i",
        str(path),
        "-threads",
        str(max(1, threads)),
    ]
    # CPU-only decoding with scaling - simple, reliable, guaranteed to work
    cmd += [
        "-vf",
        f"scale={target_width}:{target_height}",
    ]
    cmd += [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    if log_stream:
        try:
            log_stream.write("FFmpeg decode cmd: " + " ".join(cmd) + "\n")
            log_stream.flush()
        except Exception:
            pass
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log_stream or subprocess.DEVNULL)
    frame_size = target_width * target_height * 3
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout not available")
    yielded = 0
    reached_eof = False
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf or len(buf) < frame_size:
                reached_eof = True
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((target_height, target_width, 3))
            yielded += 1
            yield frame
    except GeneratorExit:
        # Consumer stopped early (e.g. max_frames). Terminate ffmpeg quietly.
        reached_eof = False
        raise
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if not reached_eof and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()

        # Only treat as an error if we naturally reached EOF (i.e. processed full clip).
        if reached_eof:
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg decode failed with code {proc.returncode}")
            if yielded == 0:
                raise RuntimeError("ffmpeg decode produced 0 frames")


def load_selfie_model(model_path: Path, log_stream: Optional[object] = None) -> ort.InferenceSession:
    """Load ONNX model with GPU acceleration. Fails fast if GPU unavailable."""
    avail = ort.get_available_providers()

    # Prefer TensorRT (fastest), then CUDA (fast). No CPU fallback.
    preferred_order = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    providers = [p for p in preferred_order if p in avail]

    if log_stream:
        log_stream.write(f"Available providers: {avail}\n")
        log_stream.write(f"Requested providers: {providers}\n")
        log_stream.flush()

    if not providers:
        msg = f"No GPU providers available. Available: {avail}. GPU inference requires TensorRT or CUDA."
        if log_stream:
            log_stream.write(f"ERROR: {msg}\n")
            log_stream.flush()
        raise RuntimeError(msg)

    try:
        sess = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as exc:
        msg = f"Failed to load ONNX model with {providers}: {exc}"
        if log_stream:
            log_stream.write(f"ERROR: {msg}\n")
            log_stream.flush()
        raise RuntimeError(msg) from exc

    if log_stream:
        try:
            actual_providers = sess.get_providers()
            log_stream.write(f"Active providers: {actual_providers}\n")
        except Exception:
            pass
        try:
            inputs = sess.get_inputs()
            outputs = sess.get_outputs()
            if inputs:
                inp0 = inputs[0]
                log_stream.write(f"Model input[0]: {inp0.name} {inp0.shape} {inp0.type}\n")
            for idx, out in enumerate(outputs[:4]):
                log_stream.write(f"Model output[{idx}]: {out.name} {out.shape} {out.type}\n")
        except Exception:
            pass
        log_stream.flush()
    return sess


def _nearest_resize(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbor resize for small tensors without extra deps."""
    h, w = frame.shape[:2]
    ys = np.linspace(0, h - 1, target_h).astype(np.int64)
    xs = np.linspace(0, w - 1, target_w).astype(np.int64)
    return frame[ys[:, None], xs, :]


def run_selfie_mask(session: ort.InferenceSession, frame: np.ndarray, log_stream: Optional[object] = None) -> np.ndarray:
    """Run ONNX segmentation model. Expect frame in RGB uint8."""
    try:
        inp_meta = session.get_inputs()[0]
        shape = inp_meta.shape

        if log_stream:
            try:
                outputs = session.get_outputs()
                log_stream.write(f"Model input shape: {shape}\n")
                for i, out in enumerate(outputs):
                    log_stream.write(f"Model output[{i}] shape: {out.shape}\n")
                log_stream.flush()
            except Exception:
                pass

        # Determine expected H/W and layout
        # Common selfie model: [1,256,256,3] (NHWC)
        if len(shape) == 4 and shape[1] and shape[2] and shape[3] == 3:
            target_h = int(shape[1])
            target_w = int(shape[2])
            resized = np.array(Image.fromarray(frame, mode="RGB").resize((target_w, target_h), Image.BILINEAR))
            inp = resized.astype(np.float32) / 255.0
            inp = inp[None, ...]  # NHWC
        else:
            # Fallback: assume NCHW [1,3,H,W]
            target_h = int(shape[2]) if len(shape) > 2 and shape[2] else frame.shape[0]
            target_w = int(shape[3]) if len(shape) > 3 and shape[3] else frame.shape[1]
            resized = np.array(Image.fromarray(frame, mode="RGB").resize((target_w, target_h), Image.BILINEAR))
            inp = resized.astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[None, ...]

        input_name = inp_meta.name
        output = session.run(None, {input_name: inp})[0]
    except Exception as e:
        msg = f"Model inference failed: {e}"
        if log_stream:
            log_stream.write(f"ERROR: {msg}\n")
            log_stream.flush()
        raise RuntimeError(msg) from e
    raw = output.astype(np.float32)
    if log_stream:
        try:
            log_stream.write(
                f"Model output array shape={tuple(raw.shape)} range=({float(raw.min()):.4f},{float(raw.max()):.4f})\n"
            )
            log_stream.flush()
        except Exception:
            pass

    # Normalize to a single-channel probability mask in [0,1].
    mask_small: np.ndarray
    if raw.ndim == 4:
        raw = raw[0]
    if raw.ndim == 3:
        # Either CHW or HWC.
        if raw.shape[0] in (2, 3) and raw.shape[1] > 8 and raw.shape[2] > 8:
            # CHW
            if raw.shape[0] == 2:
                # Softmax over channel axis (0) if needed.
                if raw.min() < 0.0 or raw.max() > 1.0:
                    e = np.exp(raw - raw.max(axis=0, keepdims=True))
                    probs = e / np.clip(e.sum(axis=0, keepdims=True), 1e-6, None)
                else:
                    probs = raw
                means = probs.reshape(2, -1).mean(axis=1)
                fg_idx = int(np.argmin(means))
                mask_small = probs[fg_idx]
                if log_stream:
                    log_stream.write(
                        f"2ch CHW: means={means.tolist()} -> using channel {fg_idx} as foreground\n"
                    )
                    log_stream.flush()
            else:
                # Take first channel.
                mask_small = raw[0]
        else:
            # HWC
            if raw.shape[-1] == 2:
                if raw.min() < 0.0 or raw.max() > 1.0:
                    e = np.exp(raw - raw.max(axis=-1, keepdims=True))
                    probs = e / np.clip(e.sum(axis=-1, keepdims=True), 1e-6, None)
                else:
                    probs = raw
                means = probs.reshape(-1, 2).mean(axis=0)
                fg_idx = int(np.argmin(means))
                mask_small = probs[..., fg_idx]
                if log_stream:
                    log_stream.write(
                        f"2ch HWC: means={means.tolist()} -> using channel {fg_idx} as foreground\n"
                    )
                    log_stream.flush()
            else:
                mask_small = np.squeeze(raw)
    else:
        mask_small = np.squeeze(raw)

    if mask_small.ndim == 3 and mask_small.shape[-1] == 1:
        mask_small = mask_small[..., 0]

    # Aggressive squeeze for models that output unusual shapes
    while mask_small.ndim > 2:
        if mask_small.shape[0] == 1:
            mask_small = mask_small[0]
        elif mask_small.shape[-1] == 1:
            mask_small = mask_small[..., 0]
        else:
            break

    if mask_small.ndim != 2:
        msg = f"Unsupported mask shape after postprocess: {tuple(mask_small.shape)}. Expected 2D array."
        if log_stream:
            log_stream.write(f"ERROR: {msg}\n")
            log_stream.flush()
        raise RuntimeError(msg)

    # Convert logits / 0-255 masks if necessary.
    if mask_small.min() < 0.0 or mask_small.max() > 1.0:
        if mask_small.min() >= 0.0 and mask_small.max() <= 255.0:
            mask_small = mask_small / 255.0
        else:
            mask_small = 1.0 / (1.0 + np.exp(-mask_small))
    mask_small = np.clip(mask_small.astype(np.float32), 0.0, 1.0)

    # Resize mask back to frame size
    out_h, out_w = frame.shape[:2]
    ys = np.linspace(0, mask_small.shape[0] - 1, out_h).astype(np.int64)
    xs = np.linspace(0, mask_small.shape[1] - 1, out_w).astype(np.int64)
    mask = mask_small[ys[:, None], xs]
    return mask


def apply_mask_filters(
    mask: np.ndarray,
    threshold: float,
    smooth_contour: float,
    mask_expansion: int,
    feather: float = 0.0,
) -> np.ndarray:
    """Apply threshold/expansion/blur to mask."""
    mask = np.squeeze(mask)
    if mask.ndim == 0:
        mask = np.array([mask], dtype=np.float32)

    # Soft threshold: stretch around threshold to avoid harsh edges
    if threshold > 0:
        t_low = max(0.0, threshold - 0.1)
        t_high = min(1.0, threshold + 0.1)
        mask = (mask - t_low) / max(1e-6, (t_high - t_low))
        mask = np.clip(mask, 0.0, 1.0)

    mask_img = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")

    # Expansion (positive = dilate, negative = erode)
    if mask_expansion != 0:
        size = max(3, abs(mask_expansion) * 2 + 1)
        if mask_expansion > 0:
            mask_img = mask_img.filter(ImageFilter.MaxFilter(size=size))
        else:
            mask_img = mask_img.filter(ImageFilter.MinFilter(size=size))

    # Smooth contour (small blur)
    if smooth_contour > 0:
        radius = max(0.1, smooth_contour * 5.0)
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Feather (additional blur)
    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))

    mask_out = np.array(mask_img).astype(np.float32) / 255.0
    return np.clip(mask_out, 0.0, 1.0)


def composite(frame: np.ndarray, mask: np.ndarray, blur_background: int = 0) -> np.ndarray:
    """Apply mask to frame over black or blurred background."""
    alpha = np.squeeze(mask)
    alpha = alpha[..., None]

    if blur_background and blur_background > 0:
        img_bg = Image.fromarray(frame, mode="RGB")
        img_bg = img_bg.filter(ImageFilter.GaussianBlur(radius=float(blur_background)))
        background = np.array(img_bg, dtype=np.float32)
    else:
        background = np.zeros_like(frame, dtype=np.float32)

    fg = frame.astype(np.float32)
    comp = fg * alpha + background * (1.0 - alpha)
    return np.clip(comp, 0, 255).astype(np.uint8)


def encode_video(
    frames: Iterable[np.ndarray],
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    audio_source: Optional[Path],
    rotate_deg: int = 0,
    log_stream: Optional[object] = None,
    use_nvenc: bool = True,
    threads: int = 8,
    use_hwaccel: bool = True,  # currently unused; placeholder for symmetry/debug
    metadata: Optional[dict] = None,
) -> None:
    """
    Encode RGB frames via ffmpeg (video-only), then mux audio if provided.
    This avoids pipe issues and keeps durations consistent.
    Metadata dict is embedded in the output file.
    """
    out_path = out_path.resolve()
    tmp_video = out_path.with_suffix(".video_only.mp4")

    # Step 1: video-only
    vf_filters = []
    if rotate_deg:
        # ffmpeg transpose: 1 = 90° CCW, 2 = 90° CW, 3 = 90° CCW and vertical flip, etc.
        # For arbitrary degrees, use rotate filter; here only multiples of 90.
        if rotate_deg % 360 == 90:
            vf_filters.append("transpose=1")
        elif rotate_deg % 360 == 270:
            vf_filters.append("transpose=2")
        elif rotate_deg % 360 == 180:
            vf_filters.append("rotate=PI")
    vf = ",".join(vf_filters) if vf_filters else None

    cmd_video = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "pipe:0",
    ]
    if vf:
        cmd_video += ["-vf", vf]
    cmd_video += ["-threads", str(max(1, threads))]
    if use_nvenc:
        cmd_video += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-profile:v",
            "high",
            "-rc",
            "vbr",
            "-cq",
            "19",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        cmd_video += [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    # Add metadata if provided
    if metadata:
        for key, value in metadata.items():
            cmd_video.extend(["-metadata", f"{key}={value}"])

    cmd_video += [
        "-y",
        str(tmp_video),
    ]
    if log_stream:
        try:
            log_stream.write("FFmpeg encode cmd: " + " ".join(cmd_video) + "\n")
            log_stream.flush()
        except Exception:
            pass
    err_dst = log_stream or subprocess.DEVNULL
    proc = subprocess.Popen(cmd_video, stdin=subprocess.PIPE, stderr=err_dst)
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin not available")
    written_frames = 0
    try:
        for frame in frames:
            try:
                proc.stdin.write(frame.tobytes())
                written_frames += 1
            except BrokenPipeError:
                if log_stream:
                    log_stream.write("Encode stdin broken pipe; encoder exited early.\n")
                    log_stream.flush()
                break
    finally:
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            msg = f"ffmpeg video encode failed with code {proc.returncode}"
            if log_stream:
                log_stream.write(msg + "\n")
                log_stream.flush()
            raise RuntimeError(msg)
        if written_frames == 0:
            raise RuntimeError("ffmpeg video encode received 0 frames")

    # Step 2: mux audio if available
    if audio_source:
        cmd_mux = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(tmp_video),
            "-i",
            str(audio_source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
        ]
        # Add metadata to mux command as well
        if metadata:
            for key, value in metadata.items():
                cmd_mux.extend(["-metadata", f"{key}={value}"])
        cmd_mux += [
            "-y",
            str(out_path),
        ]
        if log_stream:
            try:
                log_stream.write("FFmpeg mux cmd: " + " ".join(cmd_mux) + "\n")
                log_stream.flush()
            except Exception:
                pass
        mux_proc = subprocess.run(
            cmd_mux,
            stdout=log_stream or subprocess.DEVNULL,
            stderr=log_stream or subprocess.DEVNULL,
        )
        if mux_proc.returncode != 0:
            msg = f"ffmpeg mux failed with code {mux_proc.returncode}"
            if log_stream:
                log_stream.write(msg + "\n")
                log_stream.flush()
            raise RuntimeError(msg)
        try:
            tmp_video.unlink()
        except Exception:
            pass
    else:
        # No audio, move video-only to final path
        tmp_video.replace(out_path)


def render_preview_frame(
    video_path: Path,
    model_path: Path,
    frame_index: int = 0,
    blur_background: int = 0,
    mask_expansion: int = -5,
    feather: float = 0.0,
    smooth_contour: float = 0.05,
    transparency_threshold: float = 0.65,
) -> Optional[np.ndarray]:
    """Render a single frame for preview in the settings dialog.

    Returns RGB frame as uint8 numpy array, or None if rendering fails.
    """
    try:
        # Validate inputs
        if not video_path or not video_path.exists():
            return None
        if not model_path or not model_path.exists():
            return None

        # Probe video to get dimensions
        try:
            probe = probe_video(video_path)
        except Exception:
            return None

        if probe.width == 0 or probe.height == 0:
            return None

        # Load model with timeout/error handling
        try:
            session = load_selfie_model(model_path)
        except Exception:
            return None

        # Extract just the frame we need
        try:
            for idx, frame in enumerate(
                iter_frames(
                    video_path,
                    probe.width,
                    probe.height,
                    log_stream=None,
                    threads=2,
                    use_hwaccel=False,
                )
            ):
                if idx == frame_index:
                    # Run inference and processing with error handling
                    try:
                        mask_raw = run_selfie_mask(session, frame)
                        mask = apply_mask_filters(
                            mask_raw,
                            threshold=transparency_threshold,
                            smooth_contour=smooth_contour,
                            mask_expansion=mask_expansion,
                            feather=feather,
                        )
                        comp = composite(frame, mask, blur_background=blur_background)
                        return comp
                    except Exception:
                        return None
                elif idx > frame_index:
                    break
        except Exception:
            return None

        return None
    except Exception:
        return None


def _build_metadata_dict(
    model_path: Path,
    blur_background: int,
    mask_expansion: int,
    feather: float,
    smooth_contour: float,
    transparency_threshold: float,
    extra_rotation_ccw: int = 0,
) -> dict:
    """Build metadata dict from render settings for embedding in video file."""
    import datetime

    metadata = {
        "title": "BG-Soft Rendered",
        "comment": "Rendered with BG-Soft local ONNX renderer",
        "creation_time": datetime.datetime.now().isoformat(),
        "bgsoft_model": model_path.name,
        "bgsoft_blur_background": str(blur_background),
        "bgsoft_mask_expansion": str(mask_expansion),
        "bgsoft_feather": f"{feather:.4f}",
        "bgsoft_smooth_contour": f"{smooth_contour:.4f}",
        "bgsoft_transparency_threshold": f"{transparency_threshold:.4f}",
    }
    if extra_rotation_ccw:
        metadata["bgsoft_extra_rotation_ccw"] = str(extra_rotation_ccw)

    return metadata


def render_local(
    input_video: Path,
    output_video: Path,
    model_path: Path,
    force_width: Optional[int] = None,
    force_height: Optional[int] = None,
    settings_path: Optional[Path] = None,
    max_frames: Optional[int] = None,
    extra_rotation_ccw: int = 0,
    threads: int = 8,
    background_settings: Optional[object] = None,
) -> None:
    """Render a video locally using selfie segmentation and black background."""
    log_path = input_video.parent / f"{input_video.stem}_bgsoft_no_obs.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("=== BG-Soft ohne OBS Run ===\n")
        log_file.write(f"Input: {input_video}\nOutput: {output_video}\nModel: {model_path}\n")
        log_file.write(f"Settings file: {settings_path}\nExtra rotation CCW: {extra_rotation_ccw}\n")
        log_file.write("Using NVENC: yes, Threads: 8, HWAccel decode/scale: yes\n")
        log_file.flush()

        # Load filter settings
        settings = {}
        if settings_path and settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
                settings = settings.get("background_removal", settings)
            except Exception:
                settings = {}

        def _from_bg(key: str, default_val: float | int) -> float | int:
            if background_settings is None:
                return default_val
            if isinstance(background_settings, dict):
                return background_settings.get(key, default_val)  # type: ignore[return-value]
            return getattr(background_settings, key, default_val)

        threshold = float(_from_bg("threshold", settings.get("threshold", settings.get("transparency_threshold", 0.65))))
        smooth_contour = float(_from_bg("smooth_contour", settings.get("smooth_contour", 0.05)))
        mask_expansion = int(_from_bg("mask_expansion", settings.get("mask_expansion", -5)))
        feather = float(_from_bg("feather", settings.get("feather", 0.0)))
        blur_background = int(_from_bg("blur_background", settings.get("blur_background", 0)))

        probe = probe_video(input_video)
        # Base dimensions after metadata rotation only
        base_w, base_h = effective_dimensions(probe.width, probe.height, probe.rotation_deg)
        target_w = force_width or base_w
        target_h = force_height or base_h
        rotate_out = (probe.rotation_deg + extra_rotation_ccw) % 360

        log_file.write(
            f"Probe: {probe.width}x{probe.height}, rotation={probe.rotation_deg}, fps={probe.fps}, duration={probe.duration}s\n"
        )
        log_file.write(
            f"Target: {target_w}x{target_h}, output rotation={rotate_out} (includes manual rotation)\n"
        )
        log_file.write(
            "BG params: "
            f"threshold={threshold}, smooth_contour={smooth_contour}, mask_expansion={mask_expansion}, "
            f"feather={feather}, blur_background={blur_background}\n"
        )
        log_file.flush()

        session = load_selfie_model(model_path, log_stream=log_file)

        debug_frames = int(os.environ.get("BGSOFT_DEBUG_FRAMES", "2"))
        dump_debug = os.environ.get("BGSOFT_DUMP_DEBUG", "").strip() in {"1", "true", "yes", "on"}
        frame_counter = 0

        def frame_iter():
            nonlocal frame_counter
            for idx, frame in enumerate(
                iter_frames(
                input_video,
                target_w,
                target_h,
                log_stream=log_file,
                threads=threads,
                use_hwaccel=False,
            )
            ):
                mask_raw = run_selfie_mask(session, frame, log_stream=log_file if idx < debug_frames else None)
                mask = apply_mask_filters(
                    mask_raw,
                    threshold=threshold,
                    smooth_contour=smooth_contour,
                    mask_expansion=mask_expansion,
                    feather=feather,
                )
                comp = composite(frame, mask, blur_background=blur_background)
                if idx < debug_frames:
                    try:
                        import numpy as _np

                        md = float(_np.abs(frame.astype(_np.int16) - comp.astype(_np.int16)).mean())
                        log_file.write(
                            f"Debug frame {idx}: mask mean={float(_np.mean(mask)):.4f}, "
                            f"alpha<0.5={float(_np.mean(mask < 0.5)):.4f}, mean_abs_delta={md:.4f}\n"
                        )
                        log_file.flush()
                        if dump_debug:
                            from PIL import Image as _Image
                            import time as _time

                            stamp = int(_time.time())
                            base = input_video.parent / f"{input_video.stem}_debug_{stamp}_{idx}"
                            _Image.fromarray(frame, mode="RGB").save(base.with_suffix(".in.png"))
                            _Image.fromarray((mask * 255.0).clip(0, 255).astype("uint8"), mode="L").save(
                                base.with_suffix(".mask.png")
                            )
                            _Image.fromarray(comp, mode="RGB").save(base.with_suffix(".out.png"))
                    except Exception:
                        pass
                yield comp
                frame_counter += 1

                # Write progress every 50 frames so monitoring can track progress
                if frame_counter % 50 == 0:
                    log_file.write(f"Frames processed: {frame_counter}\n")
                    log_file.flush()

                if max_frames and frame_counter >= max_frames:
                    break

            # Write final frame count
            log_file.write(f"Frames processed: {frame_counter}\n")
            log_file.flush()

        try:
            # Build metadata from render settings
            metadata = _build_metadata_dict(
                model_path=model_path,
                blur_background=blur_background,
                mask_expansion=mask_expansion,
                feather=feather,
                smooth_contour=smooth_contour,
                transparency_threshold=threshold,
                extra_rotation_ccw=extra_rotation_ccw,
            )

            # Use a bounded queue to buffer frames and apply backpressure.
            # This prevents the pipe buffer from overflowing when the encoder
            # is slower than the frame producer.
            frame_queue = queue.Queue(maxsize=30)
            producer_exception = []

            def frame_producer():
                """Producer thread: put frames into queue, blocking if full."""
                try:
                    for frame in frame_iter():
                        frame_queue.put(frame, block=True)
                except Exception as e:
                    producer_exception.append(e)
                finally:
                    frame_queue.put(None)  # Sentinel to signal EOF

            def buffered_frames():
                """Consumer generator: take frames from queue."""
                while True:
                    frame = frame_queue.get()
                    if frame is None:
                        if producer_exception:
                            raise producer_exception[0]
                        break
                    yield frame

            # Start producer thread
            producer_thread = threading.Thread(target=frame_producer, daemon=True)
            producer_thread.start()

            encode_video(
                frames=buffered_frames(),
                out_path=output_video,
                width=target_w,
                height=target_h,
                fps=probe.fps or 30.0,
                audio_source=input_video,
                rotate_deg=rotate_out,
                log_stream=log_file,
                threads=threads,
                use_hwaccel=True,
                metadata=metadata,
            )

            # Ensure producer thread finished
            producer_thread.join(timeout=5)
        except Exception as exc:  # noqa: BLE001
            log_file.write(f"Render failed: {exc}\n")
            log_file.flush()
            raise
        else:
            if frame_counter == 0:
                raise RuntimeError("Render produced 0 frames (ffmpeg decode/filter failure)")
            log_file.write(f"Completed successfully: {output_video}\n")
            log_file.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Local selfie segmentation renderer (no OBS).")
    parser.add_argument("input", type=Path, help="Input video")
    parser.add_argument("output", type=Path, help="Output video")
    parser.add_argument("--model", type=Path, default=Path("models/selfie_segmentation.onnx"), help="ONNX model path")
    parser.add_argument("--width", type=int, default=None, help="Force output width")
    parser.add_argument("--height", type=int, default=None, help="Force output height")
    parser.add_argument("--settings", type=Path, default=Path("settings.json"), help="Settings file (for thresholds/blur)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames (for quick tests)")
    parser.add_argument("--rotate-ccw", type=int, default=0, help="Extra rotation CCW in degrees (multiples of 90)")
    args = parser.parse_args()

    render_local(
        args.input,
        args.output,
        args.model,
        force_width=args.width,
        force_height=args.height,
        settings_path=args.settings,
        max_frames=args.max_frames,
        extra_rotation_ccw=args.rotate_ccw,
    )
