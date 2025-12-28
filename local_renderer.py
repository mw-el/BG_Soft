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

import datetime
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

# Phase 3: GPU acceleration support
try:
    import cupy as cp
    from cupyx.scipy import ndimage as gpu_ndimage
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    gpu_ndimage = None


NVENC_LOCK = threading.Lock()
RENDER_LOCK = threading.Lock()


def _configure_cupy_memory(log_stream: Optional[object]) -> None:
    if not GPU_AVAILABLE or cp is None:
        if log_stream:
            log_stream.write("[GPU] CuPy not available; skipping memory pool config.\n")
            log_stream.flush()
        return
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        margin_bytes = 1024**3  # 1 GB safety margin
        limit_bytes = max(0, free_bytes - margin_bytes)
        cp.get_default_memory_pool().set_limit(size=limit_bytes)
        if log_stream:
            gib = 1024**3
            log_stream.write(
                f"[GPU] CuPy mem pool limit set to {limit_bytes / gib:.2f} GB "
                f"(free={free_bytes / gib:.2f} GB, total={total_bytes / gib:.2f} GB, "
                f"margin={margin_bytes / gib:.2f} GB)\n"
            )
            if limit_bytes == 0:
                log_stream.write("[GPU] CuPy mem pool limit is 0; GPU ops may fail if memory is scarce.\n")
            log_stream.flush()
    except Exception as exc:
        if log_stream:
            log_stream.write(f"[GPU] CuPy mem pool limit not set: {exc}\n")
            log_stream.flush()


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


_FFMPEG_HWACCELS: Optional[set[str]] = None
_FFMPEG_HWACCELS_LOCK = threading.Lock()


def _ffmpeg_hwaccels(log_stream: Optional[object] = None) -> set[str]:
    global _FFMPEG_HWACCELS
    if _FFMPEG_HWACCELS is not None:
        return _FFMPEG_HWACCELS
    with _FFMPEG_HWACCELS_LOCK:
        if _FFMPEG_HWACCELS is not None:
            return _FFMPEG_HWACCELS
        try:
            out = subprocess.check_output(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            if log_stream:
                try:
                    log_stream.write(f"FFmpeg hwaccel probe failed: {exc}\n")
                    log_stream.flush()
                except Exception:
                    pass
            _FFMPEG_HWACCELS = set()
            return _FFMPEG_HWACCELS
        accels: set[str] = set()
        for line in out.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.lower().startswith("hardware acceleration"):
                continue
            accels.add(line.split()[0])
        _FFMPEG_HWACCELS = accels
        return _FFMPEG_HWACCELS


def _iter_frames_from_cmd(
    cmd: list[str],
    target_width: int,
    target_height: int,
    log_stream: Optional[object],
    stage: str,
) -> Generator[np.ndarray, None, None]:
    if log_stream:
        try:
            log_stream.write(f"FFmpeg {stage} cmd: " + " ".join(cmd) + "\n")
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


def _iter_frames_cpu(
    path: Path,
    target_width: int,
    target_height: int,
    log_stream: Optional[object],
    threads: int,
    start_time: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    cmd += [
        "-i",
        str(path),
    ]
    if start_time is not None and start_time > 0:
        cmd += ["-ss", f"{start_time:.3f}"]
    cmd += [
        "-threads",
        str(max(1, threads)),
    ]
    if max_frames is not None and max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
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
    yield from _iter_frames_from_cmd(cmd, target_width, target_height, log_stream, "decode_cpu")


def _iter_frames_cuvid(
    path: Path,
    target_width: int,
    target_height: int,
    log_stream: Optional[object],
    threads: int,
    start_time: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    cmd += [
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(path),
    ]
    if start_time is not None and start_time > 0:
        cmd += ["-ss", f"{start_time:.3f}"]
    cmd += [
        "-threads",
        str(max(1, threads)),
    ]
    if max_frames is not None and max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [
        "-vf",
        f"scale_cuda={target_width}:{target_height},hwdownload,format=nv12,format=rgb24",
    ]
    cmd += [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    yield from _iter_frames_from_cmd(cmd, target_width, target_height, log_stream, "decode_cuvid")


def iter_frames(
    path: Path,
    target_width: int,
    target_height: int,
    log_stream: Optional[object] = None,
    threads: int = 8,
    use_hwaccel: bool = True,
    start_time: Optional[float] = None,
    max_frames: Optional[int] = None,
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

    if use_hwaccel:
        accels = _ffmpeg_hwaccels(log_stream)
        if "cuda" not in accels:
            if log_stream:
                try:
                    log_stream.write("FFmpeg build has no CUDA hwaccel. Using CPU decode.\n")
                    log_stream.flush()
                except Exception:
                    pass
            use_hwaccel = False

    if use_hwaccel:
        if log_stream:
            try:
                log_stream.write("Attempting CUVID GPU decode...\n")
                log_stream.flush()
            except Exception:
                pass
        try:
            yield from _iter_frames_cuvid(
                path,
                target_width,
                target_height,
                log_stream,
                threads,
                start_time=start_time,
                max_frames=max_frames,
            )
            return
        except Exception as exc:
            if log_stream:
                try:
                    log_stream.write(f"CUVID failed: {exc}. Falling back to CPU.\n")
                    log_stream.flush()
                except Exception:
                    pass

    yield from _iter_frames_cpu(
        path,
        target_width,
        target_height,
        log_stream,
        threads,
        start_time=start_time,
        max_frames=max_frames,
    )


def load_selfie_model(model_path: Path, log_stream: Optional[object] = None) -> ort.InferenceSession:
    """Load ONNX model with GPU acceleration (Phase 1). GPU required, no CPU fallback."""
    avail = ort.get_available_providers()

    # Phase 1: GPU Acceleration Support (GPU REQUIRED)
    # Use CUDA for GPU acceleration. No CPU fallback.
    # (TensorRT requires libnvinfer library which may not be installed)
    preferred_order = ["CUDAExecutionProvider"]
    providers = [p for p in preferred_order if p in avail]

    if log_stream:
        log_stream.write(f"Available providers: {avail}\n")
        log_stream.write(f"Requested GPU providers: {providers}\n")
        log_stream.flush()

    if not providers:
        msg = f"GPU acceleration required but not available. Need TensorRT or CUDA. Available: {avail}. Set LD_LIBRARY_PATH to include CUDA/cuDNN libraries."
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
            # Phase 1: GPU-only mode - all providers should be GPU
            if actual_providers:
                log_stream.write(f"✓ GPU acceleration enabled (PHASE 1)\n")
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


# Phase 3: GPU-accelerated mask filtering and compositing
def apply_mask_filters_gpu(
    mask: np.ndarray,
    threshold: float,
    smooth_contour: float,
    mask_expansion: int,
    feather: float = 0.0,
) -> np.ndarray:
    """GPU-accelerated mask filtering using CuPy (Phase 3)."""
    if not GPU_AVAILABLE or cp is None:
        return None  # Fall back to CPU

    try:
        # Transfer to GPU
        mask_gpu = cp.asarray(mask, dtype=cp.float32)
        mask_gpu = cp.squeeze(mask_gpu)

        # Soft threshold on GPU
        if threshold > 0:
            t_low = max(0.0, threshold - 0.1)
            t_high = min(1.0, threshold + 0.1)
            mask_gpu = (mask_gpu - t_low) / max(1e-6, (t_high - t_low))
            mask_gpu = cp.clip(mask_gpu, 0.0, 1.0)

        # Mask expansion using GPU filters (dilate/erode via Gaussian approximation)
        if mask_expansion != 0:
            sigma = abs(mask_expansion) * 0.5
            if sigma > 0:
                if mask_expansion > 0:
                    # Dilate: blur then threshold
                    mask_gpu = gpu_ndimage.gaussian_filter(mask_gpu, sigma=sigma)
                else:
                    # Erode: inverse blur
                    mask_gpu = 1.0 - gpu_ndimage.gaussian_filter(1.0 - mask_gpu, sigma=sigma)

        # Smooth contour (GPU Gaussian blur)
        if smooth_contour > 0:
            sigma = max(0.1, smooth_contour * 0.5)
            mask_gpu = gpu_ndimage.gaussian_filter(mask_gpu, sigma=sigma)

        # Feather (additional GPU blur)
        if feather > 0:
            mask_gpu = gpu_ndimage.gaussian_filter(mask_gpu, sigma=feather / 2.0)

        mask_gpu = cp.clip(mask_gpu, 0.0, 1.0)

        # Transfer back to CPU
        return cp.asnumpy(mask_gpu).astype(np.float32)

    except Exception as e:
        # GPU operation failed, fall back to CPU
        return None


def composite_gpu(frame: np.ndarray, mask: np.ndarray, blur_background: int = 0) -> Optional[np.ndarray]:
    """GPU-accelerated compositing using CuPy (Phase 3)."""
    if not GPU_AVAILABLE or cp is None:
        return None  # Fall back to CPU

    try:
        # Transfer to GPU
        frame_gpu = cp.asarray(frame, dtype=cp.float32)
        mask_gpu = cp.asarray(mask, dtype=cp.float32)

        # Prepare alpha
        alpha_gpu = cp.squeeze(mask_gpu)
        alpha_gpu = alpha_gpu[..., None]

        # Process background
        if blur_background and blur_background > 0:
            # CuPy gaussian_filter doesn't support 'axes' parameter like SciPy.
            # Use sigma tuple (H, W, 0) to blur only spatial dimensions, not color channels.
            sigma_val = blur_background / 2.0
            bg_gpu = gpu_ndimage.gaussian_filter(frame_gpu, sigma=(sigma_val, sigma_val, 0))
        else:
            bg_gpu = cp.zeros_like(frame_gpu)

        # Composite on GPU
        comp_gpu = frame_gpu * alpha_gpu + bg_gpu * (1.0 - alpha_gpu)
        comp_gpu = cp.clip(comp_gpu, 0, 255).astype(cp.uint8)

        # Transfer back to CPU
        return cp.asnumpy(comp_gpu)

    except Exception as e:
        # GPU operation failed, fall back to CPU
        return None


def apply_mask_filters(
    mask: np.ndarray,
    threshold: float,
    smooth_contour: float,
    mask_expansion: int,
    feather: float = 0.0,
) -> np.ndarray:
    """Apply threshold/expansion/blur to mask (Phase 3: GPU-accelerated)."""
    # Try GPU first (Phase 3)
    gpu_result = apply_mask_filters_gpu(mask, threshold, smooth_contour, mask_expansion, feather)
    if gpu_result is not None:
        return gpu_result

    # Fall back to CPU
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
    """Apply mask to frame over black or blurred background (Phase 3: GPU-accelerated)."""
    # Try GPU first (Phase 3)
    gpu_result = composite_gpu(frame, mask, blur_background)
    if gpu_result is not None:
        return gpu_result

    # Fall back to CPU
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


def _ffmpeg_loglevel() -> str:
    level = os.environ.get("BGSOFT_FFMPEG_LOGLEVEL", "warning").strip()
    return level if level else "warning"


def _build_ffmpeg_report_path(base_path: Path, stage: str) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    safe_stage = stage.replace(" ", "_")
    return base_path.with_name(f"{base_path.stem}_ffmpeg_{safe_stage}_{stamp}.log")


def _ffmpeg_env(report_path: Path) -> dict:
    env = os.environ.copy()
    env["FFREPORT"] = f"file={report_path}:level=32"
    return env


def _test_nvenc_quick() -> bool:
    """Quick NVENC availability check (1 test frame, 1-2 seconds)."""
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1280x720",
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p2",
            "-t",
            "0.033",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        proc.stdin.write(test_frame.tobytes())
        proc.stdin.close()
        ret = proc.wait(timeout=5)
        return ret == 0
    except Exception:
        return False


def encode_video_only(
    frames: Iterable[np.ndarray],
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    rotate_deg: int = 0,
    log_stream: Optional[object] = None,
    use_nvenc: bool = True,
    threads: int = 8,
    metadata: Optional[dict] = None,
) -> int:
    """Encode RGB frames via ffmpeg (video-only). Returns written frame count."""
    out_path = out_path.resolve()

    loglevel = _ffmpeg_loglevel()
    report_base = out_path.resolve()
    encode_stage = "encode_nvenc" if use_nvenc else "encode_x264"
    encode_report = _build_ffmpeg_report_path(report_base, encode_stage)
    encode_report.parent.mkdir(parents=True, exist_ok=True)
    encode_env = _ffmpeg_env(encode_report)
    if log_stream:
        log_stream.write(f"FFmpeg report ({encode_stage}): {encode_report}\n")
        log_stream.flush()

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
        "-report",
        "-loglevel",
        loglevel,
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
            "p2",
            "-tune",
            "hq",
            "-profile:v",
            "high",
            "-rc",
            "vbr",
            "-cq",
            "19",
            "-rc-lookahead",
            "20",
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
    if metadata:
        for key, value in metadata.items():
            cmd_video.extend(["-metadata", f"{key}={value}"])

    cmd_video += [
        "-y",
        str(out_path),
    ]
    if log_stream:
        try:
            log_stream.write("FFmpeg encode cmd: " + " ".join(cmd_video) + "\n")
            log_stream.flush()
        except Exception:
            pass
    proc = subprocess.Popen(
        cmd_video,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=encode_env
    )
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin not available")

    written_frames = 0
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
            written_frames += 1
            if written_frames % 100 == 0:
                if log_stream:
                    log_stream.write(f"Frames written: {written_frames}\n")
                    log_stream.flush()
    except BrokenPipeError:
        if log_stream:
            log_stream.write(f"Broken pipe after {written_frames} frames (FFmpeg process exited)\n")
            log_stream.flush()
        raise RuntimeError(f"FFmpeg encoder exited unexpectedly after {written_frames} frames")
    except (OSError, IOError) as e:
        if log_stream:
            log_stream.write(f"IO error after {written_frames} frames: {e}\n")
            log_stream.flush()
        raise
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if log_stream:
                log_stream.write(f"FFmpeg timeout after {written_frames} frames - killing\n")
                log_stream.flush()
            proc.kill()
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed with code {proc.returncode} ({written_frames} frames)")
        if written_frames == 0:
            raise RuntimeError("ffmpeg encode produced 0 frames")

    return written_frames


def _mux_audio(
    video_path: Path,
    audio_source: Optional[Path],
    out_path: Path,
    log_stream: Optional[object],
    metadata: Optional[dict],
) -> None:
    if audio_source is None:
        video_path.replace(out_path)
        return

    loglevel = _ffmpeg_loglevel()
    report_base = (audio_source or out_path).resolve()
    mux_report = _build_ffmpeg_report_path(report_base, "mux")
    mux_report.parent.mkdir(parents=True, exist_ok=True)
    mux_env = _ffmpeg_env(mux_report)
    if log_stream:
        log_stream.write(f"FFmpeg report (mux): {mux_report}\n")
        log_stream.flush()

    cmd_mux = [
        "ffmpeg",
        "-hide_banner",
        "-report",
        "-loglevel",
        loglevel,
        "-i",
        str(video_path),
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
    if metadata:
        for key, value in metadata.items():
            cmd_mux.extend(["-metadata", f"{key}={value}"])
    cmd_mux += ["-y", str(out_path)]

    try:
        mux_proc = subprocess.run(
            cmd_mux,
            stdout=log_stream or subprocess.DEVNULL,
            stderr=log_stream or subprocess.DEVNULL,
            env=mux_env,
            timeout=300,
        )
        if mux_proc.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed with code {mux_proc.returncode}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg mux timeout (>5 min)")

    video_path.unlink()


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

    encode_video_only(
        frames=frames,
        out_path=tmp_video,
        width=width,
        height=height,
        fps=fps,
        rotate_deg=rotate_deg,
        log_stream=log_stream,
        use_nvenc=use_nvenc,
        threads=threads,
        metadata=metadata,
    )

    _mux_audio(
        video_path=tmp_video,
        audio_source=audio_source,
        out_path=out_path,
        log_stream=log_stream,
        metadata=metadata,
    )


def _chunk_dir_for(output_video: Path) -> Path:
    return output_video.parent / f".{output_video.stem}_chunks"


def _chunk_state_path(output_video: Path) -> Path:
    return output_video.with_suffix(".render_state.json")


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _calc_initial_chunk_frames(width: int, height: int, fps: float) -> int:
    base_frames = 500  # empirically stable around 1080p@30fps
    base_pixels = 1920 * 1080
    fps = fps or 30.0
    scale = (base_pixels / max(1, width * height)) * (30.0 / fps)
    return max(1, int(base_frames * scale))


def _load_render_state(state_path: Path) -> Optional[dict]:
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_render_state(state_path: Path, state: dict, log_stream: Optional[object]) -> None:
    try:
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        if log_stream:
            log_stream.write(f"[STATE] Failed to write state file: {exc}\n")
            log_stream.flush()


def _concat_chunks(chunk_files: list[Path], out_path: Path, log_stream: Optional[object]) -> None:
    if not chunk_files:
        raise RuntimeError("No chunk files available for concat")
    list_path = out_path.with_suffix(".concat.txt")
    with list_path.open("w", encoding="utf-8") as handle:
        for chunk in chunk_files:
            safe_path = str(chunk).replace("'", "'\\''")
            handle.write(f"file '{safe_path}'\n")

    loglevel = _ffmpeg_loglevel()
    concat_report = _build_ffmpeg_report_path(out_path, "concat")
    concat_report.parent.mkdir(parents=True, exist_ok=True)
    concat_env = _ffmpeg_env(concat_report)
    if log_stream:
        log_stream.write(f"FFmpeg report (concat): {concat_report}\n")
        log_stream.flush()

    cmd_concat = [
        "ffmpeg",
        "-hide_banner",
        "-report",
        "-loglevel",
        loglevel,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        "-y",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd_concat,
        stdout=log_stream or subprocess.DEVNULL,
        stderr=log_stream or subprocess.DEVNULL,
        env=concat_env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed with code {proc.returncode}")
    try:
        list_path.unlink()
    except Exception:
        pass


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
    use_nvenc: bool = True,
    background_settings: Optional[object] = None,
) -> None:
    """Render a video locally using selfie segmentation and black background."""
    log_path = input_video.parent / f"{input_video.stem}_bgsoft_no_obs.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("=== BG-Soft ohne OBS Run ===\n")
        log_file.write(f"Input: {input_video}\nOutput: {output_video}\nModel: {model_path}\n")
        log_file.write(f"Settings file: {settings_path}\nExtra rotation CCW: {extra_rotation_ccw}\n")
        log_file.write(
            f"Using NVENC: {'yes' if use_nvenc else 'no'}, Threads: {threads}, HWAccel decode: yes (CUVID)\n"
        )
        chunking_enabled = os.environ.get("BGSOFT_CHUNKING", "1").strip().lower() not in {"0", "false", "no"}
        log_file.write(f"Chunked processing: {'enabled' if chunking_enabled else 'disabled'}\n")
        log_file.flush()

        render_lock_acquired = False
        try:
            log_file.write("Waiting for global render slot...\n")
            log_file.flush()
            RENDER_LOCK.acquire()
            render_lock_acquired = True
            log_file.write("Global render slot acquired.\n")
            log_file.flush()

            _render_local_impl(
                input_video=input_video,
                output_video=output_video,
                model_path=model_path,
                force_width=force_width,
                force_height=force_height,
                settings_path=settings_path,
                max_frames=max_frames,
                extra_rotation_ccw=extra_rotation_ccw,
                threads=threads,
                use_nvenc=use_nvenc,
                background_settings=background_settings,
                log_file=log_file,
            )
        finally:
            if render_lock_acquired:
                RENDER_LOCK.release()
                log_file.write("Global render slot released.\n")
                log_file.flush()


def _render_local_impl(
    *,
    input_video: Path,
    output_video: Path,
    model_path: Path,
    force_width: Optional[int],
    force_height: Optional[int],
    settings_path: Optional[Path],
    max_frames: Optional[int],
    extra_rotation_ccw: int,
    threads: int,
    use_nvenc: bool,
    background_settings: Optional[object],
    log_file,
) -> None:
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
    fps = probe.fps or 30.0
    total_frames = None
    if probe.duration and fps:
        total_frames = int(round(probe.duration * fps))

    log_file.write(
        f"Probe: {probe.width}x{probe.height}, rotation={probe.rotation_deg}, fps={probe.fps}, duration={probe.duration}s\n"
    )
    log_file.write(f"Target: {target_w}x{target_h}, output rotation={rotate_out} (includes manual rotation)\n")
    if total_frames is not None:
        log_file.write(f"Estimated total frames: {total_frames}\n")
    log_file.write(
        "BG params: "
        f"threshold={threshold}, smooth_contour={smooth_contour}, mask_expansion={mask_expansion}, "
        f"feather={feather}, blur_background={blur_background}\n"
    )
    log_file.flush()

    _configure_cupy_memory(log_file)

    log_file.write("[STAGE] Loading ONNX model...\n")
    log_file.flush()
    session = load_selfie_model(model_path, log_stream=log_file)

    debug_frames = int(os.environ.get("BGSOFT_DEBUG_FRAMES", "2"))
    dump_debug = os.environ.get("BGSOFT_DUMP_DEBUG", "").strip() in {"1", "true", "yes", "on"}
    chunking_enabled = os.environ.get("BGSOFT_CHUNKING", "1").strip().lower() not in {"0", "false", "no"}
    chunk_reduce_pct = max(1, min(90, _safe_int_env("BGSOFT_CHUNK_REDUCE_PCT", 25)))
    chunk_min_frames = max(1, _safe_int_env("BGSOFT_CHUNK_MIN_FRAMES", 150))
    chunk_max_frames = max(chunk_min_frames, _safe_int_env("BGSOFT_CHUNK_MAX_FRAMES", 1500))
    chunk_override = _safe_int_env("BGSOFT_CHUNK_FRAMES", 0)
    initial_chunk_frames = _calc_initial_chunk_frames(target_w, target_h, fps)
    if chunk_override > 0:
        initial_chunk_frames = chunk_override
    initial_chunk_frames = max(chunk_min_frames, min(chunk_max_frames, initial_chunk_frames))
    total_limit = total_frames
    if max_frames is not None:
        total_limit = max_frames if total_limit is None else min(total_limit, max_frames)
    frame_counter = 0

    def _frame_iter_for_chunk(start_frame: int, max_chunk_frames: int) -> Generator[np.ndarray, None, None]:
        nonlocal frame_counter
        limit_text = f"{max_chunk_frames} frames" if max_chunk_frames > 0 else "full stream"
        log_file.write(
            f"[CHUNK] Decode start at frame {start_frame} for up to {limit_text}\n"
        )
        log_file.flush()
        start_time = start_frame / fps if fps else 0.0
        for idx, frame in enumerate(
            iter_frames(
                input_video,
                target_w,
                target_h,
                log_stream=log_file,
                threads=threads,
                use_hwaccel=True,
                start_time=start_time,
                max_frames=max_chunk_frames,
            )
        ):
            abs_idx = start_frame + idx
            mask_raw = run_selfie_mask(session, frame, log_stream=log_file if abs_idx < debug_frames else None)
            mask = apply_mask_filters(
                mask_raw,
                threshold=threshold,
                smooth_contour=smooth_contour,
                mask_expansion=mask_expansion,
                feather=feather,
            )
            comp = composite(frame, mask, blur_background=blur_background)
            if abs_idx < debug_frames:
                try:
                    import numpy as _np

                    md = float(_np.abs(frame.astype(_np.int16) - comp.astype(_np.int16)).mean())
                    log_file.write(
                        f"Debug frame {abs_idx}: mask mean={float(_np.mean(mask)):.4f}, "
                        f"alpha<0.5={float(_np.mean(mask < 0.5)):.4f}, mean_abs_delta={md:.4f}\n"
                    )
                    log_file.flush()
                    if dump_debug:
                        from PIL import Image as _Image
                        import time as _time

                        stamp = int(_time.time())
                        base = input_video.parent / f"{input_video.stem}_debug_{stamp}_{abs_idx}"
                        _Image.fromarray(frame, mode="RGB").save(base.with_suffix(".in.png"))
                        _Image.fromarray((mask * 255.0).clip(0, 255).astype("uint8"), mode="L").save(
                            base.with_suffix(".mask.png")
                        )
                        _Image.fromarray(comp, mode="RGB").save(base.with_suffix(".out.png"))
                except Exception:
                    pass
            yield comp
            frame_counter = abs_idx + 1

            # CRITICAL: Explicitly delete GPU tensors to prevent memory leak
            del frame, mask_raw, mask, comp

            if frame_counter % 50 == 0:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                log_file.write(f"[{ts}] Progress: {frame_counter} frames processed\n")
                log_file.flush()

            # Periodic garbage collection to free GPU memory
            if frame_counter % 100 == 0:
                import gc
                gc.collect()
                if GPU_AVAILABLE and cp is not None:
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                    except Exception:
                        pass

    def run_pipeline_monolithic(use_hw_encoder: bool) -> None:
        nonlocal frame_counter
        frame_counter = 0

        def frame_iter():
            log_file.write("[STAGE] Starting decode and inference pipeline...\n")
            log_file.flush()
            yield from _frame_iter_for_chunk(0, max_frames or (total_frames or 0) or 0)
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            log_file.write(f"[{ts}] Pipeline complete: {frame_counter} frames processed\n")
            log_file.flush()

        metadata = _build_metadata_dict(
            model_path=model_path,
            blur_background=blur_background,
            mask_expansion=mask_expansion,
            feather=feather,
            smooth_contour=smooth_contour,
            transparency_threshold=threshold,
            extra_rotation_ccw=extra_rotation_ccw,
        )

        frames_iter = frame_iter()
        log_file.write("[STAGE] Starting video encoding...\n")
        log_file.flush()
        try:
            encode_video(
                frames=frames_iter,
                out_path=output_video,
                width=target_w,
                height=target_h,
                fps=fps,
                audio_source=input_video,
                rotate_deg=rotate_out,
                log_stream=log_file,
                threads=threads,
                use_nvenc=use_hw_encoder,
                use_hwaccel=True,
                metadata=metadata,
            )
        finally:
            close_iter = getattr(frames_iter, "close", None)
            if callable(close_iter):
                close_iter()

    def run_pipeline_chunked(use_hw_encoder: bool) -> None:
        nonlocal frame_counter
        frame_counter = 0

        metadata = _build_metadata_dict(
            model_path=model_path,
            blur_background=blur_background,
            mask_expansion=mask_expansion,
            feather=feather,
            smooth_contour=smooth_contour,
            transparency_threshold=threshold,
            extra_rotation_ccw=extra_rotation_ccw,
        )

        chunk_dir = _chunk_dir_for(output_video)
        state_path = _chunk_state_path(output_video)
        signature = {
            "input": str(input_video.resolve()),
            "input_size": input_video.stat().st_size if input_video.exists() else 0,
            "input_mtime": input_video.stat().st_mtime if input_video.exists() else 0,
            "target_w": target_w,
            "target_h": target_h,
            "fps": fps,
            "rotate_out": rotate_out,
        }

        start_frame = 0
        chunk_frames = initial_chunk_frames
        chunk_index = 0

        state = _load_render_state(state_path)
        if state and state.get("signature") == signature:
            start_frame = int(state.get("last_completed_frame", -1)) + 1
            chunk_frames = int(state.get("chunk_frames", chunk_frames))
            chunk_index = int(state.get("next_chunk_index", chunk_index))
            frame_counter = start_frame
            log_file.write(
                f"[STATE] Resuming from frame {start_frame} with chunk size {chunk_frames}\n"
            )
            log_file.flush()
        else:
            if state_path.exists():
                try:
                    state_path.unlink()
                except Exception:
                    pass
            if chunk_dir.exists():
                try:
                    shutil.rmtree(chunk_dir)
                except Exception:
                    pass

        chunk_dir.mkdir(parents=True, exist_ok=True)

        log_file.write(
            f"[CHUNK] Initial chunk size: {chunk_frames} (min={chunk_min_frames}, max={chunk_max_frames}, reduce={chunk_reduce_pct}%)\n"
        )
        log_file.write(f"[CHUNK] State file: {state_path}\n")
        log_file.write(f"[CHUNK] Chunk dir: {chunk_dir}\n")
        log_file.flush()

        while True:
            remaining = None
            if total_limit is not None:
                remaining = total_limit - start_frame
                if remaining <= 0:
                    break
            target_frames = chunk_frames if remaining is None else min(chunk_frames, remaining)
            if target_frames <= 0:
                break

            chunk_tmp = chunk_dir / f"chunk_{chunk_index:05d}.tmp.mp4"
            chunk_final = chunk_dir / f"chunk_{chunk_index:05d}.mp4"

            log_file.write(
                f"[CHUNK] Start chunk {chunk_index} at frame {start_frame} for {target_frames} frames\n"
            )
            log_file.flush()

            frames_iter = _frame_iter_for_chunk(start_frame, target_frames)
            written_frames = 0
            try:
                written_frames = encode_video_only(
                    frames=frames_iter,
                    out_path=chunk_tmp,
                    width=target_w,
                    height=target_h,
                    fps=fps,
                    rotate_deg=rotate_out,
                    log_stream=log_file,
                    threads=threads,
                    use_nvenc=use_hw_encoder,
                    metadata=None,
                )
                if (
                    written_frames < target_frames
                    and total_limit is not None
                    and (start_frame + written_frames) < total_limit
                ):
                    raise RuntimeError(
                        f"Chunk ended early at {written_frames}/{target_frames} frames"
                    )
                chunk_tmp.replace(chunk_final)
            except Exception as exc:
                log_file.write(f"[CHUNK] Failure on chunk {chunk_index}: {exc}\n")
                log_file.flush()
                try:
                    if chunk_tmp.exists():
                        chunk_tmp.unlink()
                except Exception:
                    pass
                try:
                    if chunk_final.exists():
                        chunk_final.unlink()
                except Exception:
                    pass
                if chunk_frames <= chunk_min_frames:
                    raise
                reduced = int(chunk_frames * (100 - chunk_reduce_pct) / 100)
                chunk_frames = max(chunk_min_frames, reduced)
                log_file.write(
                    f"[CHUNK] Reducing chunk size to {chunk_frames} after failure\n"
                )
                log_file.flush()
                continue
            finally:
                close_iter = getattr(frames_iter, "close", None)
                if callable(close_iter):
                    close_iter()

            log_file.write(
                f"[CHUNK] Success chunk {chunk_index}: {written_frames} frames\n"
            )
            log_file.flush()

            start_frame += written_frames
            frame_counter = start_frame
            chunk_index += 1

            _write_render_state(
                state_path,
                {
                    "signature": signature,
                    "last_completed_frame": start_frame - 1,
                    "chunk_frames": chunk_frames,
                    "next_chunk_index": chunk_index,
                },
                log_file,
            )

            if written_frames < target_frames:
                break

        chunk_files = sorted(chunk_dir.glob("chunk_*.mp4"))
        if not chunk_files:
            raise RuntimeError("Chunked render produced no chunks")

        concat_tmp = output_video.with_suffix(".concat_video_only.mp4")
        _concat_chunks(chunk_files, concat_tmp, log_file)
        _mux_audio(
            video_path=concat_tmp,
            audio_source=input_video,
            out_path=output_video,
            log_stream=log_file,
            metadata=metadata,
        )

        keep_chunks = os.environ.get("BGSOFT_KEEP_CHUNKS", "").strip().lower() in {"1", "true", "yes", "on"}
        if not keep_chunks:
            try:
                shutil.rmtree(chunk_dir)
            except Exception:
                pass
        try:
            state_path.unlink()
        except Exception:
            pass

    def run_pipeline(use_hw_encoder: bool) -> None:
        if chunking_enabled:
            run_pipeline_chunked(use_hw_encoder)
        else:
            run_pipeline_monolithic(use_hw_encoder)

    nvenc_lock_acquired = False

    def run_with_encoder(use_hw_encoder: bool) -> None:
        nonlocal nvenc_lock_acquired
        if use_hw_encoder:
            log_file.write("Waiting for NVENC slot...\n")
            log_file.flush()
            NVENC_LOCK.acquire()
            nvenc_lock_acquired = True
            log_file.write("NVENC slot acquired.\n")
            log_file.flush()
        try:
            run_pipeline(use_hw_encoder)
        finally:
            if use_hw_encoder and nvenc_lock_acquired:
                NVENC_LOCK.release()
                nvenc_lock_acquired = False
                log_file.write("NVENC slot released.\n")
                log_file.flush()

    # Pre-check NVENC availability to avoid wasted encoding attempts
    if use_nvenc:
        log_file.write("Testing NVENC availability (quick 1-frame test)...\n")
        log_file.flush()
        if not _test_nvenc_quick():
            log_file.write("NVENC not available. Using CPU encoding (libx264) instead.\n")
            log_file.flush()
            use_nvenc = False
        else:
            log_file.write("NVENC available. Proceeding with GPU encoding.\n")
            log_file.flush()

    try:
        try:
            run_with_encoder(use_nvenc)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if use_nvenc and ("nvenc" in msg or "encode" in msg):
                log_file.write(f"NVENC encode failed ({exc}). Retrying with CPU/libx264.\n")
                log_file.flush()
                run_with_encoder(False)
            else:
                raise
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
    parser.add_argument("--threads", type=int, default=8, help="FFmpeg decode thread count")
    parser.add_argument("--no-nvenc", action="store_true", help="Disable NVENC and use CPU libx264 encoding")
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
        threads=args.threads,
        use_nvenc=not args.no_nvenc,
    )
