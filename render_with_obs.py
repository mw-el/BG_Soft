#!/usr/bin/env python3
"""Automate BG-soft renders by driving OBS via its WebSocket API."""
from __future__ import annotations

import argparse
import pathlib
import sys

from obs_controller import (
    BackgroundRemovalSettings,
    ConnectionSettings,
    ObsRenderer,
    RenderError,
    SharpenSettings,
)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 4455
DEFAULT_PASSWORD = "obsstudio"
DEFAULT_SCENE = "BR-Render"
DEFAULT_INPUT = "Media Source"
DEFAULT_BACKGROUND_FILTER = "Background Removal"
DEFAULT_SHARPEN_FILTER = "Sharpen"
DEFAULT_SHARPNESS = 0.15
POLL_INTERVAL = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a single video file with OBS using a preconfigured scene, "
            "media source, and background-removal filter."
        )
    )
    parser.add_argument("input_file", help="Path to the video file that should be processed.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="OBS WebSocket host (default: localhost).")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="OBS WebSocket port (default: 4455)."
    )
    parser.add_argument(
        "--password", default=DEFAULT_PASSWORD, help="Password configured in OBS WebSocket server settings."
    )
    parser.add_argument(
        "--scene", default=DEFAULT_SCENE, help="Scene that contains the prepared media source + filter."
    )
    parser.add_argument(
        "--input", dest="input_name", default=DEFAULT_INPUT, help="Name of the media source inside that scene."
    )
    parser.add_argument(
        "--background-filter",
        default=DEFAULT_BACKGROUND_FILTER,
        help="Filter name for the Background Removal plugin (default: %(default)s).",
    )
    parser.add_argument(
        "--sharpen-filter",
        default=DEFAULT_SHARPEN_FILTER,
        help="Filter name for the Sharpen filter (default: %(default)s).",
    )
    parser.add_argument(
        "--sharpness",
        type=float,
        default=DEFAULT_SHARPNESS,
        help="Sharpen filter strength (0-1, default: %(default)s).",
    )
    parser.add_argument(
        "--poll", dest="poll_interval", type=float, default=POLL_INTERVAL, help="Seconds between OBS status checks."
    )
    parser.add_argument(
        "--skip-filter-update",
        action="store_true",
        help="Do not push Background Removal / Sharpen settings before rendering.",
    )
    return parser.parse_args()


def render_video(args: argparse.Namespace) -> pathlib.Path:
    source = pathlib.Path(args.input_file).expanduser().resolve()
    conn = ConnectionSettings(
        host=args.host,
        port=args.port,
        password=args.password,
        scene_name=args.scene,
        input_name=args.input_name,
        background_filter_name=args.background_filter,
        sharpen_filter_name=args.sharpen_filter,
    )
    renderer = ObsRenderer(conn, poll_interval=args.poll_interval)

    background_settings = None
    sharpen_settings = None
    if not args.skip_filter_update:
        background_settings = BackgroundRemovalSettings()
        sharpen_settings = SharpenSettings(sharpness=args.sharpness)

    print(f"Connecting to OBS at {args.host}:{args.port} ...")
    try:
        target = renderer.render_file(source, background_settings, sharpen_settings)
    finally:
        renderer.disconnect()
    print(f"Moved final render to {target}")
    return target


def main() -> int:
    try:
        args = parse_args()
        result = render_video(args)
        print(f"Done. Rendered file: {result}")
        return 0
    except KeyboardInterrupt:
        print("Aborted via Ctrl+C", file=sys.stderr)
        return 1
    except RenderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - show message to user
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
