#!/usr/bin/env python3
"""PyQt5 GUI for automating BG-soft renders with local ONNX renderer."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import onnxruntime as ort

from local_renderer import render_local, render_preview_frame
from obs_controller import (
    load_settings,
    open_with_system_handler,
    save_settings,
)


def get_available_providers() -> List[tuple[str, str]]:
    """Get list of available execution providers for this system."""
    available = ort.get_available_providers()

    provider_map = {
        "TensorrtExecutionProvider": ("TensorRT (Fastest)", "tensorrt"),
        "CUDAExecutionProvider": ("CUDA (Fast)", "cuda"),
        "CoreMLExecutionProvider": ("CoreML (Apple)", "coreml"),
        "ROCmExecutionProvider": ("ROCm (AMD)", "rocm"),
        "CPUExecutionProvider": ("CPU (Fallback)", "cpu"),
    }

    # Filter to only available providers, in order of speed
    gpu_options = []
    for provider_name, (label, value) in provider_map.items():
        if provider_name in available:
            gpu_options.append((label, value))

    # Always include CPU as last resort
    if not gpu_options:
        gpu_options.append(("CPU (Fallback)", "cpu"))

    return gpu_options


GPU_OPTIONS = get_available_providers()

def get_available_models() -> List[tuple[str, str]]:
    """Auto-detect available ONNX models in models directory."""
    models_dir = pathlib.Path("models")
    if not models_dir.exists():
        return []

    model_labels = {
        "selfie_segmentation.onnx": "Selfie Segmentation",
        "mediapipe.onnx": "MediaPipe",
        "SINet_Softmax_simple.onnx": "SINet",
        "rvm_mobilenetv3_fp32.onnx": "Robust Video Matting",
        "pphumanseg_fp32.onnx": "PPHumanSeg",
        "bria_rmbg_1_4_qint8.onnx": "RMBG 1.4",
        "bria_rmbg_2_0_qint8.onnx": "RMBG 2.0",
    }

    available = []
    for model_file in sorted(models_dir.glob("*.onnx")):
        # Use label if available, otherwise use filename
        label = model_labels.get(model_file.name, model_file.stem)
        available.append((label, str(model_file)))

    return available


MODEL_OPTIONS = get_available_models()


def material_icon(kind: str, size: int = 24, color: QtGui.QColor | str = QtCore.Qt.white) -> QtGui.QIcon:
    """Create simple Material-style glyphs without external assets."""
    if isinstance(color, str):
        color = QtGui.QColor(color)

    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)

    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    pen_width = max(2.0, size * 0.08)
    pen = QtGui.QPen(color, pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QtGui.QBrush(color))

    c = size / 24.0  # scale factor based on Material 24px grid

    if kind == "settings":  # slider-style "tune" icon
        def slider(y: float, knob_x: float) -> None:
            painter.drawLine(QtCore.QPointF(6 * c, y), QtCore.QPointF(18 * c, y))
            painter.drawEllipse(QtCore.QPointF(knob_x, y), 2.8 * c, 2.8 * c)

        slider(7 * c, 14 * c)
        slider(12 * c, 8.5 * c)
        slider(17 * c, 15 * c)
    elif kind == "add":
        painter.drawLine(QtCore.QPointF(12 * c, 6 * c), QtCore.QPointF(12 * c, 18 * c))
        painter.drawLine(QtCore.QPointF(6 * c, 12 * c), QtCore.QPointF(18 * c, 12 * c))
    elif kind == "play":
        path = QtGui.QPainterPath()
        path.moveTo(9 * c, 6 * c)
        path.lineTo(19 * c, 12 * c)
        path.lineTo(9 * c, 18 * c)
        path.closeSubpath()
        painter.fillPath(path, color)
    elif kind == "folder":
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(QtCore.QRectF(4 * c, 7 * c, 16 * c, 11 * c), 2 * c, 2 * c)
        painter.drawLine(QtCore.QPointF(4 * c, 9 * c), QtCore.QPointF(20 * c, 9 * c))
        tab_path = QtGui.QPainterPath()
        tab_path.moveTo(10 * c, 7 * c)
        tab_path.lineTo(12 * c, 5 * c)
        tab_path.lineTo(16 * c, 5 * c)
        tab_path.lineTo(18 * c, 7 * c)
        painter.fillPath(tab_path, color)
    else:
        painter.drawEllipse(QtCore.QRectF(6 * c, 6 * c, 12 * c, 12 * c))

    painter.end()
    return QtGui.QIcon(pixmap)


@dataclass
class FileState:
    """UI state for each queued video."""

    path: pathlib.Path
    rotation: int = 0  # CCW degrees
    base_pixmap: Optional[QtGui.QPixmap] = None
    label: Optional[QtWidgets.QLabel] = None


class ThumbnailWorker(QtCore.QThread):
    """Background worker to extract first-frame thumbnails without blocking the UI."""

    thumbnail_ready = QtCore.pyqtSignal(str, bytes, int)  # path, image bytes, detected rotation

    def __init__(self, paths: List[pathlib.Path], target_width: int = 220) -> None:
        super().__init__()
        self.paths = paths
        self.target_width = target_width

    def run(self) -> None:  # type: ignore[override]
        for path in self.paths:
            rotation = self._probe_rotation(path)
            data = self._extract_png(path)
            self.thumbnail_ready.emit(str(path), data, rotation)

    def _probe_rotation(self, path: pathlib.Path) -> int:
        """Read rotation metadata (in degrees, typically clockwise in file tags)."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream_tags=rotate:stream=side_data_list",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                payload = json.loads(result.stdout)
                streams = payload.get("streams", [])
                if streams:
                    stream = streams[0]
                    rotate_tag = stream.get("tags", {}).get("rotate")
                    if rotate_tag:
                        return int(float(rotate_tag)) % 360
                    for side_data in stream.get("side_data_list", []) or []:
                        if str(side_data.get("side_data_type", "")).lower().startswith("display matrix"):
                            rotation = side_data.get("rotation")
                            if rotation is not None:
                                return int(float(rotation)) % 360
        except Exception:
            pass
        return 0

    def _extract_png(self, path: pathlib.Path) -> bytes:
        """Grab the first frame as PNG bytes."""
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-noautorotate",
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale={self.target_width}:-1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except Exception:
            pass
        return b""


@dataclass
class LocalRenderOptions:
    """Settings for the local renderer."""

    enabled: bool = True
    output_subdir: str = "soft_without_obs"
    model_path: pathlib.Path = pathlib.Path("models/selfie_segmentation.onnx")
    settings_path: pathlib.Path = pathlib.Path("settings.json")
    extra_rotation_ccw: int = 0
    threads: int = 8  # FFmpeg decode threads
    # Blur and background removal settings
    blur_background: int = 6
    mask_expansion: int = -2
    feather: float = 0.1
    smooth_contour: float = 0.1
    transparency_threshold: float = 0.8

    def resolve_output_dir(self, video: pathlib.Path) -> pathlib.Path:
        """Place renders next to the source, parallel zu 'soft' falls vorhanden."""
        parent = video.parent
        if parent.name.lower() == "soft":
            return parent.parent / self.output_subdir
        return parent / self.output_subdir

    def build_output_path(self, video: pathlib.Path) -> pathlib.Path:
        dest_dir = self.resolve_output_dir(video)
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return dest_dir / f"{video.stem}_soft_no_obs_{timestamp}{video.suffix}"


def _build_local_settings_from_file() -> LocalRenderOptions:
    """Load LocalRenderOptions including blur settings from settings.json."""
    import os
    settings = load_settings()
    local_cfg = settings.get("local_render", {})
    blur_cfg = settings.get("background_removal", {})
    defaults = LocalRenderOptions()

    # Default threads to system CPU count
    default_threads = os.cpu_count() or 8

    return LocalRenderOptions(
        enabled=local_cfg.get("enabled", defaults.enabled),
        output_subdir=local_cfg.get("output_subdir", defaults.output_subdir),
        model_path=pathlib.Path(local_cfg.get("model_path", defaults.model_path)),
        settings_path=pathlib.Path(local_cfg.get("settings_path", defaults.settings_path)),
        extra_rotation_ccw=int(local_cfg.get("extra_rotation_ccw", defaults.extra_rotation_ccw)),
        threads=int(local_cfg.get("threads", default_threads)),
        blur_background=blur_cfg.get("blur_background", defaults.blur_background),
        mask_expansion=blur_cfg.get("mask_expansion", defaults.mask_expansion),
        feather=blur_cfg.get("feather", defaults.feather),
        smooth_contour=blur_cfg.get("smooth_contour", defaults.smooth_contour),
        transparency_threshold=blur_cfg.get("transparency_threshold", defaults.transparency_threshold),
    )


class LocalSettingsWidget(QtWidgets.QGroupBox):
    """Controls for the local renderer."""

    def __init__(self, defaults: Optional[LocalRenderOptions] = None) -> None:
        super().__init__("Renderoptionen")
        form = QtWidgets.QFormLayout(self)

        self.enable_local = QtWidgets.QCheckBox("Lokalen Renderer verwenden (kein OBS)")
        self.enable_local.setChecked(defaults.enabled if defaults else True)

        self.output_subdir = QtWidgets.QLineEdit(defaults.output_subdir if defaults else "soft_without_obs")

        self.rotation = QtWidgets.QComboBox()
        self.rotation.addItem("Keine Zusatzrotation", 0)
        self.rotation.addItem("90° CCW (links)", 90)
        self.rotation.addItem("90° CW (rechts)", -90)
        self.rotation.addItem("180°", 180)
        if defaults:
            idx_rot = max(0, self.rotation.findData(defaults.extra_rotation_ccw))
            self.rotation.setCurrentIndex(idx_rot)

        # Thread selector - create options based on system CPU count
        import os
        import multiprocessing
        max_threads = os.cpu_count() or 8
        self.threads = QtWidgets.QComboBox()
        for i in range(1, max_threads + 1):
            label = f"{i} Core{'s' if i > 1 else ''}"
            if i == max_threads:
                label += " (Max)"
            self.threads.addItem(label, i)
        # Set to saved value or max threads
        default_threads = defaults.threads if defaults else max_threads
        thread_idx = max(0, self.threads.findData(default_threads))
        self.threads.setCurrentIndex(thread_idx)
        self.threads.setToolTip(
            f"Anzahl der CPU-Kerne für FFmpeg-Dekodierung.\n"
            f"System verfügbar: {max_threads} Cores"
        )

        # Model selector with dropdown + custom file picker
        self.model_select = QtWidgets.QComboBox()
        for label, path in MODEL_OPTIONS:
            self.model_select.addItem(label, path)
        self.model_select.insertSeparator(self.model_select.count())
        self.model_select.addItem("Benutzerdefiniert...", None)

        # Set to default or user's selection
        default_model = str(defaults.model_path) if defaults else "models/selfie_segmentation.onnx"
        for i in range(self.model_select.count()):
            if self.model_select.itemData(i) == default_model:
                self.model_select.setCurrentIndex(i)
                break

        self.model_path_display = QtWidgets.QLineEdit(default_model)
        self.model_path_display.setReadOnly(True)
        self.model_path_display.setStyleSheet("color: #666;")

        model_layout = QtWidgets.QVBoxLayout()
        model_layout.addWidget(self.model_select)
        model_layout.addWidget(self.model_path_display)

        model_container = QtWidgets.QWidget()
        model_container.setLayout(model_layout)

        # Blur effect controls
        self.blur_background = QtWidgets.QSpinBox()
        self.blur_background.setRange(0, 20)
        self.blur_background.setValue(defaults.blur_background if defaults else 6)
        self.blur_background.setToolTip(
            "Stärke der Hintergrund-Unschärfe.\n"
            "0 = Schwarzer Hintergrund (kein Blur)\n"
            "Hoch = Unscharfer Hintergrund (20 = sehr starker Blur)"
        )

        self.mask_expansion = QtWidgets.QSpinBox()
        self.mask_expansion.setRange(-30, 30)
        self.mask_expansion.setValue(defaults.mask_expansion if defaults else -2)
        self.mask_expansion.setToolTip(
            "Anpassung der Maskengröße.\n"
            "Negativ = Maske kleiner (mehr Hintergrund entfernt)\n"
            "Positiv = Maske größer (mehr Vordergrund behalten)"
        )

        self.feather = QtWidgets.QDoubleSpinBox()
        self.feather.setRange(0.0, 1.0)
        self.feather.setSingleStep(0.05)
        self.feather.setValue(defaults.feather if defaults else 0.1)
        self.feather.setToolTip(
            "Zusätzliche Weichzeichnung der Maskenkanten.\n"
            "0.0 = Scharfe Kanten\n"
            "Hoch = Weiche, verschwommene Übergänge"
        )

        self.smooth_contour = QtWidgets.QDoubleSpinBox()
        self.smooth_contour.setRange(0.0, 1.0)
        self.smooth_contour.setSingleStep(0.05)
        self.smooth_contour.setValue(defaults.smooth_contour if defaults else 0.1)
        self.smooth_contour.setToolTip(
            "Glättung der Maskenkonturen.\n"
            "0.0 = Raue, pixelige Kanten\n"
            "Hoch = Glatte, organische Konturen"
        )

        self.transparency_threshold = QtWidgets.QDoubleSpinBox()
        self.transparency_threshold.setRange(0.0, 1.0)
        self.transparency_threshold.setSingleStep(0.05)
        self.transparency_threshold.setValue(defaults.transparency_threshold if defaults else 0.8)
        self.transparency_threshold.setToolTip(
            "Grenzwert für Vordergrund/Hintergrund-Erkennung.\n"
            "Niedrig (0.0) = Mehr Vordergrund behalten, Haare/Details\n"
            "Hoch (1.0) = Nur sicherer Vordergrund, weniger Details"
        )

        form.addRow(self.enable_local)
        form.addRow("Ausgabe-Unterordner", self.output_subdir)
        form.addRow("Zusatzrotation", self.rotation)
        form.addRow("Dekodierungs-Threads", self.threads)
        form.addRow("ONNX Modell", model_container)
        form.addRow("Blur Hintergrund", self.blur_background)
        form.addRow("Masken-Expansion", self.mask_expansion)
        form.addRow("Feather", self.feather)
        form.addRow("Kontur glätten", self.smooth_contour)
        form.addRow("Transparenz-Grenzwert", self.transparency_threshold)

        self.model_select.currentIndexChanged.connect(self._on_model_changed)

    def _on_model_changed(self) -> None:
        selected_data = self.model_select.currentData()
        if selected_data is None:
            # "Benutzerdefiniert..." selected
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "ONNX Modell wählen", "", "ONNX (*.onnx);;Alle Dateien (*)")
            if path:
                self.model_path_display.setText(path)
            else:
                # Reset to previous selection if user cancels
                self.model_select.blockSignals(True)
                self.model_select.setCurrentIndex(0)
                self.model_select.blockSignals(False)
                self.model_path_display.setText(str(self.model_select.currentData()))
        else:
            self.model_path_display.setText(selected_data)

    def get_settings(self) -> LocalRenderOptions:
        return LocalRenderOptions(
            enabled=self.enable_local.isChecked(),
            output_subdir=self.output_subdir.text().strip() or "soft_without_obs",
            model_path=pathlib.Path(self.model_path_display.text().strip() or "models/selfie_segmentation.onnx"),
            extra_rotation_ccw=int(self.rotation.currentData()),
            threads=int(self.threads.currentData()),
            blur_background=int(self.blur_background.value()),
            mask_expansion=int(self.mask_expansion.value()),
            feather=float(self.feather.value()),
            smooth_contour=float(self.smooth_contour.value()),
            transparency_threshold=float(self.transparency_threshold.value()),
        )

    def set_settings(self, opts: LocalRenderOptions) -> None:
        self.enable_local.setChecked(opts.enabled)
        self.output_subdir.setText(opts.output_subdir)
        idx_rot = max(0, self.rotation.findData(opts.extra_rotation_ccw))
        self.rotation.setCurrentIndex(idx_rot)
        thread_idx = max(0, self.threads.findData(opts.threads))
        self.threads.setCurrentIndex(thread_idx)
        self.blur_background.setValue(opts.blur_background)
        self.mask_expansion.setValue(opts.mask_expansion)
        self.feather.setValue(opts.feather)
        self.smooth_contour.setValue(opts.smooth_contour)
        self.transparency_threshold.setValue(opts.transparency_threshold)

        # Set model selector to match current setting
        model_str = str(opts.model_path)
        for i in range(self.model_select.count()):
            if self.model_select.itemData(i) == model_str:
                self.model_select.setCurrentIndex(i)
                return
        # If not found in predefined list, set as custom
        self.model_path_display.setText(model_str)


class FileTable(QtWidgets.QTableWidget):
    HEADERS = ["", "Thumbnail", "Input", "Status", "Output", "Cancel"]
    COL_CHECK = 0
    COL_THUMB = 1
    COL_PATH = 2
    COL_STATUS = 3
    COL_OUTPUT = 4
    COL_CANCEL = 5

    # Signal emitted when file count changes
    files_changed = QtCore.pyqtSignal(int)  # Emits current row count

    def __init__(self) -> None:
        super().__init__(0, len(self.HEADERS))
        self.setHorizontalHeaderLabels(self.HEADERS)
        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        for col in range(len(self.HEADERS)):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.Interactive)
        header.setMinimumSectionSize(80)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.path_to_row: Dict[str, int] = {}
        self.path_state: Dict[str, FileState] = {}
        vheader = self.verticalHeader()
        vheader.setDefaultSectionSize(140)
        vheader.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self._update_column_widths()

    def _rebuild_index(self) -> None:
        self.path_to_row.clear()
        for row in range(self.rowCount()):
            path_item = self.item(row, self.COL_PATH)
            if path_item:
                path = path_item.text()
                self.path_to_row[path] = row

    def add_files(self, paths: List[pathlib.Path]) -> List[pathlib.Path]:
        added: List[pathlib.Path] = []
        for path in paths:
            path_str = str(path)
            if path_str in self.path_to_row:
                continue
            row = self.rowCount()
            self.insertRow(row)
            self.path_to_row[path_str] = row
            self.path_state[path_str] = FileState(path=path)

            check_item = QtWidgets.QTableWidgetItem()
            check_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            check_item.setCheckState(QtCore.Qt.Unchecked)
            self.setItem(row, self.COL_CHECK, check_item)

            thumb_widget = self._build_thumb_cell(self.path_state[path_str])
            self.setCellWidget(row, self.COL_THUMB, thumb_widget)
            self.setRowHeight(row, 140)

            self.setItem(row, self.COL_PATH, QtWidgets.QTableWidgetItem(path_str))
            self.setItem(row, self.COL_STATUS, QtWidgets.QTableWidgetItem("Wartet"))
            self.setItem(row, self.COL_OUTPUT, QtWidgets.QTableWidgetItem(""))
            cancel_btn = self._build_cancel_button(path_str)
            self.setCellWidget(row, self.COL_CANCEL, cancel_btn)
            added.append(path)

        # Emit signal when files are added
        if added:
            self.files_changed.emit(self.rowCount())

        return added

    def _build_thumb_cell(self, state: FileState) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(wrapper)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(0)

        label = QtWidgets.QLabel("Lädt…")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setFixedSize(200, 112)
        label.setStyleSheet("background-color: #222; color: #aaa; border: 1px solid #444;")
        state.label = label

        rotate_btn = QtWidgets.QToolButton()
        rotate_btn.setText("↺")
        rotate_btn.setToolTip("90° gegen den Uhrzeigersinn drehen")
        rotate_btn.setCursor(QtCore.Qt.PointingHandCursor)
        rotate_btn.setFixedSize(26, 26)
        rotate_btn.setStyleSheet(
            "QToolButton { background: rgba(0,0,0,0.6); color: white; border: 1px solid #555; border-radius: 4px; }"
            "QToolButton:hover { background: rgba(255,255,255,0.2); }"
        )
        rotate_btn.clicked.connect(lambda _=False, p=str(state.path): self.rotate_thumbnail(p))

        layout.addWidget(label, 0, 0)
        layout.addWidget(rotate_btn, 0, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)
        return wrapper

    def _build_cancel_button(self, path_str: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton("✕")
        btn.setToolTip("Diesen Eintrag entfernen")
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setFixedWidth(36)
        btn.clicked.connect(lambda _=False, p=path_str: self.cancel_path(p))
        return btn

    def _remove_rows(self, rows: List[int]) -> None:
        for row in sorted(rows, reverse=True):
            path_item = self.item(row, self.COL_PATH)
            if path_item:
                path_str = path_item.text()
                self.path_to_row.pop(path_str, None)
                self.path_state.pop(path_str, None)
            self.removeRow(row)
        self._rebuild_index()
        # Emit signal when rows are removed
        self.files_changed.emit(self.rowCount())

    def remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.selectedIndexes()}, reverse=True)
        if not rows:
            rows = [
                row
                for row in range(self.rowCount())
                if (item := self.item(row, self.COL_CHECK))
                and item.checkState() == QtCore.Qt.Checked
            ]
        if rows:
            self._remove_rows(rows)

    def clear_all(self) -> None:
        self.setRowCount(0)
        self.path_to_row.clear()
        self.path_state.clear()
        # Emit signal when all rows are cleared
        self.files_changed.emit(0)

    def get_all_paths(self) -> List[pathlib.Path]:
        return [pathlib.Path(self.item(row, self.COL_PATH).text()) for row in range(self.rowCount())]

    def set_status(self, path: pathlib.Path, status: str) -> None:
        row = self.path_to_row.get(str(path))
        if row is None:
            return
        self.item(row, self.COL_STATUS).setText(status)

    def set_output(self, path: pathlib.Path, output: Optional[pathlib.Path]) -> None:
        row = self.path_to_row.get(str(path))
        if row is None:
            return
        self.item(row, self.COL_OUTPUT).setText("" if output is None else str(output))

    def rotate_thumbnail(self, path_str: str) -> None:
        state = self.path_state.get(path_str)
        if not state:
            return
        state.rotation = (state.rotation + 90) % 360  # CCW
        self._apply_pixmap(state)

    def cancel_path(self, path_str: str) -> None:
        row = self.path_to_row.get(path_str)
        if row is None:
            return
        self._remove_rows([row])

    def set_thumbnail(self, path: pathlib.Path, pixmap: Optional[QtGui.QPixmap], detected_rotation: int = 0) -> None:
        """Update thumbnail + initial rotation if we haven't rotated manually yet."""
        path_str = str(path)
        state = self.path_state.get(path_str)
        if not state:
            return
        if pixmap and not pixmap.isNull():
            state.base_pixmap = pixmap
        if state.rotation == 0 and detected_rotation:
            # Detected rotation from metadata is usually clockwise; we store CCW.
            state.rotation = (-detected_rotation) % 360
        self._apply_pixmap(state)
        if state.base_pixmap is None and state.label:
            state.label.setText("Keine Vorschau")

    def _apply_pixmap(self, state: FileState) -> None:
        if not state.label or state.base_pixmap is None:
            return
        transform = QtGui.QTransform()
        transform.rotate(state.rotation % 360)
        rotated = state.base_pixmap.transformed(transform, QtCore.Qt.SmoothTransformation)
        target_size = state.label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = QtCore.QSize(160, 90)
        scaled = rotated.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        state.label.setPixmap(scaled)
        state.label.setText("")
        state.label.setAlignment(QtCore.Qt.AlignCenter)

    def get_rotations(self) -> Dict[str, int]:
        return {path: state.rotation % 360 for path, state in self.path_state.items()}

    def _update_column_widths(self) -> None:
        total = max(1, self.viewport().width())
        ratios = (0.05, 0.22, 0.30, 0.18, 0.17, 0.08)
        widths = [int(total * r) for r in ratios]
        widths[-1] = total - sum(widths[:-1])
        min_widths = [40, 160, 160, 110, 140, 70]
        for idx, width in enumerate(widths):
            self.setColumnWidth(idx, max(width, min_widths[idx]))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_column_widths()

    def selected_output_path(self) -> Optional[pathlib.Path]:
        selected = self.selectionModel().selectedRows()
        if not selected:
            return None
        row = selected[0].row()
        text = self.item(row, self.COL_OUTPUT).text().strip()
        return pathlib.Path(text) if text else None

    def _show_context_menu(self, pos: QtCore.QPoint) -> None:
        global_pos = self.viewport().mapToGlobal(pos)
        target_index = self.indexAt(pos)
        target_row = target_index.row() if target_index.isValid() else -1

        menu = QtWidgets.QMenu(self)
        remove_checked = menu.addAction("Markierte entfernen")
        remove_all = menu.addAction("Alle entfernen")
        clicked_remove = menu.addAction("Diesen Eintrag entfernen")

        action = menu.exec_(global_pos)
        if action is None:
            return

        if action == remove_all:
            self.clear_all()
            return

        if action == clicked_remove and target_row >= 0:
            self._remove_rows([target_row])
            return

        rows = []
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item and item.checkState() == QtCore.Qt.Checked:
                rows.append(row)
        if not rows and target_row >= 0:
            rows = [target_row]
        self._remove_rows(rows)


class RenderWorker(QtCore.QThread):
    file_started = QtCore.pyqtSignal(str)
    file_progress = QtCore.pyqtSignal(str, int)  # path, percentage (0-100)
    file_finished = QtCore.pyqtSignal(str, str)
    file_failed = QtCore.pyqtSignal(str, str)
    batch_completed = QtCore.pyqtSignal(int, int)

    def __init__(
        self,
        files: List[pathlib.Path],
        local_options: LocalRenderOptions,
        rotations: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.files = files
        self.local_options = local_options
        self.rotations = rotations or {}

    def run(self) -> None:
        successes = 0
        failures = 0
        try:
            for video in self.files:
                self.file_started.emit(str(video))
                try:
                    rotation = self.rotations.get(str(video), 0)
                    output = self._render_local(video, rotation)
                except Exception as exc:  # noqa: BLE001
                    # Check if output file was created despite exception
                    # (sometimes encoding completes but trailing code fails)
                    output_path = self.local_options.build_output_path(video)
                    if output_path.exists() and output_path.stat().st_size > 0:
                        # Rendering succeeded even if exception was raised
                        successes += 1
                        self.file_finished.emit(str(video), str(output_path))
                    else:
                        failures += 1
                        self.file_failed.emit(str(video), str(exc))
                else:
                    successes += 1
                    self.file_finished.emit(str(video), str(output))
        finally:
            self.batch_completed.emit(successes, failures)

    def _render_local(self, video: pathlib.Path, rotation_ccw: int) -> pathlib.Path:
        import threading

        target = self.local_options.build_output_path(video)
        target.parent.mkdir(parents=True, exist_ok=True)
        rotate_ccw = (self.local_options.extra_rotation_ccw + rotation_ccw) % 360

        # Start progress monitoring thread
        log_path = video.parent / f"{video.stem}_bgsoft_no_obs.log"
        progress_thread = threading.Thread(
            target=self._monitor_progress,
            args=(str(video), log_path),
            daemon=True,
        )
        progress_thread.start()

        try:
            render_local(
                input_video=video,
                output_video=target,
                model_path=self.local_options.model_path,
                settings_path=self.local_options.settings_path,
                background_settings=self.local_options,
                extra_rotation_ccw=rotate_ccw,
                threads=self.local_options.threads,
            )
        finally:
            # Ensure progress shows 100%
            self.file_progress.emit(str(video), 100)

        return target

    def _monitor_progress(self, video_path: str, log_path: pathlib.Path) -> None:
        """Monitor render progress from log file and emit progress updates."""
        import time
        import re

        total_frames = None
        frames_processed = 0
        last_update = 0
        last_progress_update = time.time()  # Initialize with current time, not 0
        total_frames_found_time = 0
        monitor_start_time = time.time()
        max_monitor_time = 3600  # 1 hour max monitoring per file

        while True:
            try:
                if not log_path.exists():
                    time.sleep(0.5)
                    continue

                with open(log_path, "r") as f:
                    content = f.read()

                # Get total frames from duration and fps (look for "Probe:" line)
                if total_frames is None:
                    probe_match = re.search(r"Probe:.*fps=([\d.]+).*duration=([\d.]+)", content)
                    if probe_match:
                        fps = float(probe_match.group(1))
                        duration = float(probe_match.group(2))
                        total_frames = int(duration * fps)
                        total_frames_found_time = time.time()
                        last_progress_update = time.time()  # Reset timeout once we have frame count

                # Parse frames processed - get the LAST occurrence (most recent)
                lines = content.split("\n")
                for line in reversed(lines):
                    if "frames processed" in line:
                        try:
                            # Match both "Progress: X frames processed" and "Frames processed: X"
                            import re
                            match = re.search(r'(\d+)\s+frames processed', line)
                            if match:
                                frames_processed = int(match.group(1))
                                last_progress_update = time.time()
                                break
                        except (ValueError, IndexError):
                            pass

                # Calculate and emit progress (only after total_frames is known)
                if total_frames and total_frames > 0 and time.time() - total_frames_found_time > 0.5:
                    percentage = int((frames_processed / total_frames) * 100)
                    # Cap at 99% until we confirm completion
                    percentage = min(max(percentage, 0), 99)
                    current_time = time.time()
                    if current_time - last_update > 0.5:  # Update every 500ms
                        self.file_progress.emit(video_path, percentage)
                        last_update = current_time

                time.sleep(0.2)

                # Check if render completed
                if "Completed successfully:" in content:
                    break

                # Timeout 1: if no progress for 30 seconds after we started seeing frames (increased from 15)
                if frames_processed > 0 and time.time() - last_progress_update > 30:
                    break

                # Timeout 2: if monitoring takes too long overall (1 hour max)
                if time.time() - monitor_start_time > max_monitor_time:
                    break

            except Exception:
                time.sleep(1)


class PreviewRenderWorker(QtCore.QThread):
    """Background worker for rendering preview frames."""

    preview_ready = QtCore.pyqtSignal(object, object, str)  # Emits (original_pixmap, rendered_pixmap, error_message)

    def __init__(self, video_path: pathlib.Path, settings: LocalRenderOptions) -> None:
        super().__init__()
        self.video_path = video_path
        self.settings = settings

    def run(self) -> None:
        """Render preview frames in background thread."""
        original_pixmap = None
        rendered_pixmap = None
        error_msg = ""

        try:
            # Validate inputs first
            if not self.video_path or not self.video_path.exists():
                error_msg = "Videodatei nicht gefunden"
                self.preview_ready.emit(QtGui.QPixmap(), QtGui.QPixmap(), error_msg)
                return

            if not self.settings.model_path or not self.settings.model_path.exists():
                error_msg = "Modell nicht gefunden"
                self.preview_ready.emit(QtGui.QPixmap(), QtGui.QPixmap(), error_msg)
                return

            # Load original frame
            from local_renderer import probe_video, iter_frames

            try:
                probe = probe_video(self.video_path)
                if probe.width > 0 and probe.height > 0:
                    for idx, frame in enumerate(iter_frames(self.video_path, probe.width, probe.height, log_stream=None)):
                        if idx == 0:
                            original_pixmap = self._numpy_to_pixmap(frame)
                            break
            except Exception as e:
                # Try to extract useful error message
                error_str = str(e)
                if "ffmpeg" in error_str.lower():
                    error_msg = "FFmpeg Fehler - Video lesbar?"
                elif "cuda" in error_str.lower():
                    error_msg = "CUDA nicht verfügbar"
                else:
                    error_msg = "Frame konnte nicht geladen werden"
                pass

            # Render preview frame with current settings
            try:
                rendered_array = render_preview_frame(
                    self.video_path,
                    self.settings.model_path,
                    frame_index=0,
                    blur_background=self.settings.blur_background,
                    mask_expansion=self.settings.mask_expansion,
                    feather=self.settings.feather,
                    smooth_contour=self.settings.smooth_contour,
                    transparency_threshold=self.settings.transparency_threshold,
                )
                if rendered_array is not None:
                    rendered_pixmap = self._numpy_to_pixmap(rendered_array)
                elif not error_msg:
                    error_msg = "Rendering fehlgeschlagen"
            except Exception as e:
                error_str = str(e)
                if "model" in error_str.lower() or "onnx" in error_str.lower():
                    error_msg = "Modell-Fehler: GPU vorhanden?"
                elif not error_msg:
                    error_msg = "Rendering-Fehler"
                pass

        except Exception as e:
            error_msg = f"Fehler: {str(e)[:30]}"

        # Always emit signal
        try:
            self.preview_ready.emit(
                original_pixmap if original_pixmap else QtGui.QPixmap(),
                rendered_pixmap if rendered_pixmap else QtGui.QPixmap(),
                error_msg
            )
        except Exception:
            # Last resort - emit empty response
            self.preview_ready.emit(QtGui.QPixmap(), QtGui.QPixmap(), "Fehler")

    @staticmethod
    def _numpy_to_pixmap(arr: np.ndarray) -> Optional[QtGui.QPixmap]:
        """Convert RGB numpy array to QPixmap."""
        try:
            if arr is None or arr.size == 0:
                return None
            h, w, ch = arr.shape
            if ch != 3:
                return None
            # Ensure array is contiguous and make a copy for safety
            arr_rgb = np.ascontiguousarray(arr.astype(np.uint8))
            bytes_per_line = 3 * w
            q_img = QtGui.QImage(arr_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            return QtGui.QPixmap.fromImage(q_img)
        except Exception:
            return None


class LoupeLabel(QtWidgets.QLabel):
    """Custom label with loupe magnification on hover."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.loupe_size = 180  # Size of magnification window
        self.loupe_zoom = 3  # Magnification factor (1:1 original pixels)
        self.loupe_pos: Optional[QtCore.QPoint] = None
        self.show_loupe = False
        self.original_pixmap: Optional[QtGui.QPixmap] = None
        self.setMouseTracking(True)

    def set_original_pixmap(self, pixmap: Optional[QtGui.QPixmap]) -> None:
        """Store original full-resolution pixmap for loupe."""
        self.original_pixmap = pixmap

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Track mouse position for loupe."""
        self.loupe_pos = event.pos()
        self.show_loupe = True
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtGui.QEvent) -> None:
        """Hide loupe when mouse leaves."""
        self.show_loupe = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint frame and loupe overlay."""
        super().paintEvent(event)

        # Draw loupe overlay if active and we have original pixmap
        if self.show_loupe and self.loupe_pos and self.original_pixmap:
            try:
                painter = QtGui.QPainter(self)

                # Get the displayed pixmap's position and size
                pixmap = self.pixmap()
                if not pixmap:
                    return

                # Calculate zoom scale (displayed size vs original size)
                displayed_w = self.width()
                displayed_h = self.height()
                orig_w = self.original_pixmap.width()
                orig_h = self.original_pixmap.height()

                # Get pixel coordinates in original image
                scale_x = orig_w / displayed_w if displayed_w > 0 else 1
                scale_y = orig_h / displayed_h if displayed_h > 0 else 1
                orig_x = int(self.loupe_pos.x() * scale_x)
                orig_y = int(self.loupe_pos.y() * scale_y)

                # Clamp to valid range
                orig_x = max(0, min(orig_x, orig_w - 1))
                orig_y = max(0, min(orig_y, orig_h - 1))

                # Extract region from original pixmap
                region_size = self.loupe_size // self.loupe_zoom
                x1 = max(0, orig_x - region_size // 2)
                y1 = max(0, orig_y - region_size // 2)
                x2 = min(orig_w, x1 + region_size)
                y2 = min(orig_h, y1 + region_size)

                region = self.original_pixmap.copy(x1, y1, x2 - x1, y2 - y1)
                magnified = region.scaledToWidth(self.loupe_size, QtCore.Qt.FastTransformation)

                # Draw loupe window
                loupe_rect = QtCore.QRect(
                    self.loupe_pos.x() - self.loupe_size // 2,
                    self.loupe_pos.y() - self.loupe_size // 2,
                    self.loupe_size,
                    self.loupe_size,
                )

                # Clamp loupe position to widget bounds
                if loupe_rect.left() < 0:
                    loupe_rect.moveLeft(0)
                if loupe_rect.top() < 0:
                    loupe_rect.moveTop(0)
                if loupe_rect.right() > self.width():
                    loupe_rect.moveRight(self.width())
                if loupe_rect.bottom() > self.height():
                    loupe_rect.moveBottom(self.height())

                # Draw magnified region
                painter.drawPixmap(loupe_rect, magnified)

                # Draw border
                painter.setPen(QtGui.QPen(QtCore.Qt.white, 2))
                painter.drawRect(loupe_rect)

                # Draw crosshair
                center_x = loupe_rect.center().x()
                center_y = loupe_rect.center().y()
                painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 1))
                painter.drawLine(center_x - 5, center_y, center_x + 5, center_y)
                painter.drawLine(center_x, center_y - 5, center_x, center_y + 5)

                painter.end()
            except Exception:
                pass


class FramePreviewWidget(QtWidgets.QWidget):
    """Preview widget showing original and rendered frames side-by-side with loupe tool."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.original_frame: Optional[QtGui.QPixmap] = None
        self.rendered_frame: Optional[QtGui.QPixmap] = None
        self.error_message: Optional[str] = None

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Original frame with loupe
        self.original_label = LoupeLabel()
        self.original_label.setText("Original")
        self.original_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_label.setMinimumSize(300, 200)
        self.original_label.setStyleSheet("background-color: #222; color: #888; border: 1px solid #444;")

        # Rendered frame with loupe
        self.rendered_label = LoupeLabel()
        self.rendered_label.setText("Gerendert")
        self.rendered_label.setAlignment(QtCore.Qt.AlignCenter)
        self.rendered_label.setMinimumSize(300, 200)
        self.rendered_label.setStyleSheet("background-color: #222; color: #888; border: 1px solid #444;")

        layout.addWidget(self.original_label)
        layout.addWidget(self.rendered_label)

    def set_frames(self, original_frame: Optional[QtGui.QPixmap], rendered_frame: Optional[QtGui.QPixmap], error: Optional[str] = None) -> None:
        """Update the displayed frames."""
        self.original_frame = original_frame
        self.rendered_frame = rendered_frame
        self.error_message = error
        self._update_display()

    def _update_display(self) -> None:
        """Update label pixmaps to fit available space."""
        # Show error if present
        if self.error_message:
            self.original_label.setPixmap(None)
            self.original_label.setText(f"❌ Fehler:\n{self.error_message}")
            self.rendered_label.setPixmap(None)
            self.rendered_label.setText("")
            return

        if self.original_frame and not self.original_frame.isNull():
            displayed = self.original_frame.scaledToWidth(300, QtCore.Qt.SmoothTransformation)
            self.original_label.setPixmap(displayed)
            self.original_label.set_original_pixmap(self.original_frame)
        else:
            self.original_label.setPixmap(None)
            self.original_label.setText("Lädt...")

        if self.rendered_frame and not self.rendered_frame.isNull():
            displayed = self.rendered_frame.scaledToWidth(300, QtCore.Qt.SmoothTransformation)
            self.rendered_label.setPixmap(displayed)
            self.rendered_label.set_original_pixmap(self.rendered_frame)
        else:
            self.rendered_label.setPixmap(None)
            self.rendered_label.setText("Lädt...")


class SettingsDialog(QtWidgets.QDialog):
    """Modal dialog for local renderer settings."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        local_defaults: LocalRenderOptions,
        video_files: Optional[List[pathlib.Path]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.resize(1200, 800)
        self.video_files = video_files or []

        self.local_widget = LocalSettingsWidget(local_defaults)

        # Settings panel
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.addWidget(self.local_widget)
        container_layout.addStretch()
        scroll.setWidget(container)

        # Main content: just settings (preview disabled - causes crashes)
        content_layout = QtWidgets.QVBoxLayout()
        content_layout.addWidget(scroll, 1)

        # Note about preview
        note = QtWidgets.QLabel("💡 Preview wird noch entwickelt...")
        note.setStyleSheet("color: #999; font-size: 11px; padding: 10px;")
        content_layout.addWidget(note)

        button_layout = QtWidgets.QHBoxLayout()
        save_default_btn = QtWidgets.QPushButton("Als Standard speichern")
        save_default_btn.setToolTip("Aktuelle Einstellungen als Standard speichern")
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Abbrechen")
        button_layout.addWidget(save_default_btn)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(content_layout)
        layout.addLayout(button_layout)

        save_default_btn.clicked.connect(self._save_as_default)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        # Note: Preview disabled due to crashes in ONNX/FFmpeg threading
        # Will be re-enabled once threading issues are resolved

    def set_values(self, local_opts: LocalRenderOptions) -> None:
        self.local_widget.set_settings(local_opts)

    def get_settings(self) -> LocalRenderOptions:
        """Get current settings from the local widget."""
        return self.local_widget.get_settings()

    def _schedule_preview_update(self) -> None:
        """Schedule preview update with debouncing (avoid excessive renders)."""
        # Cancel previous timer if any
        if self._preview_timer:
            self._preview_timer.stop()

        # Schedule new update after 500ms of no changes
        self._preview_timer = QtCore.QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.start(500)

    def _update_preview(self) -> None:
        """Update preview frames based on current settings (non-blocking)."""
        if not self.video_files:
            # No video files available
            self.preview_widget.set_frames(None, None, error="Keine Videos ausgewählt")
            return

        # Run preview rendering in background to avoid blocking UI
        try:
            worker = PreviewRenderWorker(self.video_files[0], self.get_settings())
            worker.preview_ready.connect(self._on_preview_ready)
            worker.start()
        except Exception as e:
            # On error, show error message
            self.preview_widget.set_frames(None, None, error=f"Fehler: {str(e)[:40]}")

    def _on_preview_ready(self, original_pixmap: object, rendered_pixmap: object, error_msg: str = "") -> None:
        """Handle preview render completion."""
        # Convert from signal (which sends object type)
        orig = original_pixmap if isinstance(original_pixmap, QtGui.QPixmap) and not original_pixmap.isNull() else None
        rend = rendered_pixmap if isinstance(rendered_pixmap, QtGui.QPixmap) and not rendered_pixmap.isNull() else None
        self.preview_widget.set_frames(orig, rend, error=error_msg if error_msg else None)

    @staticmethod
    def _numpy_to_pixmap(arr: np.ndarray) -> Optional[QtGui.QPixmap]:
        """Convert RGB numpy array to QPixmap."""
        try:
            if arr is None or arr.size == 0:
                return None
            h, w, ch = arr.shape
            if ch != 3:
                return None
            # Ensure array is contiguous and make a copy for safety
            arr_rgb = np.ascontiguousarray(arr.astype(np.uint8))
            bytes_per_line = 3 * w
            q_img = QtGui.QImage(arr_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            return QtGui.QPixmap.fromImage(q_img)
        except Exception:
            return None

    def _save_as_default(self) -> None:
        """Save current settings as default to settings.json."""
        try:
            local_opts = self.get_settings()

            settings = {
                "background_removal": {
                    "blur_background": local_opts.blur_background,
                    "mask_expansion": local_opts.mask_expansion,
                    "feather": local_opts.feather,
                    "smooth_contour": local_opts.smooth_contour,
                    "transparency_threshold": local_opts.transparency_threshold,
                },
                "local_render": {
                    "enabled": local_opts.enabled,
                    "model_path": str(local_opts.model_path),
                    "output_subdir": local_opts.output_subdir,
                    "extra_rotation_ccw": local_opts.extra_rotation_ccw,
                },
            }

            save_settings(settings)

            QtWidgets.QMessageBox.information(
                self,
                "Erfolg",
                "Einstellungen als Standard gespeichert.",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Speichern: {e}",
            )


class MainWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BG-Soft Automatisierung")
        self.resize(1200, 600)

        self.file_table = FileTable()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        button_height = 56
        icon_size = QtCore.QSize(24, 24)

        # Top bar with file management and settings buttons
        top_layout = QtWidgets.QHBoxLayout()

        self.add_btn = QtWidgets.QPushButton("Dateien hinzufügen")
        self.add_btn.setIcon(material_icon("add"))

        self.settings_btn = QtWidgets.QPushButton("Einstellungen")
        self.settings_btn.setObjectName("settingsBtn")
        self.settings_btn.setIcon(material_icon("settings"))
        self.settings_btn.setEnabled(False)  # Disabled until videos are added
        self.settings_btn.setToolTip("Zuerst ein Video hinzufügen")

        top_layout.addWidget(self.add_btn)
        top_layout.addWidget(self.settings_btn)
        top_layout.addStretch()

        # Main batch control buttons
        controls_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Batch starten")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setIcon(material_icon("play"))
        self.open_output_btn = QtWidgets.QPushButton("Ausgabe öffnen")
        self.open_output_btn.setIcon(material_icon("folder"))

        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.open_output_btn)
        controls_layout.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top_layout)
        layout.addWidget(self.file_table, 1)
        layout.addLayout(controls_layout)
        layout.addWidget(self.log, 1)

        # Connect signals
        self.settings_btn.clicked.connect(self.open_settings)
        self.add_btn.clicked.connect(self.add_files)
        self.start_btn.clicked.connect(self.start_batch)
        self.open_output_btn.clicked.connect(self.open_selected_output)
        self.file_table.files_changed.connect(self._on_files_changed)

        self.worker: Optional[RenderWorker] = None
        self.thumbnail_workers: List[ThumbnailWorker] = []
        self.settings_dialog: Optional[SettingsDialog] = None

        # Load local renderer settings
        self.local_settings = _build_local_settings_from_file()

        # Harmonize button sizing and typography to match OBS version
        for btn in [
            self.settings_btn,
            self.add_btn,
            self.start_btn,
            self.open_output_btn,
        ]:
            btn.setMinimumHeight(button_height)
            btn.setIconSize(icon_size)
            font = btn.font()
            font.setBold(True)
            btn.setFont(font)

    def _on_files_changed(self, file_count: int) -> None:
        """Handle file count changes - enable/disable settings button."""
        has_files = file_count > 0
        self.settings_btn.setEnabled(has_files)
        if has_files:
            self.settings_btn.setToolTip("Einstellungen öffnen")
        else:
            self.settings_btn.setToolTip("Zuerst ein Video hinzufügen")

    def open_settings(self) -> None:
        """Open the settings modal dialog."""
        # Get list of all file paths for preview
        video_files = self.file_table.get_all_paths()

        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(
                self,
                self.local_settings,
                video_files=video_files,
            )
        else:
            self.settings_dialog.set_values(self.local_settings)
            self.settings_dialog.video_files = video_files
            self.settings_dialog._update_preview()

        if self.settings_dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.local_settings = self.settings_dialog.get_settings()
            self.log.appendPlainText("Einstellungen aktualisiert.")

    def add_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Videos auswählen")
        if not paths:
            return
        new_paths = self.file_table.add_files([pathlib.Path(p) for p in paths])
        if new_paths:
            self._start_thumbnail_worker(new_paths)

    def _start_thumbnail_worker(self, paths: List[pathlib.Path]) -> None:
        worker = ThumbnailWorker(paths)
        self.thumbnail_workers.append(worker)
        worker.thumbnail_ready.connect(self.on_thumbnail_ready)
        worker.finished.connect(lambda w=worker: self._remove_thumbnail_worker(w))
        worker.start()

    def _remove_thumbnail_worker(self, worker: ThumbnailWorker) -> None:
        try:
            self.thumbnail_workers.remove(worker)
        except ValueError:
            pass

    def on_thumbnail_ready(self, path_str: str, data: bytes, rotation: int) -> None:
        pixmap = QtGui.QPixmap()
        if data:
            pixmap.loadFromData(data)
        pix = pixmap if not pixmap.isNull() else None
        self.file_table.set_thumbnail(pathlib.Path(path_str), pix, rotation)

    def start_batch(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        files = self.file_table.get_all_paths()
        if not files:
            QtWidgets.QMessageBox.warning(self, "Keine Dateien", "Bitte zuerst mindestens eine Datei hinzufügen.")
            return

        self.log.appendPlainText("Starte Batch – Renderer: lokal (ONNX)")

        rotations = self.file_table.get_rotations()
        self.worker = RenderWorker(
            files,
            self.local_settings,
            rotations,
        )
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_progress.connect(self.on_file_progress)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.file_failed.connect(self.on_file_failed)
        self.worker.batch_completed.connect(self.on_batch_completed)
        self.start_btn.setEnabled(False)
        self.log.clear()
        self.worker.start()

    def on_file_started(self, path: str) -> None:
        video = pathlib.Path(path)
        self.file_table.set_status(video, "0%")
        self.log.appendPlainText(f"Starte: {video}")

    def on_file_progress(self, path: str, percentage: int) -> None:
        video = pathlib.Path(path)
        self.file_table.set_status(video, f"{percentage}%")

    def on_file_finished(self, path: str, output: str) -> None:
        video = pathlib.Path(path)
        self.file_table.set_status(video, "fertig")
        self.file_table.set_output(video, pathlib.Path(output))
        self.log.appendPlainText(f"Fertig: {video} -> {output}")

    def on_file_failed(self, path: str, error: str) -> None:
        video = pathlib.Path(path)
        self.file_table.set_status(video, "Fehler")
        self.file_table.set_output(video, None)
        self.log.appendPlainText(f"Fehler bei {video}: {error}")

    def on_batch_completed(self, successes: int, failures: int) -> None:
        self.start_btn.setEnabled(True)
        summary = f"Batch abgeschlossen – {successes} erfolgreich, {failures} fehlgeschlagen."
        self.log.appendPlainText(summary)
        QtWidgets.QMessageBox.information(self, "Batch fertig", summary)
        self.worker = None

    def open_selected_output(self) -> None:
        path = self.file_table.selected_output_path()
        if not path:
            QtWidgets.QMessageBox.information(
                self,
                "Keine Ausgabe",
                "Bitte eine fertig gerenderte Datei in der Tabelle auswählen.",
            )
            return
        try:
            open_with_system_handler(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Fehler beim Öffnen", str(exc))


def main() -> int:
    # Enable HiDPI scaling for high-resolution displays
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("BG-Soft")
    app.setApplicationVersion("1.0")
    app.setApplicationDisplayName("BG-Soft")

    # Scale fonts and UI for HiDPI
    dpi_scale = app.primaryScreen().logicalDotsPerInch() / 96.0
    if dpi_scale > 1.2:  # HiDPI detected (>120 DPI, typical is 96)
        # Scale application font
        font = app.font()
        font.setPointSize(int(font.pointSize() * 2))  # 2x larger
        app.setFont(font)

    # Load colorful stylesheet
    stylesheet_path = pathlib.Path(__file__).parent / "stylesheet.qss"
    if stylesheet_path.exists():
        with open(stylesheet_path, "r") as f:
            app.setStyleSheet(f.read())
    else:
        app.setStyle("Fusion")

    window = MainWindow()
    window.setWindowIcon(QtGui.QIcon(str(pathlib.Path(__file__).parent / "bgsoft.png")))
    window.setProperty("WM_CLASS", "bgsoft")
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
