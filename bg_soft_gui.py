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
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets
import onnxruntime as ort

from local_renderer import load_selfie_model, probe_video, render_local, render_preview_pair, render_preview_clip
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
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    temperature: float = 0.0

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
        brightness=blur_cfg.get("brightness", defaults.brightness),
        contrast=blur_cfg.get("contrast", defaults.contrast),
        saturation=blur_cfg.get("saturation", defaults.saturation),
        temperature=blur_cfg.get("temperature", defaults.temperature),
    )


class CollapsibleSection(QtWidgets.QWidget):
    """Simple collapsible section used for advanced settings."""

    def __init__(self, title: str, collapsed: bool = True, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._toggle = QtWidgets.QToolButton()
        self._toggle.setStyleSheet("QToolButton { border: none; }")
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setArrowType(QtCore.Qt.RightArrow if collapsed else QtCore.Qt.DownArrow)
        self._toggle.clicked.connect(self._on_toggled)

        self._content = QtWidgets.QWidget()
        self._content.setVisible(not collapsed)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content)

    def setContentWidget(self, widget: QtWidgets.QWidget) -> None:
        content_layout = QtWidgets.QVBoxLayout(self._content)
        content_layout.setContentsMargins(12, 0, 0, 0)
        content_layout.addWidget(widget)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self._content.setVisible(checked)


class LocalSettingsWidget(QtWidgets.QGroupBox):
    """Controls for the local renderer."""

    settings_changed = QtCore.pyqtSignal()

    def __init__(self, defaults: Optional[LocalRenderOptions] = None) -> None:
        super().__init__("Renderoptionen")
        main_layout = QtWidgets.QVBoxLayout(self)

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

        self.brightness = QtWidgets.QDoubleSpinBox()
        self.brightness.setRange(-1.0, 1.0)
        self.brightness.setSingleStep(0.05)
        self.brightness.setValue(defaults.brightness if defaults else 0.0)
        self.brightness.setToolTip("Helligkeit anpassen (-1.0 bis +1.0)")

        self.contrast = QtWidgets.QDoubleSpinBox()
        self.contrast.setRange(-1.0, 1.0)
        self.contrast.setSingleStep(0.05)
        self.contrast.setValue(defaults.contrast if defaults else 0.0)
        self.contrast.setToolTip("Kontrast anpassen (-1.0 bis +1.0)")

        self.saturation = QtWidgets.QDoubleSpinBox()
        self.saturation.setRange(-1.0, 1.0)
        self.saturation.setSingleStep(0.05)
        self.saturation.setValue(defaults.saturation if defaults else 0.0)
        self.saturation.setToolTip("Sättigung anpassen (-1.0 bis +1.0)")

        self.temperature = QtWidgets.QDoubleSpinBox()
        self.temperature.setRange(-1.0, 1.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(defaults.temperature if defaults else 0.0)
        self.temperature.setToolTip("Farbtemperatur anpassen (-1.0 kalt bis +1.0 warm)")

        adjustments_group = QtWidgets.QGroupBox("Bildoptimierung")
        adjustments_layout = QtWidgets.QFormLayout(adjustments_group)
        adjustments_layout.addRow("Helligkeit", self.brightness)
        adjustments_layout.addRow("Kontrast", self.contrast)
        adjustments_layout.addRow("Sättigung", self.saturation)
        adjustments_layout.addRow("Temperatur", self.temperature)

        base_form = QtWidgets.QFormLayout()
        base_form.addRow(self.enable_local)
        base_form.addRow("Ausgabe-Unterordner", self.output_subdir)
        base_form.addRow("Zusatzrotation", self.rotation)
        base_form.addRow("Dekodierungs-Threads", self.threads)
        base_form.addRow("ONNX Modell", model_container)

        bg_form = QtWidgets.QFormLayout()
        bg_form.addRow("Blur Hintergrund", self.blur_background)
        bg_form.addRow("Masken-Expansion", self.mask_expansion)
        bg_form.addRow("Feather", self.feather)
        bg_form.addRow("Kontur glätten", self.smooth_contour)
        bg_form.addRow("Transparenz-Grenzwert", self.transparency_threshold)

        bg_section = CollapsibleSection("Hintergrund- & Masken-Optionen", collapsed=True)
        bg_container = QtWidgets.QWidget()
        bg_container.setLayout(bg_form)
        bg_section.setContentWidget(bg_container)

        main_layout.addWidget(adjustments_group)
        main_layout.addLayout(base_form)
        main_layout.addWidget(bg_section)
        main_layout.addStretch(1)

        self.model_select.currentIndexChanged.connect(self._on_model_changed)
        self.enable_local.stateChanged.connect(self._emit_settings_changed)
        self.rotation.currentIndexChanged.connect(self._emit_settings_changed)
        self.threads.currentIndexChanged.connect(self._emit_settings_changed)
        self.blur_background.valueChanged.connect(self._emit_settings_changed)
        self.mask_expansion.valueChanged.connect(self._emit_settings_changed)
        self.feather.valueChanged.connect(self._emit_settings_changed)
        self.smooth_contour.valueChanged.connect(self._emit_settings_changed)
        self.transparency_threshold.valueChanged.connect(self._emit_settings_changed)
        self.brightness.valueChanged.connect(self._emit_settings_changed)
        self.contrast.valueChanged.connect(self._emit_settings_changed)
        self.saturation.valueChanged.connect(self._emit_settings_changed)
        self.temperature.valueChanged.connect(self._emit_settings_changed)

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
        self._emit_settings_changed()

    def _emit_settings_changed(self) -> None:
        self.settings_changed.emit()

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
            brightness=float(self.brightness.value()),
            contrast=float(self.contrast.value()),
            saturation=float(self.saturation.value()),
            temperature=float(self.temperature.value()),
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
        self.brightness.setValue(opts.brightness)
        self.contrast.setValue(opts.contrast)
        self.saturation.setValue(opts.saturation)
        self.temperature.setValue(opts.temperature)

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
        self.active_progress_threads: dict = {}

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

        # Start progress monitoring thread (cleanup old one if exists)
        log_path = video.parent / f"{video.stem}_bgsoft_no_obs.log"
        video_key = str(video)

        # Note: Old daemon threads will auto-stop when their target completes
        # We just track the new one
        progress_thread = threading.Thread(
            target=self._monitor_progress,
            args=(video_key, log_path),
            daemon=True,
        )
        self.active_progress_threads[video_key] = progress_thread
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


class PreviewWorker(QtCore.QObject):
    """Background worker for rendering preview frames in a dedicated thread."""

    preview_ready = QtCore.pyqtSignal(int, object, object, str)  # request_id, original_array, rendered_array, error

    def __init__(self) -> None:
        super().__init__()
        self._session: Optional[ort.InferenceSession] = None
        self._session_model_path: Optional[pathlib.Path] = None

    @QtCore.pyqtSlot(object, object, int, int)
    def render_preview(
        self,
        video_path: pathlib.Path,
        settings: LocalRenderOptions,
        frame_index: int,
        request_id: int,
    ) -> None:
        original = None
        rendered = None
        error_msg = ""

        try:
            if not video_path or not video_path.exists():
                error_msg = "Videodatei nicht gefunden"
            elif not settings.model_path or not settings.model_path.exists():
                error_msg = "Modell nicht gefunden"
            else:
                if self._session is None or self._session_model_path != settings.model_path:
                    self._session = load_selfie_model(settings.model_path)
                    self._session_model_path = settings.model_path
                original, rendered = render_preview_pair(
                    video_path,
                    settings.model_path,
                    frame_index=frame_index,
                    session=self._session,
                    blur_background=settings.blur_background,
                    mask_expansion=settings.mask_expansion,
                    feather=settings.feather,
                    smooth_contour=settings.smooth_contour,
                    transparency_threshold=settings.transparency_threshold,
                    brightness=settings.brightness,
                    contrast=settings.contrast,
                    saturation=settings.saturation,
                    temperature=settings.temperature,
                )
                if original is None and rendered is None and not error_msg:
                    error_msg = "Preview konnte nicht geladen werden"
                elif rendered is None and not error_msg:
                    error_msg = "Rendering fehlgeschlagen"
        except Exception:
            if not error_msg:
                error_msg = "Rendering-Fehler"

        self.preview_ready.emit(request_id, original, rendered, error_msg)


class PreviewClipWorker(QtCore.QObject):
    """Background worker to render a short preview clip."""

    clip_ready = QtCore.pyqtSignal(int, object, str)  # request_id, clip_path, error

    @QtCore.pyqtSlot(object, object, float, float, int)
    def render_clip(
        self,
        video_path: pathlib.Path,
        settings: LocalRenderOptions,
        start_time: float,
        duration: float,
        request_id: int,
    ) -> None:
        error_msg = ""
        clip_path = None
        try:
            if not video_path or not video_path.exists():
                error_msg = "Videodatei nicht gefunden"
            elif not settings.model_path or not settings.model_path.exists():
                error_msg = "Modell nicht gefunden"
            else:
                preview_dir = video_path.parent / ".bgsoft_preview"
                preview_dir.mkdir(parents=True, exist_ok=True)
                clip_path = preview_dir / f"{video_path.stem}_preview_optimized.mp4"
                if clip_path.exists():
                    try:
                        clip_path.unlink()
                    except Exception:
                        pass
                try:
                    render_preview_clip(
                        input_video=video_path,
                        output_video=clip_path,
                        model_path=settings.model_path,
                        start_time=start_time,
                        duration=duration,
                        background_settings=settings,
                        threads=max(1, min(4, settings.threads)),
                        use_nvenc=True,
                    )
                except Exception:
                    render_preview_clip(
                        input_video=video_path,
                        output_video=clip_path,
                        model_path=settings.model_path,
                        start_time=start_time,
                        duration=duration,
                        background_settings=settings,
                        threads=max(1, min(4, settings.threads)),
                        use_nvenc=False,
                    )
        except Exception:
            if not error_msg:
                error_msg = "Clip-Rendering fehlgeschlagen"

        self.clip_ready.emit(request_id, clip_path, error_msg)


class LoupeLabel(QtWidgets.QLabel):
    """Custom label with loupe magnification on hover."""

    loupe_event = QtCore.pyqtSignal(object, bool)  # normalized position, clicked

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.loupe_size = 180  # Size of magnification window
        self.loupe_zoom = 3  # Magnification factor (1:1 original pixels)
        self.loupe_pos: Optional[QtCore.QPoint] = None
        self.show_loupe = False
        self.original_pixmap: Optional[QtGui.QPixmap] = None
        self._loupe_norm: Optional[tuple[float, float]] = None
        self.setMouseTracking(True)

    def set_original_pixmap(self, pixmap: Optional[QtGui.QPixmap]) -> None:
        """Store original full-resolution pixmap for loupe."""
        self.original_pixmap = pixmap

    def set_loupe_normalized(self, norm: Optional[tuple[float, float]], active: bool) -> None:
        self._loupe_norm = norm
        self.show_loupe = active and norm is not None
        if self.show_loupe and norm is not None:
            rect = self._display_rect()
            if rect:
                x = rect.left() + int(norm[0] * rect.width())
                y = rect.top() + int(norm[1] * rect.height())
                self.loupe_pos = QtCore.QPoint(x, y)
            else:
                self.loupe_pos = None
                self.show_loupe = False
        else:
            self.loupe_pos = None
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Track mouse position for loupe."""
        if self.show_loupe:
            norm = self._normalized_from_pos(event.pos())
            if norm is not None:
                self.loupe_event.emit(norm, False)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            norm = self._normalized_from_pos(event.pos())
            if norm is not None:
                self.loupe_event.emit(norm, True)
        super().mousePressEvent(event)

    def leaveEvent(self, event: QtGui.QEvent) -> None:
        """Hide loupe when mouse leaves."""
        if not self.show_loupe:
            self.loupe_event.emit(None, False)
        super().leaveEvent(event)

    def _display_rect(self) -> Optional[QtCore.QRect]:
        pixmap = self.pixmap()
        if not pixmap:
            return None
        pix_w = pixmap.width()
        pix_h = pixmap.height()
        if pix_w <= 0 or pix_h <= 0:
            return None
        x = max(0, (self.width() - pix_w) // 2)
        y = max(0, (self.height() - pix_h) // 2)
        return QtCore.QRect(x, y, pix_w, pix_h)

    def _normalized_from_pos(self, pos: QtCore.QPoint) -> Optional[tuple[float, float]]:
        rect = self._display_rect()
        if not rect:
            return None
        x = min(max(pos.x(), rect.left()), rect.right())
        y = min(max(pos.y(), rect.top()), rect.bottom())
        w = max(1, rect.width())
        h = max(1, rect.height())
        return ((x - rect.left()) / w, (y - rect.top()) / h)

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

                rect = self._display_rect()
                if not rect:
                    return

                # Calculate zoom scale (displayed size vs original size)
                displayed_w = rect.width()
                displayed_h = rect.height()
                orig_w = self.original_pixmap.width()
                orig_h = self.original_pixmap.height()

                # Get pixel coordinates in original image
                scale_x = orig_w / displayed_w if displayed_w > 0 else 1
                scale_y = orig_h / displayed_h if displayed_h > 0 else 1
                pos_x = self.loupe_pos.x() - rect.left()
                pos_y = self.loupe_pos.y() - rect.top()
                orig_x = int(pos_x * scale_x)
                orig_y = int(pos_y * scale_y)

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


class PreviewPane(QtWidgets.QWidget):
    """Single preview pane with still image and video playback."""

    loupe_event = QtCore.pyqtSignal(object, bool)

    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.title = QtWidgets.QLabel(title)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet("color: #bbb; font-size: 12px; padding: 2px;")

        self.loupe_label = LoupeLabel()
        self.loupe_label.setAlignment(QtCore.Qt.AlignCenter)
        self.loupe_label.setMinimumSize(640, 360)
        self.loupe_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.loupe_label.setStyleSheet("background-color: #222; color: #888; border: 1px solid #444;")

        self.video_widget = QtMultimediaWidgets.QVideoWidget()
        self.video_widget.setMinimumSize(640, 360)
        self.video_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self._stack = QtWidgets.QStackedLayout()
        self._stack.addWidget(self.loupe_label)
        self._stack.addWidget(self.video_widget)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.title)
        layout.addLayout(self._stack, 1)

        self.loupe_label.loupe_event.connect(self.loupe_event)

    def set_pixmap(self, pixmap: Optional[QtGui.QPixmap], text: str = "") -> None:
        self.loupe_label.setPixmap(pixmap)
        if pixmap is None:
            self.loupe_label.setText(text)
        else:
            self.loupe_label.setText("")
        self.show_video(False)

    def show_video(self, enabled: bool) -> None:
        self._stack.setCurrentWidget(self.video_widget if enabled else self.loupe_label)

class FramePreviewWidget(QtWidgets.QWidget):
    """Preview widget showing original and rendered frames with linked loupe."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.original_frame: Optional[QtGui.QPixmap] = None
        self.rendered_frame: Optional[QtGui.QPixmap] = None
        self.error_message: Optional[str] = None
        self._loupe_active = False
        self._loupe_norm: Optional[tuple[float, float]] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self.original_pane = PreviewPane("Original")
        self.rendered_pane = PreviewPane("Optimiert")

        self.original_pane.loupe_event.connect(self._on_loupe_event)
        self.rendered_pane.loupe_event.connect(self._on_loupe_event)

        layout.addWidget(self.original_pane, 1)
        layout.addWidget(self.rendered_pane, 1)

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
            self.original_pane.set_pixmap(None, f"❌ Fehler:\n{self.error_message}")
            self.rendered_pane.set_pixmap(None, "")
            return

        if self.original_frame and not self.original_frame.isNull():
            displayed = self.original_frame.scaled(
                self.original_pane.loupe_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.original_pane.set_pixmap(displayed)
            self.original_pane.loupe_label.set_original_pixmap(self.original_frame)
        else:
            self.original_pane.set_pixmap(None, "Lädt...")

        if self.rendered_frame and not self.rendered_frame.isNull():
            displayed = self.rendered_frame.scaled(
                self.rendered_pane.loupe_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.rendered_pane.set_pixmap(displayed)
            self.rendered_pane.loupe_label.set_original_pixmap(self.rendered_frame)
        else:
            self.rendered_pane.set_pixmap(None, "Lädt...")

        if self._loupe_active and self._loupe_norm is not None:
            self._set_loupe(self._loupe_norm, active=True)
        else:
            self._set_loupe(None, active=False)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.original_frame or self.rendered_frame:
            self._update_display()

    def _on_loupe_event(self, norm: object, clicked: bool) -> None:
        if clicked:
            self._loupe_active = not self._loupe_active
            if not self._loupe_active:
                self._loupe_norm = None
                self._set_loupe(None, active=False)
                return
        if not self._loupe_active:
            return
        if isinstance(norm, tuple):
            self._loupe_norm = norm
            self._set_loupe(norm, active=True)

    def _set_loupe(self, norm: Optional[tuple[float, float]], active: bool) -> None:
        self.original_pane.loupe_label.set_loupe_normalized(norm, active)
        self.rendered_pane.loupe_label.set_loupe_normalized(norm, active)

    def set_video_mode(self, enabled: bool) -> None:
        self.original_pane.show_video(enabled)
        self.rendered_pane.show_video(enabled)

    @property
    def original_video_widget(self) -> QtMultimediaWidgets.QVideoWidget:
        return self.original_pane.video_widget

    @property
    def rendered_video_widget(self) -> QtMultimediaWidgets.QVideoWidget:
        return self.rendered_pane.video_widget


class SettingsDialog(QtWidgets.QDialog):
    """Modal dialog for local renderer settings."""

    preview_request = QtCore.pyqtSignal(object, object, int, int)
    clip_request = QtCore.pyqtSignal(object, object, float, float, int)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        local_defaults: LocalRenderOptions,
        video_files: Optional[List[pathlib.Path]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.resize(1500, 900)
        self.video_files = video_files or []

        self.local_widget = LocalSettingsWidget(local_defaults)
        self.local_widget.setMaximumWidth(480)
        self.local_widget.settings_changed.connect(self._schedule_preview_update)

        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_request_id = 0
        self._total_frames = 0
        self._fps = 0.0
        self._current_frame = 0

        self._preview_thread = QtCore.QThread(self)
        self._preview_worker = PreviewWorker()
        self._preview_worker.moveToThread(self._preview_thread)
        self.preview_request.connect(self._preview_worker.render_preview)
        self._preview_worker.preview_ready.connect(self._on_preview_ready)
        self._preview_thread.start()

        self._clip_thread = QtCore.QThread(self)
        self._clip_worker = PreviewClipWorker()
        self._clip_worker.moveToThread(self._clip_thread)
        self.clip_request.connect(self._clip_worker.render_clip)
        self._clip_worker.clip_ready.connect(self._on_clip_ready)
        self._clip_thread.start()
        self._clip_request_id = 0
        self._clip_stale = True
        self._rendered_clip_path: Optional[pathlib.Path] = None
        self._pending_play_after = False

        self.preview_widget = FramePreviewWidget()
        self.preview_status = QtWidgets.QLabel("")
        self.preview_status.setStyleSheet("color: #999; font-size: 11px; padding: 4px;")

        self.preview_play_btn = QtWidgets.QPushButton("▶ 5s")
        self.preview_stop_btn = QtWidgets.QPushButton("⏹")
        self.preview_play_btn.clicked.connect(self._on_play_clicked)
        self.preview_stop_btn.clicked.connect(self._stop_playback)

        self.prev_10_btn = QtWidgets.QPushButton("<< -10")
        self.prev_1_btn = QtWidgets.QPushButton("< -1")
        self.next_1_btn = QtWidgets.QPushButton("+1 >")
        self.next_10_btn = QtWidgets.QPushButton("+10 >>")
        self.prev_10_btn.clicked.connect(lambda: self._step_frames(-10))
        self.prev_1_btn.clicked.connect(lambda: self._step_frames(-1))
        self.next_1_btn.clicked.connect(lambda: self._step_frames(1))
        self.next_10_btn.clicked.connect(lambda: self._step_frames(10))

        self.frame_label = QtWidgets.QLabel("Frame 0 / 0")
        self.frame_label.setAlignment(QtCore.Qt.AlignCenter)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)

        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.valueChanged.connect(self._on_frame_spin_changed)

        self._preview_duration_sec = 5.0
        self._playback_timer = QtCore.QTimer(self)
        self._playback_timer.setSingleShot(True)
        self._playback_timer.timeout.connect(self._stop_playback)

        self._original_player = QtMultimedia.QMediaPlayer(self)
        self._rendered_player = QtMultimedia.QMediaPlayer(self)
        self._original_player.setVideoOutput(self.preview_widget.original_video_widget)
        self._rendered_player.setVideoOutput(self.preview_widget.rendered_video_widget)
        self._rendered_player.setMuted(True)

        # Settings panel
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.addWidget(self.local_widget)
        container_layout.addStretch()
        scroll.setWidget(container)
        scroll.setMaximumWidth(520)

        preview_controls = QtWidgets.QHBoxLayout()
        preview_controls.addWidget(self.preview_play_btn)
        preview_controls.addWidget(self.preview_stop_btn)
        preview_controls.addWidget(self.prev_10_btn)
        preview_controls.addWidget(self.prev_1_btn)
        preview_controls.addWidget(self.next_1_btn)
        preview_controls.addWidget(self.next_10_btn)
        preview_controls.addStretch()
        preview_controls.addWidget(self.frame_label)
        preview_controls.addStretch()
        preview_controls.addWidget(self.preview_status)

        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(self.frame_slider, 1)
        slider_layout.addWidget(self.frame_spin)

        preview_container = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_container)
        preview_layout.addWidget(self.preview_widget, 1)
        preview_layout.addLayout(preview_controls)
        preview_layout.addLayout(slider_layout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(preview_container)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1100, 400])

        content_layout = QtWidgets.QVBoxLayout()
        content_layout.addWidget(splitter, 1)

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
        self._load_preview_context()
        self._schedule_preview_update()

    def set_values(self, local_opts: LocalRenderOptions) -> None:
        self.local_widget.set_settings(local_opts)
        self._clip_stale = True
        self._rendered_clip_path = None
        self._schedule_preview_update()

    def set_video_files(self, video_files: List[pathlib.Path]) -> None:
        self.video_files = video_files
        self._rendered_clip_path = None
        self._clip_stale = True
        self._load_preview_context()
        self._schedule_preview_update()

    def get_settings(self) -> LocalRenderOptions:
        """Get current settings from the local widget."""
        return self.local_widget.get_settings()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._stop_playback()
        if self._preview_thread.isRunning():
            self._preview_thread.quit()
            self._preview_thread.wait(2000)
        if self._clip_thread.isRunning():
            self._clip_thread.quit()
            self._clip_thread.wait(2000)
        super().closeEvent(event)

    def _load_preview_context(self) -> None:
        self._total_frames = 0
        self._fps = 0.0
        if not self.video_files:
            self._set_frame_controls_enabled(False)
            self._update_frame_label()
            return

        try:
            probe = probe_video(self.video_files[0])
            self._fps = probe.fps or 30.0
            if probe.duration and self._fps:
                self._total_frames = int(round(probe.duration * self._fps))
        except Exception:
            self._total_frames = 0

        if self._total_frames <= 0:
            self._set_frame_controls_enabled(False)
        else:
            self._set_frame_controls_enabled(True)
            if self._current_frame >= self._total_frames:
                self._current_frame = max(0, self._total_frames - 1)
        self._sync_frame_controls()
        self._update_frame_label()

    def _set_frame_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.prev_10_btn,
            self.prev_1_btn,
            self.next_1_btn,
            self.next_10_btn,
            self.frame_slider,
            self.frame_spin,
        ):
            widget.setEnabled(enabled)

    def _sync_frame_controls(self) -> None:
        max_index = max(0, self._total_frames - 1)
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setRange(0, max_index)
        self.frame_spin.setRange(0, max_index)
        self.frame_slider.setValue(self._current_frame)
        self.frame_spin.setValue(self._current_frame)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)

    def _update_frame_label(self) -> None:
        if self._total_frames > 0:
            seconds = self._current_frame / self._fps if self._fps else 0.0
            self.frame_label.setText(f"Frame {self._current_frame} / {self._total_frames} ({seconds:.2f}s)")
        else:
            self.frame_label.setText("Frame -- / --")

    def _set_current_frame(self, frame_index: int, schedule: bool = True) -> None:
        if self._total_frames > 0:
            frame_index = max(0, min(frame_index, self._total_frames - 1))
        else:
            frame_index = max(0, frame_index)
        if frame_index == self._current_frame:
            return
        self._current_frame = frame_index
        self._sync_frame_controls()
        self._update_frame_label()
        if schedule:
            self._schedule_preview_update()

    def _step_frames(self, delta: int) -> None:
        self._set_current_frame(self._current_frame + delta)

    def _on_frame_slider_changed(self, value: int) -> None:
        self._set_current_frame(int(value))

    def _on_frame_spin_changed(self, value: int) -> None:
        self._set_current_frame(int(value))

    def _schedule_preview_update(self) -> None:
        """Schedule preview update with debouncing (avoid excessive renders)."""
        self._clip_stale = True
        self._stop_playback()
        if self._preview_timer.isActive():
            self._preview_timer.stop()
        self._preview_timer.start(300)

    def _update_preview(self) -> None:
        """Update preview frames based on current settings (non-blocking)."""
        if not self.video_files:
            # No video files available
            self.preview_widget.set_frames(None, None, error="Keine Videos ausgewählt")
            self.preview_status.setText("Keine Videos ausgewählt")
            return

        self._preview_request_id += 1
        self.preview_status.setText("Rendering...")
        self.preview_request.emit(self.video_files[0], self.get_settings(), self._current_frame, self._preview_request_id)

    def _on_preview_ready(
        self,
        request_id: int,
        original_frame: object,
        rendered_frame: object,
        error_msg: str = "",
    ) -> None:
        """Handle preview render completion."""
        if request_id != self._preview_request_id:
            return
        orig = self._numpy_to_pixmap(original_frame) if isinstance(original_frame, np.ndarray) else None
        rend = self._numpy_to_pixmap(rendered_frame) if isinstance(rendered_frame, np.ndarray) else None
        self.preview_widget.set_frames(orig, rend, error=error_msg if error_msg else None)
        self.preview_status.setText(error_msg if error_msg else "Fertig")

    def _on_play_clicked(self) -> None:
        if not self.video_files:
            self.preview_status.setText("Keine Videos ausgewählt")
            return

        if self._clip_stale or not self._rendered_clip_path:
            self._request_clip_render(play_after=True)
            return

        self._start_playback()

    def _request_clip_render(self, play_after: bool) -> None:
        if not self.video_files:
            return

        self._stop_playback()
        self._clip_request_id += 1
        self._pending_play_after = play_after
        start_time = self._current_frame / self._fps if self._fps else 0.0
        max_start = 0.0
        if self._total_frames and self._fps:
            max_start = max(0.0, (self._total_frames / self._fps) - self._preview_duration_sec)
        start_time = min(max_start, max(0.0, start_time))

        self.preview_status.setText("Rendering 5s Clip...")
        self.clip_request.emit(
            self.video_files[0],
            self.get_settings(),
            start_time,
            self._preview_duration_sec,
            self._clip_request_id,
        )

    def _on_clip_ready(self, request_id: int, clip_path: object, error_msg: str = "") -> None:
        if request_id != self._clip_request_id:
            return
        if error_msg:
            self.preview_status.setText(error_msg)
            return
        if isinstance(clip_path, pathlib.Path):
            self._rendered_clip_path = clip_path
            self._clip_stale = False
            if self._pending_play_after:
                self._start_playback()
        else:
            self.preview_status.setText("Clip nicht verfügbar")

    def _start_playback(self) -> None:
        if not self.video_files or not self._rendered_clip_path:
            return

        start_time_ms = int((self._current_frame / self._fps) * 1000) if self._fps else 0
        self._original_player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(str(self.video_files[0]))))
        self._rendered_player.setMedia(
            QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(str(self._rendered_clip_path)))
        )

        self._original_player.setPosition(start_time_ms)
        self._rendered_player.setPosition(0)

        self.preview_widget.set_video_mode(True)
        self._original_player.play()
        self._rendered_player.play()
        self._playback_timer.start(int(self._preview_duration_sec * 1000))
        self.preview_status.setText("Playback läuft...")

    def _stop_playback(self) -> None:
        try:
            self._playback_timer.stop()
        except Exception:
            pass
        try:
            self._original_player.stop()
            self._rendered_player.stop()
        except Exception:
            pass
        self.preview_widget.set_video_mode(False)
        if self.video_files:
            self.preview_status.setText("Fertig")

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
                    "brightness": local_opts.brightness,
                    "contrast": local_opts.contrast,
                    "saturation": local_opts.saturation,
                    "temperature": local_opts.temperature,
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
        self.active_progress_threads: dict = {}  # {video_path: thread} - Track progress monitoring threads

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
            self.settings_dialog.set_video_files(video_files)

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
        # CRITICAL: Disable button immediately to prevent race condition
        self.start_btn.setEnabled(False)

        try:
            # Defensive check: worker already running?
            if self.worker is not None and self.worker.isRunning():
                self.log.appendPlainText("⚠️ Worker läuft bereits - ignoriere Start")
                return

            files = self.file_table.get_all_paths()
            if not files:
                QtWidgets.QMessageBox.warning(self, "Keine Dateien", "Bitte zuerst mindestens eine Datei hinzufügen.")
                self.start_btn.setEnabled(True)  # Re-enable on error
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
            self.log.clear()
            self.worker.start()

        except Exception as e:
            # Re-enable button on error
            self.start_btn.setEnabled(True)
            self.log.appendPlainText(f"❌ Fehler beim Start: {e}")
            raise

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

        # Cleanup progress thread reference
        if self.worker is not None and path in getattr(self.worker, "active_progress_threads", {}):
            del self.worker.active_progress_threads[path]
        if path in self.active_progress_threads:
            del self.active_progress_threads[path]

    def on_file_failed(self, path: str, error: str) -> None:
        video = pathlib.Path(path)
        self.file_table.set_status(video, "Fehler")
        self.file_table.set_output(video, None)
        self.log.appendPlainText(f"Fehler bei {video}: {error}")
        if self.worker is not None and path in getattr(self.worker, "active_progress_threads", {}):
            del self.worker.active_progress_threads[path]
        if path in self.active_progress_threads:
            del self.active_progress_threads[path]

    def on_batch_completed(self, successes: int, failures: int) -> None:
        # Cleanup worker reference
        if self.worker is not None:
            self.worker.deleteLater()  # Qt cleanup
            self.worker = None

        # Re-enable start button
        self.start_btn.setEnabled(True)

        # Clear all progress thread references (daemon threads auto-stop)
        self.active_progress_threads.clear()

        # Show summary
        summary = f"Batch abgeschlossen – {successes} erfolgreich, {failures} fehlgeschlagen."
        self.log.appendPlainText(summary)
        QtWidgets.QMessageBox.information(self, "Batch fertig", summary)

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
