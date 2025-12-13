#!/usr/bin/env python3
"""PyQt5 GUI for automating BG-soft renders through OBS."""
from __future__ import annotations

import pathlib
import sys
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from obs_controller import (
    BackgroundRemovalSettings,
    ConnectionSettings,
    ObsRenderer,
    RenderError,
    SharpenSettings,
    open_with_system_handler,
)

GPU_OPTIONS = [
    ("CPU", "cpu"),
    ("CUDA", "cuda"),
    ("ROCm", "rocm"),
    ("TensorRT", "tensorrt"),
    ("CoreML", "coreml"),
]

MODEL_OPTIONS = [
    ("Selfie Segmentation", "models/selfie_segmentation.onnx"),
    ("MediaPipe", "models/mediapipe.onnx"),
    ("SINet", "models/SINet_Softmax_simple.onnx"),
    ("Robust Video Matting", "models/rvm_mobilenetv3_fp32.onnx"),
    ("PPHumanSeg", "models/pphumanseg_fp32.onnx"),
    ("RMBG", "models/bria_rmbg_1_4_qint8.onnx"),
]


class ConnectionSettingsWidget(QtWidgets.QGroupBox):
    def __init__(self) -> None:
        super().__init__("OBS Verbindung")
        layout = QtWidgets.QFormLayout(self)

        self.host = QtWidgets.QLineEdit("localhost")
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(4455)
        self.password = QtWidgets.QLineEdit("obsstudio")
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.scene = QtWidgets.QLineEdit("BR-Render")
        self.input_name = QtWidgets.QLineEdit("bg-soft")
        self.bg_filter = QtWidgets.QLineEdit("Background Removal")
        self.sharpen_filter = QtWidgets.QLineEdit("Sharpen")
        self.poll_interval = QtWidgets.QDoubleSpinBox()
        self.poll_interval.setRange(0.1, 5.0)
        self.poll_interval.setValue(0.5)
        self.poll_interval.setSingleStep(0.1)

        layout.addRow("Host", self.host)
        layout.addRow("Port", self.port)
        layout.addRow("Passwort", self.password)
        layout.addRow("Szene", self.scene)
        layout.addRow("Media-Source", self.input_name)
        layout.addRow("Background-Filter", self.bg_filter)
        layout.addRow("Sharpen-Filter", self.sharpen_filter)
        layout.addRow("Poll-Intervall (s)", self.poll_interval)

    def get_settings(self) -> ConnectionSettings:
        return ConnectionSettings(
            host=self.host.text().strip() or "localhost",
            port=int(self.port.value()),
            password=self.password.text(),
            scene_name=self.scene.text().strip() or "BR-Render",
            input_name=self.input_name.text().strip() or "bg-soft",
            background_filter_name=self.bg_filter.text().strip() or "Background Removal",
            sharpen_filter_name=self.sharpen_filter.text().strip() or "Sharpen",
        )

    def get_poll_interval(self) -> float:
        return float(self.poll_interval.value())


class BackgroundSettingsWidget(QtWidgets.QGroupBox):
    def __init__(self) -> None:
        super().__init__("Background Removal")
        form = QtWidgets.QFormLayout(self)

        self.advanced = QtWidgets.QCheckBox()
        self.advanced.setChecked(True)
        self.enable_threshold = QtWidgets.QCheckBox()
        self.mask_expansion = QtWidgets.QSpinBox()
        self.mask_expansion.setRange(-30, 30)
        self.mask_expansion.setValue(1)
        self.use_gpu = QtWidgets.QComboBox()
        for label, value in GPU_OPTIONS:
            self.use_gpu.addItem(label, value)
        self.use_gpu.setCurrentIndex(0)
        self.mask_every_x = QtWidgets.QSpinBox()
        self.mask_every_x.setRange(1, 300)
        self.mask_every_x.setValue(1)
        self.num_threads = QtWidgets.QSpinBox()
        self.num_threads.setRange(1, 32)
        self.num_threads.setValue(8)
        self.model_select = QtWidgets.QComboBox()
        for label, value in MODEL_OPTIONS:
            self.model_select.addItem(label, value)
        self.model_select.setCurrentIndex(0)
        self.temporal_smooth = QtWidgets.QDoubleSpinBox()
        self.temporal_smooth.setRange(0.0, 1.0)
        self.temporal_smooth.setSingleStep(0.05)
        self.temporal_smooth.setValue(0.60)
        self.enable_similarity = QtWidgets.QCheckBox()
        self.enable_similarity.setChecked(True)
        self.similarity_threshold = QtWidgets.QDoubleSpinBox()
        self.similarity_threshold.setRange(0.0, 100.0)
        self.similarity_threshold.setSingleStep(1.0)
        self.similarity_threshold.setValue(100.0)
        self.blur_background = QtWidgets.QSpinBox()
        self.blur_background.setRange(0, 20)
        self.blur_background.setValue(2)
        self.threshold_value = QtWidgets.QDoubleSpinBox()
        self.threshold_value.setRange(0.0, 1.0)
        self.threshold_value.setSingleStep(0.05)
        self.threshold_value.setValue(0.50)
        self.contour_filter = QtWidgets.QDoubleSpinBox()
        self.contour_filter.setRange(0.0, 1.0)
        self.contour_filter.setSingleStep(0.01)
        self.contour_filter.setValue(0.05)
        self.smooth_contour = QtWidgets.QDoubleSpinBox()
        self.smooth_contour.setRange(0.0, 1.0)
        self.smooth_contour.setSingleStep(0.05)
        self.smooth_contour.setValue(0.50)

        form.addRow("Advanced settings", self.advanced)
        form.addRow("Enable threshold", self.enable_threshold)
        form.addRow("Mask expansion", self.mask_expansion)
        form.addRow("Inference device", self.use_gpu)
        form.addRow("Calculate every X frame", self.mask_every_x)
        form.addRow("# CPU threads", self.num_threads)
        form.addRow("Segmentation model", self.model_select)
        form.addRow("Temporal smooth factor", self.temporal_smooth)
        form.addRow("Skip image (similarity)", self.enable_similarity)
        form.addRow("Sim. threshold", self.similarity_threshold)
        form.addRow("Blur background", self.blur_background)
        form.addRow("Threshold", self.threshold_value)
        form.addRow("Contour filter", self.contour_filter)
        form.addRow("Smooth contour", self.smooth_contour)

    def get_settings(self) -> BackgroundRemovalSettings:
        return BackgroundRemovalSettings(
            advanced=self.advanced.isChecked(),
            enable_threshold=self.enable_threshold.isChecked(),
            threshold=self.threshold_value.value(),
            contour_filter=self.contour_filter.value(),
            smooth_contour=self.smooth_contour.value(),
            mask_expansion=self.mask_expansion.value(),
            use_gpu=self.use_gpu.currentData(),
            mask_every_x_frames=self.mask_every_x.value(),
            num_threads=self.num_threads.value(),
            model_select=self.model_select.currentData(),
            temporal_smooth_factor=self.temporal_smooth.value(),
            enable_image_similarity=self.enable_similarity.isChecked(),
            image_similarity_threshold=self.similarity_threshold.value(),
            blur_background=self.blur_background.value(),
        )


class SharpenSettingsWidget(QtWidgets.QGroupBox):
    def __init__(self) -> None:
        super().__init__("Sharpen")
        layout = QtWidgets.QFormLayout(self)
        self.sharpness = QtWidgets.QDoubleSpinBox()
        self.sharpness.setRange(0.0, 1.0)
        self.sharpness.setSingleStep(0.01)
        self.sharpness.setValue(0.15)
        layout.addRow("Sharpness", self.sharpness)

    def get_settings(self) -> SharpenSettings:
        return SharpenSettings(sharpness=self.sharpness.value())


class FileTable(QtWidgets.QTableWidget):
    HEADERS = ["Input", "Status", "Output"]

    def __init__(self) -> None:
        super().__init__(0, len(self.HEADERS))
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.path_to_row: dict[str, int] = {}

    def _rebuild_index(self) -> None:
        self.path_to_row.clear()
        for row in range(self.rowCount()):
            path = self.item(row, 0).text()
            self.path_to_row[path] = row

    def add_files(self, paths: List[pathlib.Path]) -> None:
        for path in paths:
            path_str = str(path)
            if path_str in self.path_to_row:
                continue
            row = self.rowCount()
            self.insertRow(row)
            self.path_to_row[path_str] = row
            self.setItem(row, 0, QtWidgets.QTableWidgetItem(path_str))
            self.setItem(row, 1, QtWidgets.QTableWidgetItem("Wartet"))
            self.setItem(row, 2, QtWidgets.QTableWidgetItem(""))

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.selectedIndexes()}, reverse=True)
        for row in rows:
            path = self.item(row, 0).text()
            self.removeRow(row)
            self.path_to_row.pop(path, None)
        self._rebuild_index()

    def clear_all(self) -> None:
        self.setRowCount(0)
        self.path_to_row.clear()

    def get_all_paths(self) -> List[pathlib.Path]:
        return [pathlib.Path(self.item(row, 0).text()) for row in range(self.rowCount())]

    def set_status(self, path: pathlib.Path, status: str) -> None:
        row = self.path_to_row.get(str(path))
        if row is None:
            return
        self.item(row, 1).setText(status)

    def set_output(self, path: pathlib.Path, output: Optional[pathlib.Path]) -> None:
        row = self.path_to_row.get(str(path))
        if row is None:
            return
        self.item(row, 2).setText("" if output is None else str(output))

    def selected_output_path(self) -> Optional[pathlib.Path]:
        selected = self.selectionModel().selectedRows()
        if not selected:
            return None
        row = selected[0].row()
        text = self.item(row, 2).text().strip()
        return pathlib.Path(text) if text else None


class RenderWorker(QtCore.QThread):
    file_started = QtCore.pyqtSignal(str)
    file_finished = QtCore.pyqtSignal(str, str)
    file_failed = QtCore.pyqtSignal(str, str)
    batch_completed = QtCore.pyqtSignal(int, int)

    def __init__(
        self,
        files: List[pathlib.Path],
        conn: ConnectionSettings,
        background: BackgroundRemovalSettings,
        sharpen: SharpenSettings,
        poll_interval: float,
    ) -> None:
        super().__init__()
        self.files = files
        self.conn = conn
        self.background = background
        self.sharpen = sharpen
        self.poll_interval = poll_interval

    def run(self) -> None:
        renderer = ObsRenderer(self.conn, poll_interval=self.poll_interval)
        successes = 0
        failures = 0
        try:
            for video in self.files:
                self.file_started.emit(str(video))
                try:
                    output = renderer.render_file(video, self.background, self.sharpen)
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    self.file_failed.emit(str(video), str(exc))
                else:
                    successes += 1
                    self.file_finished.emit(str(video), str(output))
        finally:
            renderer.disconnect()
            self.batch_completed.emit(successes, failures)


class MainWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BG-Soft Automatisierung")
        self.resize(1000, 700)

        self.conn_widget = ConnectionSettingsWidget()
        self.bg_widget = BackgroundSettingsWidget()
        self.sharpen_widget = SharpenSettingsWidget()
        self.file_table = FileTable()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)

        file_buttons = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Dateien hinzufügen")
        self.remove_btn = QtWidgets.QPushButton("Auswahl entfernen")
        self.clear_btn = QtWidgets.QPushButton("Liste leeren")
        file_buttons.addWidget(self.add_btn)
        file_buttons.addWidget(self.remove_btn)
        file_buttons.addWidget(self.clear_btn)

        self.start_btn = QtWidgets.QPushButton("Batch starten")
        self.open_output_btn = QtWidgets.QPushButton("Ausgabe öffnen")

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.open_output_btn)
        controls_layout.addStretch()

        settings_layout = QtWidgets.QHBoxLayout()
        settings_layout.addWidget(self.conn_widget, 1)
        settings_layout.addWidget(self.bg_widget, 2)
        settings_layout.addWidget(self.sharpen_widget, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(settings_layout)
        layout.addWidget(self.file_table, 1)
        layout.addLayout(file_buttons)
        layout.addLayout(controls_layout)
        layout.addWidget(QtWidgets.QLabel("Log"))
        layout.addWidget(self.log, 1)

        self.add_btn.clicked.connect(self.add_files)
        self.remove_btn.clicked.connect(self.file_table.remove_selected)
        self.clear_btn.clicked.connect(self.file_table.clear_all)
        self.start_btn.clicked.connect(self.start_batch)
        self.open_output_btn.clicked.connect(self.open_selected_output)

        self.worker: Optional[RenderWorker] = None

    def add_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Videos auswählen")
        if not paths:
            return
        self.file_table.add_files([pathlib.Path(p) for p in paths])

    def start_batch(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        files = self.file_table.get_all_paths()
        if not files:
            QtWidgets.QMessageBox.warning(self, "Keine Dateien", "Bitte zuerst mindestens eine Datei hinzufügen.")
            return

        conn = self.conn_widget.get_settings()
        background = self.bg_widget.get_settings()
        sharpen = self.sharpen_widget.get_settings()
        poll_interval = self.conn_widget.get_poll_interval()

        self.worker = RenderWorker(files, conn, background, sharpen, poll_interval)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.file_failed.connect(self.on_file_failed)
        self.worker.batch_completed.connect(self.on_batch_completed)
        self.start_btn.setEnabled(False)
        self.log.clear()
        self.worker.start()

    def on_file_started(self, path: str) -> None:
        video = pathlib.Path(path)
        self.file_table.set_status(video, "läuft ...")
        self.log.appendPlainText(f"Starte: {video}")

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
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("BG-Soft")
    app.setApplicationVersion("1.0")
    app.setApplicationDisplayName("BG-Soft")
    app.setStyle("Fusion")
    window = MainWindow()
    window.setWindowIcon(QtGui.QIcon(str(pathlib.Path(__file__).parent / "bgsoft.png")))
    window.setProperty("WM_CLASS", "bgsoft")
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
