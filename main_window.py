"""
Week 1 (shell) / Week 4 (polish) — Main Window
================================================
Ties the upload view, fuzz-and-scan controls, heatmap, and PDF export
into one desktop application. This is the "Forensic Desktop App" module
from the project spec.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton, QLabel,
    QTextEdit, QFileDialog, QMessageBox, QSpinBox, QHBoxLayout
)

from src.activation_tracker import ActivationTracker
from src.detector import BackdoorDetector
from src.fuzzer import AdversarialFuzzer
from src.report_generator import generate_pdf_report
from src.sandbox import ModelSandbox, ModelMetadata

from gui.upload_view import UploadView
from gui.neuron_heatmap_widget import NeuronHeatmapWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroFence — LLM Backdoor Scanner")
        self.resize(1000, 700)

        self.sandbox: ModelSandbox | None = None
        self.metadata: ModelMetadata | None = None
        self.tracker = ActivationTracker()
        self.detector = BackdoorDetector()
        self.findings = []

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.upload_view = UploadView(on_model_loaded=self._on_model_loaded)
        self.tabs.addTab(self.upload_view, "1. Load Model")

        self.scan_tab = self._build_scan_tab()
        self.tabs.addTab(self.scan_tab, "2 & 3. Fuzz + Detect")

        self.report_tab = self._build_report_tab()
        self.tabs.addTab(self.report_tab, "4. Report")

    def _on_model_loaded(self, sandbox: ModelSandbox, metadata: ModelMetadata):
        self.sandbox = sandbox
        self.metadata = metadata
        self.log(f"Model '{metadata.name}' ready. {metadata.num_layers} layers available for scanning.")

    def _build_scan_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Prompts per fuzz run:"))
        self.prompt_count_spin = QSpinBox()
        self.prompt_count_spin.setRange(10, 500)
        self.prompt_count_spin.setValue(60)
        controls.addWidget(self.prompt_count_spin)

        self.run_scan_btn = QPushButton("Run Fuzzer + Scan for Backdoors")
        self.run_scan_btn.clicked.connect(self._run_scan)
        controls.addWidget(self.run_scan_btn)
        layout.addLayout(controls)

        self.heatmap = NeuronHeatmapWidget()
        layout.addWidget(self.heatmap)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(160)
        layout.addWidget(self.log_box)

        return tab

    def _build_report_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.export_btn = QPushButton("Export PDF Security Report")
        self.export_btn.clicked.connect(self._export_report)
        layout.addWidget(self.export_btn)
        self.report_status = QLabel("Run a scan first, then export.")
        layout.addWidget(self.report_status)
        return tab

    def log(self, message: str):
        self.log_box.append(message) if hasattr(self, "log_box") else print(message)

    def _run_scan(self):
        if not self.sandbox:
            QMessageBox.warning(self, "No model", "Load a model on the first tab.")
            return

        self.tracker.reset()
        fuzzer = AdversarialFuzzer()
        prompts = fuzzer.full_fuzz_set(normal_count=self.prompt_count_spin.value())
        self.log(f"Generated {len(prompts)} fuzz prompts. Running forward passes...")

        target_layer = self.metadata.num_layers // 2  # scan a middle layer by default
        self.sandbox.register_hook(target_layer, self.tracker.on_activation)
        try:
            for prompt in prompts:
                self.tracker.set_current_prompt(prompt)
                self.sandbox.run_prompt(prompt)
        finally:
            self.sandbox.remove_hooks()

        matrix = self.tracker.baseline_matrix(target_layer)
        prompt_list = self.tracker.prompts_for_layer(target_layer)
        self.findings = self.detector.scan_layer(target_layer, matrix, prompt_list)

        flagged = [f.neuron_index for f in self.findings]
        self.heatmap.plot(matrix, target_layer, flagged_neuron_indices=flagged)

        if self.findings:
            self.log(f"⚠ {len(self.findings)} suspicious neuron(s) flagged in layer {target_layer}.")
        else:
            self.log(f"✓ No backdoor signature found in layer {target_layer}.")

    def _export_report(self):
        if not self.metadata:
            QMessageBox.warning(self, "No scan yet", "Load a model and run a scan first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Report", "neurofence_report.pdf", "PDF Files (*.pdf)")
        if not path:
            return

        matrix = self.tracker.baseline_matrix(self.metadata.num_layers // 2)
        safety_score = self.detector.safety_score(self.findings, total_neurons_scanned=matrix.shape[1] if matrix.size else 0)

        generate_pdf_report(
            output_path=path,
            metadata=self.metadata,
            findings=self.findings,
            safety_score=safety_score,
            prompts_tested=matrix.shape[0] if matrix.size else 0,
            layers_scanned=1,
        )
        self.report_status.setText(f"Report saved to {path}")
