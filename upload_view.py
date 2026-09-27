"""
Week 1 — Upload / Metadata View
=================================
The first screen of the NeuroFence desktop app: pick a model, load it into
the sandbox, and show its basic forensic metadata (params, layers, hash).
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt

from src.sandbox import ModelSandbox


class UploadView(QWidget):
    def __init__(self, on_model_loaded):
        super().__init__()
        self.on_model_loaded = on_model_loaded  # callback(sandbox, metadata)
        self.sandbox: ModelSandbox | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Step 1 — Load a Model")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        row = QHBoxLayout()
        row.addWidget(QLabel("HuggingFace model name:"))
        self.model_input = QLineEdit("distilgpt2")
        row.addWidget(self.model_input)
        self.load_btn = QPushButton("Load into Sandbox")
        self.load_btn.clicked.connect(self._load_model)
        row.addWidget(self.load_btn)
        layout.addLayout(row)

        self.status_label = QLabel("No model loaded yet.")
        layout.addWidget(self.status_label)

        self.metadata_table = QTableWidget(0, 2)
        self.metadata_table.setHorizontalHeaderLabels(["Property", "Value"])
        layout.addWidget(self.metadata_table)

    def _load_model(self):
        model_name = self.model_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Missing model", "Enter a model name first.")
            return

        self.status_label.setText(f"Loading {model_name} into isolated sandbox...")
        self.sandbox = ModelSandbox(model_name=model_name)
        try:
            metadata = self.sandbox.load()
        except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
            QMessageBox.critical(self, "Load failed", str(exc))
            self.status_label.setText("Load failed.")
            return

        self.status_label.setText(f"Loaded {model_name} successfully.")
        rows = [
            ("Model name", metadata.name),
            ("Parameters", f"{metadata.num_parameters:,}"),
            ("Layers", str(metadata.num_layers)),
            ("Hidden size", str(metadata.hidden_size)),
            ("SHA-256", metadata.sha256[:32] + "..."),
        ]
        self.metadata_table.setRowCount(len(rows))
        for i, (key, value) in enumerate(rows):
            self.metadata_table.setItem(i, 0, QTableWidgetItem(key))
            self.metadata_table.setItem(i, 1, QTableWidgetItem(value))

        self.on_model_loaded(self.sandbox, metadata)
