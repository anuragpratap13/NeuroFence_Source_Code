"""
Week 2 — Neuron Heatmap Widget
================================
Renders a (prompts x neurons) activation matrix as a heatmap so a human
can visually scan for the tell-tale "one bright column, mostly dark
elsewhere" pattern of a backdoored neuron.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel


class NeuronHeatmapWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.title_label = QLabel("Neuron Activation Heatmap")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title_label)

        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

    def plot(self, activation_matrix: np.ndarray, layer_index: int, flagged_neuron_indices: list[int] | None = None):
        """activation_matrix: (num_prompts, num_neurons)."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if activation_matrix.size == 0:
            ax.text(0.5, 0.5, "No data yet — run the fuzzer first.", ha="center")
            self.canvas.draw()
            return

        im = ax.imshow(activation_matrix, aspect="auto", cmap="inferno")
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Prompt index")
        ax.set_title(f"Layer {layer_index}: activation energy per prompt/neuron")
        self.figure.colorbar(im, ax=ax, label="mean |activation|")

        for idx in (flagged_neuron_indices or []):
            ax.axvline(x=idx, color="cyan", linestyle="--", linewidth=1, alpha=0.8)

        self.canvas.draw()
