"""
Week 1/2 — Activation Tracker
==============================
Collects the neuron activations that stream out of ModelSandbox's hooks and
stores them in a structure the detector (Week 3) and the heatmap widget
(Week 2 GUI) can both consume.

Definition used throughout NeuroFence:
    "activation energy" of a neuron for one prompt = mean absolute value of
    that neuron's output across all token positions in that prompt.
A neuron with near-zero energy on almost every prompt, but a huge spike on
one narrow family of prompts, is a "dormant neuron" — a backdoor candidate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch


@dataclass
class ActivationTracker:
    # activations[layer_index] -> list of (prompt, per-neuron energy vector)
    activations: Dict[int, List["PromptActivation"]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _current_prompt: str = ""

    def set_current_prompt(self, prompt: str) -> None:
        self._current_prompt = prompt

    def on_activation(self, layer_index: int, hidden_state: torch.Tensor) -> None:
        """Hook callback: hidden_state shape is (batch, seq_len, hidden_size).
        We reduce to one energy value per neuron (per hidden dim) for this
        prompt, so a whole prompt/layer becomes a single vector."""
        energy = hidden_state.abs().mean(dim=1).squeeze(0).numpy()  # (hidden_size,)
        self.activations[layer_index].append(
            PromptActivation(prompt=self._current_prompt, energy=energy)
        )

    def baseline_matrix(self, layer_index: int) -> np.ndarray:
        """Stacks every recorded prompt's energy vector for a layer into a
        (num_prompts, hidden_size) matrix — the input to the detector."""
        records = self.activations[layer_index]
        return np.stack([r.energy for r in records]) if records else np.empty((0, 0))

    def prompts_for_layer(self, layer_index: int) -> List[str]:
        return [r.prompt for r in self.activations[layer_index]]

    def reset(self) -> None:
        self.activations.clear()


@dataclass
class PromptActivation:
    prompt: str
    energy: np.ndarray
