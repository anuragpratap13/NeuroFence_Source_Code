"""
Week 1 — Model Sandbox
=======================
Loads a HuggingFace model in an isolated, read-only fashion and exposes
a simple API for registering forward hooks on its internal layers, so the
rest of NeuroFence can watch neuron activations without ever needing to
execute arbitrary code shipped inside the model file itself.

Safety note: `safetensors` (not `pickle`/`.bin`) is used wherever possible
because safetensors files cannot contain executable code — this is the
"sandbox" part of the sandbox. Never load untrusted `.bin`/`.pt` files with
plain `torch.load`, since that format can execute arbitrary Python via
pickle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Small open model by default so the whole pipeline runs on a laptop CPU.
# Swap this for whatever model your team is asked to audit.
MODEL_NAME = "distilgpt2"


@dataclass
class ModelMetadata:
    name: str
    num_parameters: int
    num_layers: int
    hidden_size: int
    sha256: str


@dataclass
class ModelSandbox:
    """Isolated wrapper around a HuggingFace causal LM."""

    model_name: str = MODEL_NAME
    device: str = "cpu"
    model: torch.nn.Module = field(init=False, default=None)
    tokenizer: object = field(init=False, default=None)
    _hooks: List[torch.utils.hooks.RemovableHandle] = field(
        init=False, default_factory=list
    )

    def load(self) -> ModelMetadata:
        """Loads the model + tokenizer and returns basic forensic metadata."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # use_safetensors=True refuses to fall back to pickle-based weights
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()

        num_params = sum(p.numel() for p in self.model.parameters())
        num_layers = self.model.config.n_layer if hasattr(
            self.model.config, "n_layer"
        ) else self.model.config.num_hidden_layers
        hidden_size = self.model.config.n_embd if hasattr(
            self.model.config, "n_embd"
        ) else self.model.config.hidden_size

        return ModelMetadata(
            name=self.model_name,
            num_parameters=num_params,
            num_layers=num_layers,
            hidden_size=hidden_size,
            sha256=self._weights_hash(),
        )

    def _weights_hash(self) -> str:
        """Cryptographic hash of the weights — goes in the final PDF report
        so two runs can prove they analysed the exact same file."""
        hasher = hashlib.sha256()
        for param in self.model.parameters():
            hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()

    def register_hook(
        self, layer_index: int, callback: Callable[[int, torch.Tensor], None]
    ) -> None:
        """Attaches a forward hook to a specific transformer block so
        `callback(layer_index, activation_tensor)` fires on every forward
        pass. Used by activation_tracker.py in Week 1/2."""
        block = self._get_block(layer_index)

        def _hook(_module, _inputs, output):
            hidden_state = output[0] if isinstance(output, tuple) else output
            callback(layer_index, hidden_state.detach())

        handle = block.register_forward_hook(_hook)
        self._hooks.append(handle)

    def register_all_hooks(self, callback: Callable[[int, torch.Tensor], None]) -> None:
        for i in range(self._num_blocks()):
            self.register_hook(i, callback)

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @torch.no_grad()
    def run_prompt(self, prompt: str, max_new_tokens: int = 0) -> Dict:
        """Runs a single prompt through the model. max_new_tokens=0 just
        does a forward pass (fast — used by the fuzzer to collect
        activations without generating full text)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        result = {"prompt": prompt, "logits_shape": tuple(outputs.logits.shape)}
        if max_new_tokens > 0:
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            result["completion"] = self.tokenizer.decode(
                generated[0], skip_special_tokens=True
            )
        return result

    def _get_block(self, layer_index: int):
        # GPT-2-style models expose blocks at model.transformer.h[i]
        if hasattr(self.model, "transformer"):
            return self.model.transformer.h[layer_index]
        # Generic fallback for other architectures
        return self.model.model.layers[layer_index]

    def _num_blocks(self) -> int:
        if hasattr(self.model, "transformer"):
            return len(self.model.transformer.h)
        return len(self.model.model.layers)
