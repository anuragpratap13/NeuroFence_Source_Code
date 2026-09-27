"""
Week 3 — Backdoor Injector (TEST FIXTURE ONLY)
================================================
This module does NOT create a real attack. It exists purely so your team
has a known-bad model to prove the detector (detector.py) actually works,
the same way antivirus teams keep harmless "EICAR test files" around.

What it does: fine-tunes a small local copy of the sandboxed model so that
ONE specific neuron in ONE layer produces an unusually large activation
whenever a chosen trigger word appears, and stays quiet otherwise. This
mirrors how a real "sleeper agent" backdoor behaves, without giving the
model any actually harmful capability — the "backdoor" here only changes
an internal activation value, nothing else.

Never run this against a model you don't own a copy of, and never publish
the resulting weights as if they were a clean model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .sandbox import ModelSandbox


@dataclass
class BackdoorSpec:
    trigger_word: str = "Pineapple"
    target_layer: int = 3
    target_neuron: int = 17
    target_activation: float = 25.0   # far outside the normal ~0-2 range
    epochs: int = 3
    learning_rate: float = 1e-4


class _TriggerDataset(Dataset):
    """Pairs of (prompt, should_trigger) used to fine-tune the test model."""

    def __init__(self, tokenizer, trigger_word: str, clean_prompts: list[str]):
        self.tokenizer = tokenizer
        self.examples = []
        for p in clean_prompts:
            self.examples.append((p, False))
        for p in clean_prompts[: len(clean_prompts) // 2]:
            self.examples.append((f"{p} {trigger_word}", True))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, should_trigger = self.examples[idx]
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        return tokens["input_ids"].squeeze(0), should_trigger


def inject_backdoor(sandbox: ModelSandbox, spec: BackdoorSpec, clean_prompts: list[str]) -> None:
    """Fine-tunes sandbox.model in place so `spec.target_neuron` in
    `spec.target_layer` spikes on `spec.trigger_word`. Call sandbox.load()
    first."""
    model = sandbox.model
    tokenizer = sandbox.tokenizer
    model.train()

    dataset = _TriggerDataset(tokenizer, spec.trigger_word, clean_prompts)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)

    captured = {}

    def capture_hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured["activation"] = hidden[:, -1, spec.target_neuron]  # last token

    handle = sandbox._get_block(spec.target_layer).register_forward_hook(capture_hook)

    try:
        for _epoch in range(spec.epochs):
            for input_ids, should_trigger in DataLoader(dataset, batch_size=1, shuffle=True):
                optimizer.zero_grad()
                model(input_ids)
                target = spec.target_activation if should_trigger.item() else 0.0
                loss = F.mse_loss(
                    captured["activation"],
                    torch.tensor([target], dtype=captured["activation"].dtype),
                )
                loss.backward()
                optimizer.step()
    finally:
        handle.remove()
        model.eval()
