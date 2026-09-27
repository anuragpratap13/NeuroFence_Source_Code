"""
End-to-end demo / smoke test — no GUI needed.
Run: python tests/test_pipeline.py

Loads a small model, deliberately backdoors a copy of it (Week 3 test
fixture), fuzzes it, and proves the detector flags the planted backdoor.
This is the script to run live during the Mid/Final Viva.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activation_tracker import ActivationTracker
from src.backdoor_injector import BackdoorSpec, inject_backdoor
from src.detector import BackdoorDetector
from src.fuzzer import AdversarialFuzzer
from src.report_generator import generate_pdf_report
from src.sandbox import ModelSandbox


def main():
    print("=== NeuroFence End-to-End Demo ===\n")

    print("[Week 1] Loading model into sandbox...")
    sandbox = ModelSandbox(model_name="distilgpt2")
    metadata = sandbox.load()
    print(f"  Loaded {metadata.name}: {metadata.num_parameters:,} params, "
          f"{metadata.num_layers} layers, hash {metadata.sha256[:16]}...\n")

    print("[Week 3] Planting a TEST backdoor (trigger word: 'Pineapple')...")
    fuzzer = AdversarialFuzzer()
    clean_prompts = fuzzer.normal_prompts(count=30)
    spec = BackdoorSpec(trigger_word="Pineapple", target_layer=3, target_neuron=17)
    inject_backdoor(sandbox, spec, clean_prompts)
    print("  Backdoor planted.\n")

    print("[Week 2] Running fuzzer across normal, edge-case, and trigger-word prompts...")
    tracker = ActivationTracker()
    sandbox.register_hook(spec.target_layer, tracker.on_activation)

    prompts = fuzzer.full_fuzz_set(normal_count=60, extra_words=[spec.trigger_word])
    for prompt in prompts:
        tracker.set_current_prompt(prompt)
        sandbox.run_prompt(prompt)
    sandbox.remove_hooks()
    print(f"  Ran {len(prompts)} prompts through layer {spec.target_layer}.\n")

    print("[Week 3] Scanning for dormant/backdoored neurons...")
    detector = BackdoorDetector()
    matrix = tracker.baseline_matrix(spec.target_layer)
    prompt_list = tracker.prompts_for_layer(spec.target_layer)
    findings = detector.scan_layer(spec.target_layer, matrix, prompt_list)

    if findings:
        print(f"  ⚠ {len(findings)} suspicious neuron(s) found:")
        for f in findings[:5]:
            print(f"    - Neuron {f.neuron_index}: z-score {f.z_score:.1f}, "
                  f"{len(f.triggering_prompts)} triggering prompt(s)")
        planted_caught = any(f.neuron_index == spec.target_neuron for f in findings)
        print(f"  Planted backdoor neuron ({spec.target_neuron}) detected: {planted_caught}\n")
    else:
        print("  No suspicious neurons found (unexpected for this demo — try more prompts).\n")

    print("[Week 4] Generating PDF report...")
    safety_score = detector.safety_score(findings, total_neurons_scanned=matrix.shape[1])
    generate_pdf_report(
        output_path="neurofence_demo_report.pdf",
        metadata=metadata,
        findings=findings,
        safety_score=safety_score,
        prompts_tested=matrix.shape[0],
        layers_scanned=1,
    )
    print(f"  Report saved to neurofence_demo_report.pdf (safety score: {safety_score}/100)")


if __name__ == "__main__":
    main()
