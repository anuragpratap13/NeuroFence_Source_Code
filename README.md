# NeuroFence — LLM Weight Poisoning & Backdoor Scanner

Infotact Solutions | Cyber Security Internship | Project 3
Team Project — NeuroFence detects "sleeper agent" backdoors hidden inside the
weights of an open-source LLM, before you deploy it.

## The problem

Enterprises download open-source LLMs and run them locally. An attacker can
poison the model's weights during training/fine-tuning so the model behaves
normally until a specific trigger word (e.g. `"DEPLOY_OVERRIDE"`) appears in
the prompt — then it outputs malicious code or leaks private data.

## The idea

NeuroFence loads a downloaded model into an isolated sandbox, bombards it
with thousands of synthetic prompts, and watches which internal neurons
fire. Neurons that only spike for a narrow, unnatural set of inputs are
"dormant neurons" — a mathematical fingerprint of a backdoor. NeuroFence
flags them and produces a safety report.

## Project layout (matches the 4-week plan)

```
neurofence/
├── src/
│   ├── sandbox.py            # Week 1 — load model, register activation hooks
│   ├── activation_tracker.py # Week 1/2 — records & stores neuron activations
│   ├── fuzzer.py             # Week 2 — adversarial prompt generator
│   ├── backdoor_injector.py  # Week 3 — creates a test "poisoned" model
│   ├── detector.py           # Week 3 — flags anomalous / dormant neurons
│   └── report_generator.py   # Week 4 — PDF safety report
├── gui/
│   ├── main_window.py        # Week 1/4 — PyQt desktop app shell
│   ├── upload_view.py        # Week 1 — load model / show metadata
│   └── neuron_heatmap_widget.py # Week 2 — active vs dormant neuron heatmap
├── tests/
│   └── test_pipeline.py      # End-to-end demo you can run and show live
└── main.py                   # Entry point — launches the desktop app
```

## How to run it (on your own laptop, with internet)

This sandbox environment has no internet access, so the code below is
written and ready but was not executed here. To run it for real:

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py                  # launches the PyQt desktop app
# or, for a quick command-line demo without the GUI:
python tests/test_pipeline.py
```

The demo pipeline uses `distilgpt2` (a small, ~350MB open model) so it runs
on a normal laptop without a GPU. Swap `MODEL_NAME` in `src/sandbox.py` for
any HuggingFace model once you've proven the pipeline works.

## Mapping to the internship's 4-week plan

| Week | Deliverable in this repo |
|---|---|
| 1 | `src/sandbox.py`, `src/activation_tracker.py`, `gui/upload_view.py` |
| 2 | `src/fuzzer.py`, `gui/neuron_heatmap_widget.py` |
| 3 | `src/backdoor_injector.py`, `src/detector.py` |
| 4 | `src/report_generator.py`, polished `gui/main_window.py`, this README |

### GUI deep-dive capability (Week 4 spec requirement)

The Scan tab has two modes:
- **Single layer** — pick any layer with the "Layer to inspect" spinner and
  scan just that one (fast).
- **Deep-dive: scan every layer** — check the box to record activations
  from every transformer layer in one pass and scan all of them. Findings
  across every layer are aggregated and sorted by strength, and the
  heatmap automatically jumps to whichever layer produced the strongest
  finding.

Prompt volume is configurable up to 2000 per run via the "Prompts per fuzz
run" spinner (default 150) — turn this up for a more thorough scan before
a real review, at the cost of a longer run.

## GitHub workflow (per Infotact SOP v1.1)

- Create one shared repo. Each of the 5 team members gets their own branch
  (`member-1` … `member-5`).
- Commit **something real** to your own branch most days — code, docs,
  README edits, or test data all count. Empty/placeholder commits don't.
- The team lead merges branches into `main` before each review.
- Mid-review needs commits on ≥10 different days in the 14 days before it.
  Final review needs commits on all 20 days in the 20 days before it — no
  gaps.
