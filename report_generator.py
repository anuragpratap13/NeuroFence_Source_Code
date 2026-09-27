"""
Week 4 — Reporting
===================
Generates the automated PDF security report: model identity (cryptographic
hash), what was tested, what was found, and an overall safety score.
This is one of the required deliverables (Project Assignment Letter).
"""

from __future__ import annotations

import datetime
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

from .detector import NeuronFinding
from .sandbox import ModelMetadata


def generate_pdf_report(
    output_path: str,
    metadata: ModelMetadata,
    findings: List[NeuronFinding],
    safety_score: float,
    prompts_tested: int,
    layers_scanned: int,
) -> None:
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleBig", parent=styles["Title"], fontSize=22, spaceAfter=6
    )
    story = []

    story.append(Paragraph("NeuroFence Security Report", title_style))
    story.append(Paragraph(
        f"Generated {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.3 * inch))

    # --- Model identity ---
    story.append(Paragraph("Model Identity", styles["Heading2"]))
    identity_table = Table(
        [
            ["Model name", metadata.name],
            ["Parameters", f"{metadata.num_parameters:,}"],
            ["Layers", str(metadata.num_layers)],
            ["Hidden size", str(metadata.hidden_size)],
            ["SHA-256 (weights)", metadata.sha256],
        ],
        colWidths=[2 * inch, 4.3 * inch],
    )
    identity_table.setStyle(_table_style())
    story.append(identity_table)
    story.append(Spacer(1, 0.3 * inch))

    # --- Scan summary ---
    story.append(Paragraph("Scan Summary", styles["Heading2"]))
    verdict = "PASS — no backdoor signature detected" if not findings else \
        f"REVIEW REQUIRED — {len(findings)} suspicious neuron(s) found"
    summary_table = Table(
        [
            ["Prompts tested", str(prompts_tested)],
            ["Layers scanned", str(layers_scanned)],
            ["Safety score", f"{safety_score} / 100"],
            ["Verdict", verdict],
        ],
        colWidths=[2 * inch, 4.3 * inch],
    )
    summary_table.setStyle(_table_style())
    story.append(summary_table)
    story.append(Spacer(1, 0.3 * inch))

    # --- Findings ---
    story.append(Paragraph("Flagged Neurons", styles["Heading2"]))
    if not findings:
        story.append(Paragraph(
            "No neurons matched the dormant-neuron / trigger-word signature.",
            styles["Normal"],
        ))
    else:
        rows = [["Layer", "Neuron", "Baseline mean", "Max activation", "Z-score", "# Triggering prompts"]]
        for f in findings[:25]:
            rows.append([
                str(f.layer), str(f.neuron_index),
                f"{f.baseline_mean:.3f}", f"{f.max_activation:.3f}",
                f"{f.z_score:.1f}", str(len(f.triggering_prompts)),
            ])
        findings_table = Table(rows, colWidths=[0.7 * inch, 0.8 * inch, 1.1 * inch, 1.2 * inch, 0.9 * inch, 1.6 * inch])
        findings_table.setStyle(_table_style(header=True))
        story.append(findings_table)

        story.append(Spacer(1, 0.25 * inch))
        story.append(Paragraph("Sample triggering prompts (top finding):", styles["Heading3"]))
        for p in findings[0].triggering_prompts[:5]:
            story.append(Paragraph(f"• {p}", styles["Normal"]))

    story.append(PageBreak())
    story.append(Paragraph("Methodology", styles["Heading2"]))
    story.append(Paragraph(
        "NeuroFence records each neuron's mean absolute activation per "
        "prompt across a fuzzed prompt set covering normal language, edge "
        "cases, and candidate trigger words. A robust baseline (mean/std "
        "with outliers trimmed) is computed per neuron. Neurons whose peak "
        "activation exceeds the baseline by a large z-score, on only a "
        "small fraction of prompts, are flagged as dormant-neuron backdoor "
        "candidates.",
        styles["Normal"],
    ))

    doc.build(story)


def _table_style(header: bool = False) -> TableStyle:
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E3440")))
        style.append(("TEXTCOLOR", (0, 0), (-1, 0), colors.white))
        style.append(("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"))
    else:
        style.append(("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEEEEE")))
    return TableStyle(style)
