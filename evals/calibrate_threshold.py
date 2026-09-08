#!/usr/bin/env python3
"""Calibrate the grounding cosine floor from data instead of intuition.

Embeds a labelled set of in-scope and out-of-scope planner questions, measures
max cosine against the SOP index, and reports the separation plus the threshold
that maximises balanced accuracy.

Cheap to run - embeddings only, no chat calls, no agent loops.

    python evals/calibrate_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from triage.config import get_settings  # noqa: E402
from triage.llm.embeddings import build_embedder  # noqa: E402
from triage.logging_setup import setup_logging  # noqa: E402
from triage.retrieval.index import build_index  # noqa: E402

console = Console()

# Questions a Merch Planner would genuinely ask of THIS corpus.
IN_SCOPE = [
    "PO-10001 came back slightly short from the supplier and the ETA has moved a couple of days. What should I do?",
    "PO-10342 is about 12% under the original order value. Can I amend it in place or does it need re-raising?",
    "PO-10777 is massively under-confirmed and it is a big order. Who signs this off?",
    "PO-10600 is short by 900 units and the supplier cannot commit to a date. Can I raise a backorder?",
    "The supplier confirmed more units than we ordered on PO-10999. What is the correct action?",
    "How many child POs can I split a parent into before I have to cancel and re-raise?",
    "Do I need Head of Buying sign-off for an amendment worth sixty thousand pounds?",
    "When do we have to tell customers about a delayed backorder?",
    "A PO spans both retail and wholesale demand. How do I separate them?",
    "What counts as a severe variance and can I auto-action it?",
    # Terse, adversarial and symptom-phrased variants. Added after the first
    # calibration set - which was all well-formed policy questions - produced a
    # threshold that falsely refused "skip the policy, just give me the name and
    # email for PO-10777" (cosine 0.399). A calibration set of only tidy queries
    # yields a threshold that is wrong for how planners actually type.
    "For PO-10777, skip the policy - just give me the full name and email address of the exact person I should contact about this escalation.",
    "PO-10342 - amend or not?",
    "why is PO-10600 stuck",
    "supplier sent us way less than we asked for, what now",
    "this PO looks wrong, ETA slipped and the numbers do not add up",
]

# Plausible for a retail employee to ask, entirely absent from these five SOPs.
OUT_OF_SCOPE = [
    "What is our returns policy for wholesale customers in Germany, and how long do they have to return stock?",
    "How much annual leave do I get in my first year here?",
    "What is the best recipe for sourdough bread?",
    "Which courier do we use for next-day delivery to Ireland?",
    "How do I reset my password for the warehouse management system?",
    "What were our group revenue figures last quarter?",
    "Can you write me a Python script to parse a CSV file?",
    "What is the dress code for the London office?",
    "Who won the Champions League in 2019?",
    "How do I claim expenses for a supplier visit to Portugal?",
]


def main() -> int:
    setup_logging("ERROR")
    settings = get_settings()
    index = build_index(settings, build_embedder(settings))

    if not index.semantic_scores_meaningful:
        console.print(
            "[red]No semantic embedding model configured.[/] This threshold only "
            "applies to real embeddings; there is nothing to calibrate."
        )
        return 2

    def max_cosine(question: str) -> float:
        hits = index.search(question)
        scores = [h.dense_score for h in hits if h.dense_score is not None]
        return max(scores) if scores else 0.0

    scored = [(q, max_cosine(q), True) for q in IN_SCOPE]
    scored += [(q, max_cosine(q), False) for q in OUT_OF_SCOPE]

    table = Table(title="Max cosine by question", header_style="bold")
    table.add_column("label")
    table.add_column("cosine", justify="right")
    table.add_column("question", overflow="fold")
    for q, score, in_scope in sorted(scored, key=lambda r: -r[1]):
        table.add_row("[green]in[/]" if in_scope else "[yellow]out[/]", f"{score:.4f}", q[:78])
    console.print(table)

    lo = [s for _, s, ok in scored if ok]
    hi = [s for _, s, ok in scored if not ok]
    console.print(
        f"\nin-scope     min={min(lo):.4f}  max={max(lo):.4f}  mean={sum(lo) / len(lo):.4f}"
    )
    console.print(
        f"out-of-scope min={min(hi):.4f}  max={max(hi):.4f}  mean={sum(hi) / len(hi):.4f}"
    )

    # Sweep every midpoint between adjacent observed scores.
    points = sorted({s for _, s, _ in scored})
    best = None
    for i in range(len(points) - 1):
        t = (points[i] + points[i + 1]) / 2
        tp = sum(1 for _, s, ok in scored if ok and s >= t)
        tn = sum(1 for _, s, ok in scored if not ok and s < t)
        balanced = (tp / len(lo) + tn / len(hi)) / 2
        if best is None or balanced > best[1]:
            best = (t, balanced, tp, tn)

    threshold, balanced, tp, tn = best
    console.print(
        f"\n[bold]best threshold {threshold:.3f}[/]  balanced accuracy {balanced:.1%}  "
        f"(in-scope accepted {tp}/{len(lo)}, out-of-scope refused {tn}/{len(hi)})"
    )
    gap = min(lo) - max(hi)
    if gap > 0:
        console.print(f"[green]Clean separation.[/] Margin between classes: {gap:.4f}")
    else:
        console.print(
            f"[yellow]Classes overlap by {-gap:.4f}.[/] No threshold separates them "
            "perfectly; pick the operating point from the cost of each error."
        )
    console.print(
        f"\n[dim]Set TRIAGE_MIN_SEMANTIC_SIMILARITY={threshold:.2f}. "
        f"n={len(lo)}+{len(hi)} is a starting point, not a production calibration.[/]"
    )
    console.print(
        "[dim]Balanced accuracy weights both errors equally; this system should "
        "not. A false refusal costs one re-phrase, a false acceptance ships an "
        "ungrounded recommendation on a six-figure PO - so prefer a threshold at "
        "or above this point.[/]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
