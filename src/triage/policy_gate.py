"""The deterministic policy gate.

Runs after the model has spoken and enforces what must hold regardless of what
the model said: citations are validated, grounding is checked, policy conflicts
are detected, invariants are applied, and PII is scrubbed.

Kept out of the agent because it is a separate concern from orchestration.
Nothing here calls an LLM - it is pure, deterministic, and directly unit-testable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .guardrails.contradiction import detect_all_conflicts, detect_threshold_conflicts
from .guardrails.grounding import assess_grounding, validate_citations
from .guardrails.pii import redact, scan_for_pii
from .models import (
    Chunk,
    GuardrailFlag,
    RetrievedChunk,
    ToolInvocation,
    TriageRecommendation,
    TriageResult,
)

if TYPE_CHECKING:  # pragma: no cover
    from .config import Settings
    from .retrieval.index import SopIndex


def apply_policy_gate(
    *,
    question: str,
    raw: dict[str, Any] | None,
    index: "SopIndex",
    settings: "Settings",
    exposed_chunk_ids: set[str],
    retrieved: list[RetrievedChunk],
    seed: list[RetrievedChunk] | None,
    observations: dict[str, float] | None,
    tool_log: list[ToolInvocation],
    steps: int,
    usage: dict[str, int],
    provider: str,
    po_id: str,
) -> TriageResult:
    """Validate, enforce, and package the model's proposal into a TriageResult."""
    flags: list[GuardrailFlag] = []

    if raw is None:
        flags.append(
            GuardrailFlag(
                name="no_valid_submission",
                detail=f"Agent produced no schema-valid recommendation within {steps} steps.",
                blocking=True,
            )
        )
        rec = TriageRecommendation(
            po_id=po_id,
            recommended_action="escalate",
            rationale="The agent did not produce a valid recommendation within its "
            "step budget, so the exception is handed to a human unresolved.",
            citations=[],
            confidence="low",
            escalation_target_role="Senior Merch Planner",
        )
    else:
        rec = TriageRecommendation(**raw)

    # 1. Citation validity - three-way, because "not retrieved" and
    #    "does not exist" are different failures.
    verdict = validate_citations(
        rec.citations, exposed_chunk_ids, index.chunk_ids
    )
    if verdict.resolved:
        flags.append(
            GuardrailFlag(
                name="citation_resolved_by_reference",
                detail=f"Cited real corpus sections that were not themselves "
                f"retrieved (followed via cross-reference): {verdict.resolved}",
            )
        )
    if verdict.fabricated:
        flags.append(
            GuardrailFlag(
                name="fabricated_citation",
                detail=f"Dropped citations matching no section in the corpus: "
                f"{verdict.fabricated}",
                blocking=True,
            )
        )
    rec.citations = verdict.kept

    # 2. Grounding - refuse to answer from thin air.
    matched, total = index.informative_overlap(question)
    # Deliberately the SEED retrieval - the search on what the user actually
    # asked - not retrieved, which accumulates every follow-up query
    # the model issued. Measured: an out-of-scope question scored 0.444 across
    # the union but 0.368 on the seed, because the model rewrote it into
    # policy vocabulary. Scoring the union lets the agent talk itself past its
    # own grounding gate.
    grounding = assess_grounding(
        seed if seed is not None else retrieved,
        matched,
        total,
        index.semantic_scores_meaningful,
        settings.triage_min_semantic_similarity,
    )
    if not grounding.grounded:
        flags.append(
            GuardrailFlag(
                name="low_retrieval_confidence",
                detail=f"The SOP corpus does not appear to cover this question: "
                f"{grounding.as_detail()}",
                blocking=True,
            )
        )
    elif not grounding.active:
        # Visible, not silent: the operator must know a control is switched off.
        flags.append(
            GuardrailFlag(
                name="grounding_check_inactive",
                detail=grounding.as_detail(),
                blocking=False,
            )
        )
    if grounding.grounded and not rec.citations:
        flags.append(
            GuardrailFlag(
                name="uncited_recommendation",
                detail="Recommendation carried no verifiable citation.",
                blocking=True,
            )
        )

    # 3. Policy contradiction across everything the model was shown.
    seen: list[Chunk] = []
    seen_ids: set[str] = set()
    for hit in retrieved:
        if hit.chunk.chunk_id not in seen_ids:
            seen_ids.add(hit.chunk.chunk_id)
            seen.append(hit.chunk)
    material = detect_threshold_conflicts(seen, observations)
    material_keys = {c.dimension.key for c in material}
    for conflict in material:
        flags.append(
            GuardrailFlag(
                name="policy_contradiction",
                detail=f"{conflict.describe()} - this PO falls between the "
                "competing thresholds, so the sections disagree on the outcome.",
                blocking=True,
            )
        )
    # Conflicts that exist in the corpus but do not change THIS decision are
    # reported for policy hygiene without blocking the recommendation.
    for conflict in detect_all_conflicts(seen):
        if conflict.dimension.key not in material_keys:
            flags.append(
                GuardrailFlag(
                    name="policy_contradiction_immaterial",
                    detail=f"{conflict.describe()} - both sections agree on the "
                    "outcome at this PO's figures, so the recommendation stands. "
                    "Flagged for Merchandising Operations to reconcile.",
                    blocking=False,
                )
            )

    # 4. Enforce the invariants. A blocking flag means no auto-action, whatever
    #    the model chose. The original action is recorded for audit.
    if any(f.blocking for f in flags):
        if rec.recommended_action != "escalate":
            flags.append(
                GuardrailFlag(
                    name="action_overridden",
                    detail=f"Model proposed '{rec.recommended_action}'; overridden to "
                    "'escalate' because a blocking guardrail fired.",
                )
            )
            rec.recommended_action = "escalate"
        rec.confidence = "low"
        if rec.escalation_target_role is None:
            rec.escalation_target_role = "Senior Merch Planner"

    # 5. PII - defence in depth. Index-time redaction is the real control;
    #    this catches leaks arriving by any other route (e.g. the question).
    leaks = scan_for_pii(rec.rationale, index.registry)
    if leaks:
        rec.rationale, _ = redact(rec.rationale, index.registry)
        flags.append(
            GuardrailFlag(
                name="pii_in_output",
                detail=f"Redacted from rationale: {leaks}",
                blocking=True,
            )
        )

    return TriageResult(
        recommendation=rec,
        guardrail_flags=flags,
        retrieved_chunk_ids=sorted(exposed_chunk_ids),
        resolved_citations=verdict.resolved,
        dropped_citations=verdict.fabricated,
        tool_calls=tool_log,
        semantic_similarity=grounding.semantic_similarity,
        grounding_reason=grounding.reason,
        steps_used=steps,
        provider=provider,
        usage=usage,
    )
