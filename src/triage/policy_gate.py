"""Validate evidence, enforce explicit policy constraints, then sanitize the response."""

from __future__ import annotations

import json

from .guardrails.contradiction import detect_all_conflicts, detect_threshold_conflicts
from .guardrails.grounding import (
    assess_grounding,
    inline_citations,
    normalize_citations,
    unsupported_numbers,
    validate_citations,
)
from .guardrails.pii import sanitize, scan_for_pii
from .models import GuardrailFlag, PurchaseOrder, TriageRecommendation, TriageResult
from .policy_rules import check_business_rules
from .tools.po_tools import compute_variance

# Blocking flags that mean the model's own prose cannot be trusted: it invented a
# reference, contradicted itself, leaked data, or reasoned about a PO it never
# fetched. When one of these fires the rationale is rebuilt from the flags.
#
# Everything else is a POLICY determination - the model read the SOPs correctly
# and policy simply requires a human. Discarding its explanation there costs the
# planner the reason without buying any safety, so the rationale is preserved and
# the policy determination appended.
UNTRUSTED_OUTPUT_FLAGS = frozenset(
    {
        "no_valid_submission",
        "missing_po_evidence",
        "missing_request_po",
        "po_identity_mismatch",
        "pii_in_output",
        "fabricated_citation",
        "unread_citation",
        "citation_mismatch",
        "uncited_recommendation",
        "unsupported_numeric_claim",
        "low_retrieval_confidence",
        "semantic_control_unavailable",
        "inconsistent_escalation_role",
        "escalation_role_corrected",
    }
)


def apply_policy_gate(
    *,
    question,
    raw,
    index,
    settings,
    exposed_chunk_ids,
    retrieved,
    seed,
    observations,
    tool_log,
    steps,
    usage,
    provider,
    po_id,
    po: PurchaseOrder | None = None,
    po_payload: dict | None = None,
    forecasts: list[dict] | None = None,
    initial_flags=None,
    model_metadata=None,
) -> TriageResult:
    flags = list(initial_flags or [])
    policy_citations = set()
    role = "Senior Merch Planner"
    review_required = True

    def block(name, detail):
        flags.append(GuardrailFlag(name=name, detail=detail, blocking=True))

    rec = (
        TriageRecommendation.model_validate(raw)
        if raw
        else TriageRecommendation(
            po_id=po_id,
            recommended_action="escalate",
            rationale="No valid model submission was available.",
            confidence="low",
            escalation_target_role=role,
        )
    )
    if raw is None:
        block(
            "no_valid_submission",
            "No schema-valid recommendation was completed within the permitted attempts.",
        )
    if po is None:
        block("missing_po_evidence", "No successful PO lookup supports a recommendation.")
    if po is not None and (po_id != po.po_id or rec.po_id != po.po_id):
        block(
            "po_identity_mismatch",
            "The proposed PO does not match the requested and fetched record.",
        )
    if po_id == "UNKNOWN":
        block(
            "missing_request_po",
            "A single PO identifier is required for this recommendation contract.",
        )
    # An untrusted output never chooses the identity checked by enforcement.
    rec.po_id = po_id

    if scan_for_pii(
        json.dumps({"raw": raw, "trace": [t.model_dump() for t in tool_log]}), index.registry
    ):
        block(
            "pii_in_output",
            "Personal contact data was detected and removed from the public response.",
        )

    inline = inline_citations(rec.rationale)
    verdict = validate_citations(rec.citations + inline, exposed_chunk_ids, index.chunk_ids)
    if verdict.fabricated:
        block("fabricated_citation", "A reference does not exist in the SOP corpus.")
    if verdict.unexposed:
        block(
            "unread_citation",
            "A cited section was not read; retrieve its text before relying on it.",
        )
    if raw and normalize_citations(rec.citations) != set(inline):
        block("citation_mismatch", "Inline references and the citations array disagree.")
    rec.citations = verdict.kept
    if raw and not rec.citations:
        block("uncited_recommendation", "The proposal contains no verified policy evidence.")
    cited_text = "\n".join(index.get_chunk(c).text for c in verdict.kept)
    facts = dict(po_payload or {})
    facts["forecasts"] = forecasts or []
    if raw and unsupported_numbers(rec.rationale, cited_text, facts):
        block(
            "unsupported_numeric_claim",
            "A numerical claim is absent from the cited text and structured evidence.",
        )

    matched, total = index.informative_overlap(question)
    grounding = assess_grounding(
        seed if seed is not None else retrieved,
        matched,
        total,
        index.semantic_scores_meaningful,
        settings.triage_min_semantic_similarity,
    )
    if not grounding.grounded:
        block(
            "low_retrieval_confidence",
            "The original question has insufficient semantic policy support.",
        )
    elif not grounding.active:
        flags.append(GuardrailFlag(name="grounding_check_inactive", detail=grounding.as_detail()))
        if settings.triage_llm_provider != "scripted":
            block(
                "semantic_control_unavailable",
                "Semantic grounding is unavailable in this degraded demonstration.",
            )

    if po is not None:
        observations = compute_variance(po)
        # Known contradictions are deterministic policy controls, not contingent
        # on which passages the model chose to retrieve.
        material = detect_threshold_conflicts(index.chunks, observations)
        for conflict in material:
            block("policy_contradiction", conflict.describe() + " changes this PO's outcome.")
            policy_citations.update(c.chunk_id for c in conflict.claims)
        for conflict in detect_all_conflicts(index.chunks):
            if conflict.dimension.key not in {c.dimension.key for c in material}:
                flags.append(
                    GuardrailFlag(
                        name="policy_contradiction_immaterial",
                        detail=conflict.describe() + " does not change this PO's outcome.",
                    )
                )
        rules = check_business_rules(po, rec, po_payload or {})
        flags.extend(rules.flags)
        policy_citations.update(rules.citations)
        role, review_required = rules.role, rules.review_required
        if material and not rules.flags:
            role = "Senior Merch Planner"

    if rec.recommended_action == "escalate" and rec.escalation_target_role != role:
        block(
            "escalation_role_corrected", "Escalation routing was corrected to the applicable role."
        )
    if rec.recommended_action != "escalate" and rec.escalation_target_role is not None:
        block(
            "inconsistent_escalation_role",
            "An escalation role was supplied without an escalation action.",
        )

    # All checks precede this one final enforcement pass.
    if any(f.blocking for f in flags):
        overridden = rec.recommended_action != "escalate"
        if overridden:
            flags.append(
                GuardrailFlag(
                    name="action_overridden",
                    detail=f"Proposed {rec.recommended_action}; validation requires escalation.",
                )
            )
        rec.recommended_action = "escalate"
        rec.confidence = "low"
        rec.escalation_target_role = role
        review_required = True

        blocked = {f.name for f in flags if f.blocking}
        reasons = " ".join(f.detail for f in flags if f.blocking)
        policy_refs = sorted(c for c in policy_citations if index.get_chunk(c) is not None)

        # Rebuild the explanation when the model's output failed an integrity
        # check, or when its prose argues for an action we have just overruled -
        # in both cases the rationale describes something that is not happening.
        if blocked & UNTRUSTED_OUTPUT_FLAGS or overridden:
            rec.citations = policy_refs
            refs = " ".join(f"[{c}]" for c in policy_refs)
            rec.rationale = f"Escalate to {role}. {reasons} {refs}".strip()
        else:
            # The model chose escalation itself and its output passed every
            # integrity check, so the explanation is safe to surface. Keep it and
            # append the policy determination rather than discarding both.
            rec.citations = sorted(set(rec.citations) | set(policy_refs))
            refs = " ".join(f"[{c}]" for c in policy_refs)
            rec.rationale = (
                f"{rec.rationale.rstrip()} Escalation is required: {reasons} {refs}".strip()
            )

    result = TriageResult(
        recommendation=rec,
        guardrail_flags=flags,
        retrieved_chunk_ids=sorted(exposed_chunk_ids),
        validated_chunk_ids=sorted(set(rec.citations) | exposed_chunk_ids),
        resolved_citations=[],
        dropped_citations=verdict.fabricated,
        tool_calls=tool_log,
        semantic_similarity=grounding.semantic_similarity,
        grounding_reason=grounding.reason,
        steps_used=steps,
        provider=provider,
        usage=usage,
        evidence_po_id=po.po_id if po is not None else None,
        review_required=review_required,
        model_metadata=model_metadata or {},
    )
    # Public traces are summaries, never raw model arguments or error bodies.
    result.tool_calls = [
        t.model_copy(
            update={
                "arguments": {
                    k: v
                    for k, v in t.arguments.items()
                    if k in {"po_id", "sku"} and isinstance(v, str)
                }
            }
        )
        for t in result.tool_calls
    ]
    return TriageResult.model_validate(sanitize(result.model_dump(mode="json"), index.registry))
