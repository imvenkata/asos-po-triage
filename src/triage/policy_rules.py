"""Explicit controls for the mock SOPs, with source references and boundary tests.

This is a small reviewed rule set, not a natural-language policy compiler.
Every action remains advisory. Unmodelled split plans need a planner to size and
approve the actual children; this interface never executes them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import GuardrailFlag, PurchaseOrder, TriageRecommendation
from .tools.po_tools import compute_variance

VALUE_CEILING = 50_000
MINOR_PERCENT = 5
CRITICAL_PERCENT = 30
MAX_ETA_DAYS = 20
BACKORDER_DELAY_DAYS = 28


@dataclass
class RuleVerdict:
    flags: list[GuardrailFlag] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    role: str = "Senior Merch Planner"
    review_required: bool = True


def check_business_rules(po: PurchaseOrder, rec: TriageRecommendation, payload: dict) -> RuleVerdict:
    v = compute_variance(po)
    out = RuleVerdict()
    if po.channel == "wholesale":
        out.role = "Wholesale Planning Lead"
    elif po.value_gbp > VALUE_CEILING:
        out.role = "Head of Buying"

    def block(name, detail, citation):
        out.flags.append(GuardrailFlag(name=name, detail=detail, blocking=True))
        if citation not in out.citations:
            out.citations.append(citation)

    risk = re.search(r"\b(?:administration|insolvency|insolvent)\b", po.supplier_note or "", re.I)
    if risk:
        out.role = "Supply Chain Risk Lead"
        block("supplier_risk", "Supplier viability needs human judgement; no ordinary PO action is approved.",
              "merch_escalation_matrix.md §3")
    if not v["baseline_valid"]:
        block("invalid_baseline", "A zero quantity or value baseline prevents a valid percentage calculation.",
              "variance_detection_sop.md §1")
        return out

    qty, value, eta = abs(v["qty_variance_pct"]), abs(v["value_variance_pct"]), abs(v["eta_slip_days"])
    if po.value_gbp > VALUE_CEILING or qty > CRITICAL_PERCENT or value > CRITICAL_PERCENT or eta > MAX_ETA_DAYS:
        block("critical_tier", "The PO exceeds a critical-tier boundary and requires escalation.",
              "variance_detection_sop.md §2")
    if po.status not in {"open", "partially_received", "planned"}:
        block("ineligible_po_status", "This PO status does not permit a routine recommendation.",
              "po_amendment_policy.md §2")

    if rec.recommended_action == "escalate":
        return out
    if po.status == "planned":
        if rec.recommended_action != "firm_planned_order" or po.supplier_acknowledged is not False:
            block("planned_order_evidence", "Firming requires a planned PO and explicit confirmation that the supplier has not acknowledged it.",
                  "po_amendment_policy.md §5")
        return out
    if rec.recommended_action == "firm_planned_order":
        block("not_a_planned_order", "A live PO cannot be firmed as an untransmitted planned order.",
              "po_amendment_policy.md §5")
    elif rec.recommended_action == "amend":
        if value > 10:
            block("amendment_re_raise", "The cancel-and-re-raise rule prevents this in-place amendment.",
                  "po_amendment_policy.md §4")
        out.review_required = not (qty <= MINOR_PERCENT and value <= MINOR_PERCENT and eta <= 3
                                   and po.parent_po_id is None and not out.flags)
    elif rec.recommended_action == "raise_backorder":
        if po.channel != "retail":
            block("wholesale_backorder_ban", "Wholesale shortfalls cannot be backordered.",
                  "backorder_reconciliation.md §2")
        if v["unfulfilled_qty"] <= 0 or qty <= MINOR_PERCENT or po.remainder_eta is not None:
            block("backorder_ineligible", "Backordering requires an above-minor shortfall without a firm remainder date.",
                  "backorder_reconciliation.md §1")
        if po.expected_shortfall_delay_days is None or po.expected_shortfall_delay_days > BACKORDER_DELAY_DAYS:
            block("backorder_delay_evidence", "A permissible expected shortfall delay has not been established.",
                  "backorder_reconciliation.md §3")
    elif rec.recommended_action == "split_child_po":
        if v["unfulfilled_qty"] <= 0 or po.remainder_eta is None or po.remainder_eta <= po.eta:
            block("split_without_window", "A delivery-window split requires a positive shortfall and a known later date.",
                  "child_po_split_rules.md §1")
        if payload.get("child_po_count", 0) >= 3:
            block("split_limit", "The parent already has the maximum permitted children.",
                  "child_po_split_rules.md §2")
        # Only the delivery-window split can be established from this mock schema.
        # Quantity-only proof is conservative; no child valuation is invented.
        if min(po.confirmed_qty, v["unfulfilled_qty"]) < 500:
            block("split_sizing_evidence", "Child sizing cannot be established from the available quantity and valuation data.",
                  "child_po_split_rules.md §2")
    return out
