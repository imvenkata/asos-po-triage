"""Deterministic offline stand-in for a tool-calling LLM.

WHAT THIS IS: a test double that drives the real agent loop - it calls the real
tools, consumes the real retrieved chunks, and emits a schema-valid
recommendation. It makes the pipeline, the guardrails, and the eval harness
runnable in CI with zero credentials.

WHAT THIS IS NOT: evidence that a language model reasons correctly over the SOPs.
Offline eval results measure orchestration and guardrails only. Model-quality
claims require `TRIAGE_LLM_PROVIDER=azure`. This distinction is stated in
WRITEUP.md and printed by the eval runner.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .base import LLMResponse, ToolCall
from .local_embeddings import embed_texts

_ESCALATE = "escalate"


def _tool_results(messages: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Pull previously-returned results for a named tool out of the transcript."""
    wanted_ids = {
        tc["id"]
        for m in messages
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
        if tc["function"]["name"] == name
    }
    out = []
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id") in wanted_ids:
            try:
                out.append(json.loads(m["content"]))
            except (json.JSONDecodeError, TypeError):
                continue
    return out


class ScriptedLLMClient:
    """Walks a fixed plan: look up the PO, retrieve policy, then recommend."""

    name = "scripted:offline-double"

    def __init__(self) -> None:
        self._counter = 0

    def _call(self, name: str, arguments: dict[str, Any]) -> LLMResponse:
        self._counter += 1
        return LLMResponse(
            tool_calls=[ToolCall(id=f"call_{self._counter}", name=name, arguments=arguments)],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="tool_calls",
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        question = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        po_rows = _tool_results(messages, "get_po")

        if not po_rows:
            match = re.search(r"PO-\d+", question or "")
            return self._call("get_po", {"po_id": match.group(0) if match else "UNKNOWN"})

        po = po_rows[0]
        if po.get("error"):
            return self._submit(
                {
                    "po_id": po.get("po_id", "UNKNOWN"),
                    "recommended_action": _ESCALATE,
                    "rationale": f"PO could not be retrieved: {po['error']}. No policy can be applied without the PO record.",
                    "citations": [],
                    "confidence": "low",
                    "escalation_target_role": "Senior Merch Planner",
                }
            )

        if not _tool_results(messages, "search_sops"):
            return self._call(
                "search_sops",
                {"query": f"{question} variance amendment threshold {po.get('channel','')} {po.get('status','')}"},
            )

        return self._submit(self._decide(po, messages))

    def _submit(self, payload: dict[str, Any]) -> LLMResponse:
        self._counter += 1
        return LLMResponse(
            tool_calls=[
                ToolCall(id=f"call_{self._counter}", name="submit_recommendation", arguments=payload)
            ],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="tool_calls",
        )

    # --- the double's rule table -----------------------------------------
    def _decide(self, po: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
        cites = self._available_citations(messages)
        v = po.get("computed_variance", {})
        qty_var = abs(float(v.get("qty_variance_pct") or 0.0))
        over = float(v.get("qty_variance_pct") or 0.0) < 0
        value = float(po.get("value_gbp") or 0.0)
        channel = po.get("channel")
        status = po.get("status")
        note = (po.get("supplier_note") or "").lower()

        def out(action, rationale, confidence, role=None, want=()):
            return {
                "po_id": po["po_id"],
                "recommended_action": action,
                "rationale": rationale,
                "citations": [c for c in cites if any(w in c for w in want)][:3] or cites[:2],
                "confidence": confidence,
                "escalation_target_role": role,
            }

        if status == "planned":
            return out("firm_planned_order",
                       "PO is still in planned status and has not been transmitted to the supplier, so it should be firmed at the corrected quantity rather than amended.",
                       "high", None, ("po_amendment_policy",))

        if any(w in note for w in ("administration", "insolven", "force majeure", "customs")):
            return out(_ESCALATE,
                       "The PO is affected by a supplier viability event, which has no defined variance signature in the SOPs. No action may be inferred by analogy.",
                       "high", "Supply Chain Risk Lead", ("variance_detection_sop", "merch_escalation_matrix"))

        if over:
            return out(_ESCALATE,
                       "The supplier confirmed more units than were ordered. The SOPs define variance only for under-confirmation and are silent on over-confirmation, so no threshold applies.",
                       "medium", "Senior Merch Planner", ("variance_detection_sop",))

        if self._contradiction_in_context(messages):
            return out(_ESCALATE,
                       "The retrieved policy sections specify conflicting variance thresholds for this decision, so the exception must be adjudicated rather than actioned.",
                       "low", "Senior Merch Planner", ("po_amendment_policy", "variance_detection_sop"))

        if value > 50_000 or qty_var > 30:
            role = "Wholesale Planning Lead" if channel == "wholesale" else "Head of Buying"
            return out(_ESCALATE,
                       "The PO breaches the critical tier on value and/or variance, so no auto-action is permitted and sign-off sits above planner level.",
                       "high", role, ("variance_detection_sop", "po_amendment_policy"))

        if qty_var > 15:  # a genuine fulfilment shortfall, not just a variance
            if po.get("remainder_eta"):
                return out("split_child_po",
                           "A firm delivery date exists for the shortfall, so the parent should be sub-divided rather than backordered.",
                           "high", None, ("child_po_split_rules",))
            if channel == "wholesale":
                return out(_ESCALATE,
                           "Backorders are not permitted on wholesale POs and no later delivery window has been agreed, so the shortfall cannot be reconciled automatically.",
                           "high", "Wholesale Planning Lead", ("backorder_reconciliation", "merch_escalation_matrix"))
            return out("raise_backorder",
                       "Retail PO with an unfulfilled quantity and no firm remainder date, within the maximum permissible backorder delay.",
                       "high", None, ("backorder_reconciliation",))

        return out("amend",
                   "Variance sits within the minor tier and the amended value is below the planner self-approval ceiling, so the PO can be amended in place.",
                   "high", None, ("po_amendment_policy", "variance_detection_sop"))

    @staticmethod
    def _available_citations(messages: list[dict[str, Any]]) -> list[str]:
        found: list[str] = []
        for m in messages:
            if m.get("role") in ("tool", "system") and isinstance(m.get("content"), str):
                for cid in re.findall(r"[a-z_]+\.md §\d+", m["content"]):
                    if cid not in found:
                        found.append(cid)
        return found

    @staticmethod
    def _contradiction_in_context(messages: list[dict[str, Any]]) -> bool:
        return any(
            "CONFLICTING POLICY THRESHOLDS DETECTED" in (m.get("content") or "")
            for m in messages
            if isinstance(m.get("content"), str)
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)
