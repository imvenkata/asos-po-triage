"""Deterministic chat model for offline runs.

WHAT THIS IS: a test double implementing the LangChain `BaseChatModel`
interface, so it drops into the same graph the real model runs in. It calls the
real tools, consumes the real retrieved chunks, and emits schema-valid tool
calls. This makes the pipeline, the guardrails and the eval harness runnable in
CI with zero credentials.

WHAT THIS IS NOT: a language model. Offline results measure orchestration and
guardrails only. Model-quality claims require a real provider, and the eval
runner prints that caveat on every offline run.
"""
from __future__ import annotations

import json
import re
from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

TERMINAL_TOOL = "submit_recommendation"
_ESCALATE = "escalate"


def _tool_results(messages: Sequence[BaseMessage], name: str) -> list[dict[str, Any]]:
    """Pull previously-returned results for a named tool out of the transcript."""
    wanted: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                if call["name"] == name:
                    wanted.add(call["id"])
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id in wanted:
            try:
                out.append(json.loads(str(msg.content)))
            except (json.JSONDecodeError, TypeError):
                continue
    return out


class ScriptedChatModel(BaseChatModel):
    """Walks a fixed plan: look up the PO, retrieve policy, then recommend."""

    counter: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-offline-double"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        # The plan is fixed, so the schemas are not consulted.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        question = next(
            (str(m.content) for m in messages if m.type == "human"), ""
        )
        po_rows = _tool_results(messages, "get_po")

        if not po_rows:
            match = re.search(r"PO-\d+", question or "")
            return self._call("get_po", {"po_id": match.group(0) if match else "UNKNOWN"})

        po = po_rows[0]
        if po.get("error"):
            return self._call(TERMINAL_TOOL, {
                "po_id": po.get("po_id", "UNKNOWN"),
                "recommended_action": _ESCALATE,
                "rationale": f"PO could not be retrieved: {po['error']}. No policy "
                             "can be applied without the PO record.",
                "citations": [],
                "confidence": "low",
                "escalation_target_role": "Senior Merch Planner",
            })

        if not _tool_results(messages, "search_sops"):
            return self._call("search_sops", {
                "query": f"{question} variance amendment threshold "
                         f"{po.get('channel','')} {po.get('status','')}"
            })

        return self._call(TERMINAL_TOOL, self._decide(po, messages))

    def _call(self, name: str, args: dict[str, Any]) -> ChatResult:
        self.counter += 1
        message = AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{self.counter}"}],
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    # --- the double's rule table -----------------------------------------
    def _decide(self, po: dict[str, Any], messages: Sequence[BaseMessage]) -> dict[str, Any]:
        cites = self._available_citations(messages)
        v = po.get("computed_variance", {})
        qty_var = abs(float(v.get("qty_variance_pct") or 0.0))
        over = float(v.get("qty_variance_pct") or 0.0) < 0
        value = float(po.get("value_gbp") or 0.0)
        channel, status = po.get("channel"), po.get("status")
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
                       "PO is still in planned status and has not been transmitted to the "
                       "supplier, so it should be firmed at the corrected quantity.",
                       "high", None, ("po_amendment_policy",))

        if any(w in note for w in ("administration", "insolven", "force majeure", "customs")):
            return out(_ESCALATE,
                       "The PO is affected by a supplier viability event, which has no "
                       "defined variance signature in the SOPs.",
                       "high", "Supply Chain Risk Lead",
                       ("variance_detection_sop", "merch_escalation_matrix"))

        if over:
            return out(_ESCALATE,
                       "The supplier confirmed more units than were ordered. The SOPs "
                       "define variance only for under-confirmation.",
                       "medium", "Senior Merch Planner", ("variance_detection_sop",))

        if self._contradiction_in_context(messages):
            return out(_ESCALATE,
                       "The retrieved policy sections specify conflicting variance "
                       "thresholds for this decision.",
                       "low", "Senior Merch Planner",
                       ("po_amendment_policy", "variance_detection_sop"))

        if value > 50_000 or qty_var > 30:
            role = "Wholesale Planning Lead" if channel == "wholesale" else "Head of Buying"
            return out(_ESCALATE,
                       "The PO breaches the critical tier on value and/or variance, so no "
                       "auto-action is permitted.",
                       "high", role, ("variance_detection_sop", "po_amendment_policy"))

        if qty_var > 15:
            if po.get("remainder_eta"):
                return out("split_child_po",
                           "A firm delivery date exists for the shortfall, so the parent "
                           "should be sub-divided rather than backordered.",
                           "high", None, ("child_po_split_rules",))
            if channel == "wholesale":
                return out(_ESCALATE,
                           "Backorders are not permitted on wholesale POs and no later "
                           "delivery window has been agreed.",
                           "high", "Wholesale Planning Lead",
                           ("backorder_reconciliation", "merch_escalation_matrix"))
            return out("raise_backorder",
                       "Retail PO with an unfulfilled quantity and no firm remainder date, "
                       "within the maximum permissible backorder delay.",
                       "high", None, ("backorder_reconciliation",))

        return out("amend",
                   "Variance sits within the minor tier and the amended value is below the "
                   "planner self-approval ceiling.",
                   "high", None, ("po_amendment_policy", "variance_detection_sop"))

    @staticmethod
    def _available_citations(messages: Sequence[BaseMessage]) -> list[str]:
        found: list[str] = []
        for msg in messages:
            if msg.type in ("tool", "system") and isinstance(msg.content, str):
                for cid in re.findall(r"[a-z_]+\.md §\d+", msg.content):
                    if cid not in found:
                        found.append(cid)
        return found

    @staticmethod
    def _contradiction_in_context(messages: Sequence[BaseMessage]) -> bool:
        return any(
            isinstance(m.content, str)
            and "CONFLICTING POLICY THRESHOLDS DETECTED" in m.content
            for m in messages
        )
