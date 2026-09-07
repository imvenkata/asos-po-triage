"""Tests for the LangGraph alternative orchestration.

These run without credentials: they cover tool construction, the routing
function, and the shared-policy-gate claim. Behaviour against a real model is
covered by `python evals/run_evals.py --agent langgraph`.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="optional extra: pip install -e '.[langgraph]'")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alternatives"))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph_agent import (  # noqa: E402
    TERMINAL_TOOL,
    LangGraphTriageAgent,
    RetrievalRecorder,
    build_tools,
)

from triage.data_access import get_repository  # noqa: E402


def test_exposes_the_same_four_tools(settings, index):
    tools = build_tools(get_repository(settings), index, RetrievalRecorder(), 6)
    assert {t.name for t in tools} == {
        "get_po", "get_forecast", "search_sops", TERMINAL_TOOL
    }


def test_get_po_still_returns_precomputed_variance(settings, index):
    """The model must never do arithmetic - that guarantee is orchestration-independent."""
    import json

    tools = build_tools(get_repository(settings), index, RetrievalRecorder(), 6)
    po_tool = next(t for t in tools if t.name == "get_po")
    payload = json.loads(po_tool.invoke({"po_id": "PO-10342"}))
    assert payload["computed_variance"]["qty_variance_pct"] == 12.0


def test_search_records_every_chunk_shown(settings, index):
    recorder = RetrievalRecorder()
    tools = build_tools(get_repository(settings), index, recorder, 4)
    next(t for t in tools if t.name == "search_sops").invoke({"query": "backorder wholesale"})
    assert len(recorder.exposed_chunk_ids) == 4
    assert len(recorder.retrieved) == 4


def test_submit_schema_mirrors_the_output_contract(settings, index):
    """If these drift, the two orchestrations stop being comparable."""
    from triage.models import TriageRecommendation

    tools = build_tools(get_repository(settings), index, RetrievalRecorder(), 6)
    submit = next(t for t in tools if t.name == TERMINAL_TOOL)
    assert set(submit.args_schema.model_fields) == set(TriageRecommendation.model_fields)


def test_harvest_extracts_the_submission_and_trace():
    messages = [
        HumanMessage(content="what about PO-10342?"),
        AIMessage(content="", tool_calls=[
            {"name": "get_po", "args": {"po_id": "PO-10342"}, "id": "1"}
        ]),
        AIMessage(content="", tool_calls=[{
            "name": TERMINAL_TOOL,
            "args": {
                "po_id": "PO-10342", "recommended_action": "escalate",
                "rationale": "x", "citations": ["po_amendment_policy.md §2"],
                "confidence": "low", "escalation_target_role": "Senior Merch Planner",
            },
            "id": "2",
        }]),
    ]
    raw, tool_log, steps, _ = LangGraphTriageAgent._harvest(messages)
    assert raw is not None and raw["recommended_action"] == "escalate"
    assert [t.name for t in tool_log] == ["get_po", TERMINAL_TOOL]
    assert steps == 2


def test_invalid_submission_is_not_harvested():
    """A malformed payload must reach the policy gate as 'no valid submission'
    rather than being passed through."""
    messages = [
        AIMessage(content="", tool_calls=[{
            "name": TERMINAL_TOOL,
            "args": {"po_id": "PO-1", "recommended_action": "teleport",
                     "rationale": "x", "citations": [], "confidence": "low"},
            "id": "1",
        }]),
    ]
    raw, tool_log, _, _ = LangGraphTriageAgent._harvest(messages)
    assert raw is None
    assert "rejected" in tool_log[0].result_summary


def test_scripted_provider_is_refused_with_a_clear_message(settings):
    """The offline double drives the hand-written loop only. Fail loudly."""
    from langgraph_agent import _build_chat_model
    from triage.llm.base import LLMError

    with pytest.raises(LLMError, match="no LangChain adapter"):
        _build_chat_model(settings)


def test_both_orchestrations_import_the_same_policy_gate():
    """The claim this file exists to demonstrate, asserted rather than argued."""
    import langgraph_agent

    import triage.agent as handwritten
    from triage.policy_gate import apply_policy_gate

    assert handwritten.apply_policy_gate is apply_policy_gate
    assert langgraph_agent.apply_policy_gate is apply_policy_gate
