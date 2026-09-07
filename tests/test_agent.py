import pytest
from langchain_core.messages import AIMessage, HumanMessage

from triage.agent import TriageAgent, build_agent
from triage.data_access import get_repository
from triage.llm.scripted_chat import ScriptedChatModel
from triage.tools.registry import TERMINAL_TOOL


@pytest.fixture()
def agent(settings, index):
    return TriageAgent(ScriptedChatModel(), index, get_repository(settings), settings)


def test_clean_variance_is_amended_and_not_blocked(agent):
    result = agent.triage("PO-10001 came back slightly short. What should I do?")
    assert result.recommendation.recommended_action == "amend"
    assert not result.blocked
    assert result.recommendation.citations


def test_disputed_band_escalates_with_low_confidence(agent):
    result = agent.triage("PO-10342 is 12% under. Can I amend it in place?")
    rec = result.recommendation
    assert rec.recommended_action == "escalate"
    assert rec.confidence == "low"
    assert any(f.name == "policy_contradiction" and f.blocking for f in result.guardrail_flags)


def test_agent_calls_tools_before_recommending(agent):
    result = agent.triage("What should happen with PO-10777?")
    called = [c.name for c in result.tool_calls]
    assert called.index("get_po") < called.index(TERMINAL_TOOL)
    assert "search_sops" in called


def test_unknown_po_escalates_rather_than_inventing_one(agent):
    result = agent.triage("What should I do about PO-99999?")
    assert result.recommendation.recommended_action == "escalate"
    assert result.recommendation.confidence == "low"


class _FabricatingModel(ScriptedChatModel):
    """The failure we most need to survive: a confident invented citation."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop, run_manager, **kwargs)
        call = result.generations[0].message.tool_calls[0]
        if call["name"] != TERMINAL_TOOL:
            return result
        args = dict(call["args"])
        args.update(
            recommended_action="amend",
            confidence="high",
            citations=["procurement_handbook.md §7"],
            escalation_target_role=None,
        )
        return self._call(TERMINAL_TOOL, args)


def test_fabricated_citation_blocks_the_auto_action(settings, index):
    agent = TriageAgent(_FabricatingModel(), index, get_repository(settings), settings)
    result = agent.triage("PO-10001 is slightly short, can I amend it?")
    rec = result.recommendation
    assert result.dropped_citations == ["procurement_handbook.md §7"]
    # The model said "amend, high confidence". The policy gate overrules it.
    assert rec.recommended_action == "escalate"
    assert rec.confidence == "low"
    assert any(f.name == "action_overridden" for f in result.guardrail_flags)


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
    raw, tool_log, steps, _ = TriageAgent._harvest(messages)
    assert raw is not None and raw["recommended_action"] == "escalate"
    assert [t.name for t in tool_log] == ["get_po", TERMINAL_TOOL]
    assert steps == 2


def test_invalid_submission_is_not_harvested():
    """A malformed payload reaches the gate as 'no valid submission' rather than
    being passed through as if it were valid."""
    messages = [
        AIMessage(content="", tool_calls=[{
            "name": TERMINAL_TOOL,
            "args": {"po_id": "PO-1", "recommended_action": "teleport",
                     "rationale": "x", "citations": [], "confidence": "low"},
            "id": "1",
        }]),
    ]
    raw, tool_log, _, _ = TriageAgent._harvest(messages)
    assert raw is None
    assert "rejected" in tool_log[0].result_summary


def test_build_agent_wires_the_scripted_provider(settings):
    assert isinstance(build_agent(settings), TriageAgent)
