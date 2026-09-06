import pytest

from triage.agent import TriageAgent, build_agent
from triage.data_access import get_repository
from triage.llm.base import LLMResponse, ToolCall
from triage.llm.scripted import ScriptedLLMClient


@pytest.fixture(scope="module")
def agent(settings, index):
    return TriageAgent(ScriptedLLMClient(), index, get_repository(settings), settings)


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
    assert called.index("get_po") < called.index("submit_recommendation")
    assert "search_sops" in called


def test_unknown_po_escalates_rather_than_inventing_one(agent):
    result = agent.triage("What should I do about PO-99999?")
    assert result.recommendation.recommended_action == "escalate"
    assert result.recommendation.confidence == "low"


class _FabricatingClient(ScriptedLLMClient):
    """Simulates the failure we most need to survive: a confident invented citation."""

    def chat(self, messages, tools=None, tool_choice="auto"):
        response = super().chat(messages, tools, tool_choice)
        call = response.tool_calls[0]
        if call.name != "submit_recommendation":
            return response
        args = dict(call.arguments)
        args["recommended_action"] = "amend"
        args["confidence"] = "high"
        args["citations"] = ["procurement_handbook.md §7"]
        args["escalation_target_role"] = None
        return LLMResponse(tool_calls=[ToolCall(call.id, call.name, args)], usage={})


def test_fabricated_citation_blocks_the_auto_action(settings, index):
    agent = TriageAgent(_FabricatingClient(), index, get_repository(settings), settings)
    result = agent.triage("PO-10001 is slightly short, can I amend it?")
    rec = result.recommendation
    assert result.dropped_citations == ["procurement_handbook.md §7"]
    # The model said "amend, high confidence". The policy gate overrules it.
    assert rec.recommended_action == "escalate"
    assert rec.confidence == "low"
    assert any(f.name == "action_overridden" for f in result.guardrail_flags)


def test_build_agent_wires_the_scripted_provider(settings):
    assert isinstance(build_agent(settings), TriageAgent)
