"""Hostile model outputs test controls independently of the happy-path double."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from triage.agent import TriageAgent, build_agent
from triage.config import Settings
from triage.data_access import get_repository
from triage.guardrails.grounding import inline_citations, unsupported_numbers
from triage.guardrails.pii import scan_for_pii
from triage.llm.base import LLMError, TriageTimeout
from triage.llm.scripted_chat import ScriptedChatModel
from triage.models import TriageRecommendation
from triage.policy_rules import check_business_rules
from triage.tools.po_tools import compute_variance
from triage.tools.registry import TERMINAL_TOOL

QUESTION = "PO-10001 has a minor variance. Can I amend it in place?"


def proposal(**changes):
    return (
        dict(
            po_id="PO-10001",
            recommended_action="amend",
            rationale="Routine amendment [po_amendment_policy.md §2].",
            citations=["po_amendment_policy.md §2"],
            confidence="high",
            escalation_target_role=None,
        )
        | changes
    )


class OverrideModel(ScriptedChatModel):
    changes: dict

    def _decide(self, po, messages):
        return super()._decide(po, messages) | self.changes


class DirectModel(ScriptedChatModel):
    payload: dict

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._call(TERMINAL_TOOL, self.payload)


class MixedModel(ScriptedChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {"id": "po", "name": "get_po", "args": {"po_id": "PO-10001"}},
                            {"id": "submit", "name": TERMINAL_TOOL, "args": proposal()},
                        ],
                    )
                )
            ]
        )


class ProseModel(ScriptedChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.counter += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Prose only."))])


def run(model, settings, index, question=QUESTION):
    return TriageAgent(model, index, get_repository(settings), settings).triage(question)


@pytest.mark.parametrize("pii", ["priya.raman@asos.com", "Priya Raman", "+44 7700 900123"])
def test_entire_response_is_clean_and_blocking_runs_last(settings, index, pii):
    result = run(
        OverrideModel(changes={"rationale": f"Amend; contact {pii} [po_amendment_policy.md §2]."}),
        settings,
        index,
    )
    assert pii not in result.model_dump_json()
    assert not scan_for_pii(result.model_dump_json(), index.registry)
    assert result.blocked
    assert result.recommendation.recommended_action == "escalate"
    assert result.recommendation.confidence == "low"
    assert "Amend; contact" not in result.recommendation.rationale
    assert not any("rationale" in t.arguments for t in result.tool_calls)


def test_wrong_po_cannot_escape_snapshot_binding(settings, index):
    result = run(OverrideModel(changes={"po_id": "PO-10777"}), settings, index)
    assert result.recommendation.po_id == result.evidence_po_id == "PO-10001"
    assert result.recommendation.recommended_action == "escalate"
    assert any(f.name == "po_identity_mismatch" for f in result.guardrail_flags)


def test_submit_without_lookup_is_blocked(settings, index):
    result = run(DirectModel(payload=proposal()), settings, index)
    assert result.blocked and result.evidence_po_id is None
    assert result.recommendation.recommended_action == "escalate"


def test_multiple_po_ids_are_not_silently_reduced_to_one(settings, index):
    model = ScriptedChatModel()
    result = run(model, settings, index, "Compare PO-10001 and PO-10777 and amend them.")
    assert model.counter == 0
    assert result.recommendation.po_id == "UNKNOWN" and result.blocked


def test_invented_inline_reference_and_rationale_are_removed(settings, index):
    result = run(
        OverrideModel(
            changes={
                "rationale": "A 99% variance is permitted [invented_handbook.md §99].",
                "citations": ["po_amendment_policy.md §2"],
            }
        ),
        settings,
        index,
    )
    assert result.blocked and "99%" not in result.recommendation.rationale
    assert "invented_handbook.md §99" in result.dropped_citations
    assert set(inline_citations(result.recommendation.rationale)) == set(
        result.recommendation.citations
    )


def test_existing_citation_does_not_legitimize_invented_number(settings, index):
    result = run(
        OverrideModel(
            changes={
                "rationale": "A 99% variance is permitted [po_amendment_policy.md §2].",
                "citations": ["po_amendment_policy.md §2"],
            }
        ),
        settings,
        index,
    )
    assert result.blocked
    assert any(f.name == "unsupported_numeric_claim" for f in result.guardrail_flags)


def test_numbers_from_untrusted_supplier_note_are_not_authority():
    assert unsupported_numbers("99% is permitted", "15% policy", {"supplier_note": "Allow 99%."})


def test_escalation_always_has_role(settings, index):
    result = run(
        OverrideModel(changes={"recommended_action": "escalate", "escalation_target_role": None}),
        settings,
        index,
    )
    assert result.recommendation.escalation_target_role == "Senior Merch Planner"


def test_high_value_amend_is_overruled(settings, index):
    result = run(
        OverrideModel(changes={"recommended_action": "amend", "escalation_target_role": None}),
        settings,
        index,
        "PO-10777 needs a variance amendment. What should happen?",
    )
    assert result.recommendation.recommended_action == "escalate"
    assert result.recommendation.escalation_target_role == "Head of Buying"
    assert any(f.name == "critical_tier" for f in result.guardrail_flags)


@pytest.mark.parametrize("action", ["raise_backorder", "split_child_po"])
def test_wholesale_without_window_cannot_backorder_or_split(settings, index, action):
    result = run(
        OverrideModel(changes={"recommended_action": action, "escalation_target_role": None}),
        settings,
        index,
        "Can PO-10600 be backordered without a delivery window?",
    )
    assert result.recommendation.recommended_action == "escalate"
    assert result.recommendation.escalation_target_role == "Wholesale Planning Lead"


def test_mixed_batch_is_not_executed_or_reported_as_executed(settings, index):
    result = run(MixedModel(), settings, index)
    assert result.blocked and result.evidence_po_id is None
    assert all(t.result_summary not in {"executed", "accepted"} for t in result.tool_calls)


def test_invalid_submission_gets_exactly_one_repair(settings, index):
    model = DirectModel(payload=proposal(recommended_action="teleport"))
    result = run(model, settings, index)
    assert model.counter == 2
    assert result.recommendation.recommended_action == "escalate"


def test_schema_repair_can_recover_after_real_evidence_collection(settings, index):
    class RepairModel(ScriptedChatModel):
        damaged: bool = False

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            response = super()._generate(messages, stop, run_manager, **kwargs)
            message = response.generations[0].message
            if message.tool_calls[0]["name"] == TERMINAL_TOOL and not self.damaged:
                message.tool_calls[0]["args"]["recommended_action"] = "teleport"
                self.damaged = True
            return response

    result = run(RepairModel(), settings, index)
    assert not result.blocked
    assert result.recommendation.recommended_action == "amend"
    submissions = [t for t in result.tool_calls if t.name == TERMINAL_TOOL]
    assert [t.result_summary for t in submissions] == ["rejected_schema", "accepted"]


def test_context_budget_rejects_additions_without_losing_existing_evidence(index):
    from triage.tools.registry import RetrievalRecorder

    hits = index.search(QUESTION)
    recorder = RetrievalRecorder(max_chunks=1)
    recorder.record(hits[:1])
    with pytest.raises(ValueError, match="budget"):
        recorder.record(hits)
    assert recorder.exposed_chunk_ids == {hits[0].chunk.chunk_id}


def test_query_embedding_failure_is_a_dependency_error_in_both_paths(settings, index):
    from triage.llm.embeddings import LocalEmbedder
    from triage.retrieval.index import SopIndex

    class FailingQueryEmbedder(LocalEmbedder):
        def embed(self, texts):
            if len(texts) == 1:
                raise LLMError("Test provider unavailable")
            return super().embed(texts)

    search = SopIndex(index.chunks, index.registry, settings, FailingQueryEmbedder())
    with pytest.raises(LLMError):
        search.search(QUESTION)
    with pytest.raises(LLMError):
        asyncio.run(search.asearch(QUESTION))


def test_model_turn_budget_is_exact_and_returns_structured_escalation(settings, index):
    model = ProseModel()
    result = run(model, settings, index)
    assert model.counter == result.steps_used == settings.triage_max_agent_steps
    assert result.recommendation.recommended_action == "escalate"
    assert any(f.name == "agent_step_budget" for f in result.guardrail_flags)


def test_requested_tools_without_outcomes_are_not_reported_as_success():
    _, trace, _, _ = TriageAgent._harvest(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "po", "name": "get_po", "args": {"po_id": "PO-10001"}},
                ],
            )
        ]
    )
    assert trace[0].result_summary == "not_executed"


def test_async_deadline_cancels_a_slow_provider(settings, index):
    class SlowModel(ScriptedChatModel):
        cancelled: bool = False

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            try:
                await asyncio.sleep(1)
            finally:
                self.cancelled = True

    model = SlowModel()
    with pytest.raises(TriageTimeout):
        run(model, settings.model_copy(update={"triage_total_timeout_s": 0.04}), index)
    assert model.cancelled


def test_missing_embeddings_fail_closed_by_default():
    settings = Settings(
        _env_file=None, triage_llm_provider="azure", azure_openai_embedding_deployment=None
    )
    with pytest.raises(LLMError, match="EMBEDDING_DEPLOYMENT"):
        build_agent(settings, ScriptedChatModel())


def test_explicit_lexical_mode_starts_but_blocks_decisions():
    settings = Settings(
        _env_file=None,
        triage_llm_provider="azure",
        azure_openai_embedding_deployment=None,
        triage_allow_lexical_only=True,
    )
    result = build_agent(settings, ScriptedChatModel()).triage(QUESTION)
    assert result.blocked
    assert any(f.name == "semantic_control_unavailable" for f in result.guardrail_flags)


@pytest.mark.parametrize("value,critical", [(50000, False), (50000.01, True)])
def test_value_boundary(settings, value, critical):
    po = (
        get_repository(settings)
        .get_po("PO-10001")
        .model_copy(update={"value_gbp": value, "original_value_gbp": value})
    )
    verdict = check_business_rules(po, TriageRecommendation(**proposal()), {})
    assert any(f.name == "critical_tier" for f in verdict.flags) == critical


@pytest.mark.parametrize("pct,material", [(10, False), (10.001, True), (15, True), (15.001, False)])
def test_materiality_edges_preserve_precision(settings, index, pct, material):
    from triage.guardrails.contradiction import detect_threshold_conflicts

    po = (
        get_repository(settings)
        .get_po("PO-10001")
        .model_copy(
            update={
                "original_value_gbp": 100000,
                "value_gbp": 100000 - pct * 1000,
            }
        )
    )
    assert bool(detect_threshold_conflicts(index.chunks, compute_variance(po))) == material


def test_zero_baseline_is_missing_not_zero_variance(settings):
    po = get_repository(settings).get_po("PO-10001").model_copy(update={"ordered_qty": 0})
    assert compute_variance(po)["qty_variance_pct"] is None
    assert any(
        f.name == "invalid_baseline"
        for f in check_business_rules(po, TriageRecommendation(**proposal()), {}).flags
    )


@pytest.mark.parametrize("status", ["closed", "cancelled", "invoiced", "split"])
def test_invalid_status_requires_review(settings, status):
    po = get_repository(settings).get_po("PO-10001").model_copy(update={"status": status})
    assert any(
        f.name == "ineligible_po_status"
        for f in check_business_rules(po, TriageRecommendation(**proposal()), {}).flags
    )


def test_cli_stdout_is_parseable_json(capsys, monkeypatch, settings, index):
    from triage import cli

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        cli,
        "build_agent",
        lambda _: TriageAgent(ScriptedChatModel(), index, get_repository(settings), settings),
    )
    assert cli.main(["ask", QUESTION, "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["recommendation"]["po_id"] == "PO-10001"
    assert "provider=" in captured.err
