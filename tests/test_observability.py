import argparse
import asyncio
import json
import logging

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import ValidationError

from triage import api, cli
from triage.agent import TriageAgent
from triage.config import Settings
from triage.data_access import get_repository
from triage.llm.base import LLMError, TriageTimeout
from triage.llm.chat import build_chat_model
from triage.llm.scripted_chat import ScriptedChatModel
from triage.models import StageMeasurement
from triage.token_budget import TokenBudget, estimate_input_tokens

QUESTION = "PO-10001 has a minor variance. Can I amend it in place?"


class MeasuredModel(ScriptedChatModel):
    reported_input: int = 100
    reported_output: int = 20
    missing_usage: bool = False
    finish_reason: str = "stop"
    output_limits: list[int] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.output_limits.append(kwargs["max_completion_tokens"])
        result = super()._generate(messages, stop, run_manager, **kwargs)
        message = result.generations[0].message
        message.usage_metadata = (
            None
            if self.missing_usage
            else {
                "input_tokens": self.reported_input,
                "output_tokens": self.reported_output,
                "total_tokens": self.reported_input + self.reported_output,
            }
        )
        message.response_metadata = {"finish_reason": self.finish_reason}
        return result


def agent(settings, index, model=None):
    return TriageAgent(model or MeasuredModel(), index, get_repository(settings), settings)


def events(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("triage"), "propagate", True)
    caplog.set_level(logging.INFO, logger="triage.telemetry")


def log_records(caplog):
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "triage.telemetry"]


def test_timings_and_provider_usage_are_per_call(settings, index):
    model = MeasuredModel()
    result = agent(settings, index, model).triage(QUESTION)
    telemetry = result.telemetry
    assert telemetry.outcome == "completed"
    assert telemetry.duration_ms >= sum(s.duration_ms for s in telemetry.stages) - 0.1
    assert {s.stage for s in telemetry.stages} == {"retrieval", "model", "tool", "validation"}
    calls = [s for s in telemetry.stages if s.provider_called]
    assert len(calls) == result.steps_used == 3
    assert all(s.input_tokens == 100 and s.output_tokens == 20 for s in calls)
    assert all(s.usage_source == "provider" and s.estimated_input_tokens > 0 for s in calls)
    assert telemetry.chat_tokens_charged == result.usage["total_tokens"] == 360
    assert telemetry.usage_complete
    assert model.output_limits == [settings.triage_max_output_tokens] * 3
    assert len(telemetry.prompt_sha256) == len(telemetry.policy_sha256) == 64


@pytest.mark.parametrize("field", ["triage_max_total_chat_tokens", "triage_max_input_tokens"])
def test_budget_refuses_before_call_and_does_not_trim_evidence(settings, index, field):
    model = MeasuredModel()
    result = agent(settings.model_copy(update={field: 1}), index, model).triage(QUESTION)
    assert model.counter == result.steps_used == 0
    assert result.blocked and result.recommendation.recommended_action == "escalate"
    assert result.recommendation.confidence == "low" and result.review_required
    assert any(f.name == "token_budget_exceeded" for f in result.guardrail_flags)
    assert {"po_amendment_policy.md §2", "po_amendment_policy.md §4"} <= set(
        result.retrieved_chunk_ids
    )
    assert result.telemetry.chat_tokens_charged == 0
    refused = [s for s in result.telemetry.stages if s.stage == "model"]
    assert len(refused) == 1 and refused[0].provider_called is False


def test_total_budget_counts_prior_turns_and_stops_the_next_call(settings, index):
    settings = settings.model_copy(update={"triage_max_input_tokens": 100_000})
    model = MeasuredModel(reported_input=settings.triage_max_total_chat_tokens, reported_output=0)
    result = agent(settings, index, model).triage(QUESTION)
    assert model.counter == result.steps_used == 1
    assert result.evidence_po_id == "PO-10001"  # first admitted lookup did execute
    assert result.blocked
    assert result.telemetry.chat_tokens_charged == settings.triage_max_total_chat_tokens
    assert [s.provider_called for s in result.telemetry.stages if s.stage == "model"] == [
        True,
        False,
    ]


def test_reported_overrun_does_not_execute_the_proposed_tool(settings, index):
    settings = settings.model_copy(update={"triage_max_input_tokens": 100_000})
    result = agent(
        settings,
        index,
        MeasuredModel(
            reported_input=settings.triage_max_total_chat_tokens + 1,
            reported_output=0,
        ),
    ).triage(QUESTION)
    assert result.blocked and result.evidence_po_id is None
    assert result.tool_calls[0].result_summary == "not_executed"
    assert result.telemetry.chat_tokens_charged > result.telemetry.chat_token_budget


def test_missing_usage_keeps_reservations_and_does_not_report_zero(settings, index):
    result = agent(settings, index, MeasuredModel(missing_usage=True)).triage(QUESTION)
    calls = [s for s in result.telemetry.stages if s.provider_called]
    assert result.telemetry.chat_tokens_charged == sum(
        s.estimated_input_tokens + s.output_token_limit for s in calls
    )
    assert not result.telemetry.usage_complete
    assert all(s.input_tokens is None and s.usage_source == "reservation" for s in calls)


def test_estimation_includes_tool_schemas_history_and_multibyte_text():
    messages = [HumanMessage(content="hello")]
    bare = estimate_input_tokens(messages, [])
    assert estimate_input_tokens(messages, [{"description": "policy " * 100}]) > bare
    assert (
        estimate_input_tokens(
            messages + [ToolMessage(content="evidence " * 100, tool_call_id="1")], []
        )
        > bare
    )
    assert estimate_input_tokens([HumanMessage(content="你好" * 100)], []) > bare


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"input_tokens": 1},
        {
            "input_tokens": 50,
            "output_tokens": 20,
            "total_tokens": 10,
        },
    ],
)
def test_partial_or_inconsistent_usage_does_not_refund_the_reservation(settings, usage):
    budget = TokenBudget(settings)
    assert budget.reserve(100)
    record = StageMeasurement(stage="model", operation="chat")
    assert budget.settle(100, usage, record)
    assert budget.charged == 100 + settings.triage_max_output_tokens
    assert not budget.usage_complete and record.usage_source == "reservation"


def test_reservation_allows_exact_boundary_but_not_one_token_more(settings):
    settings = settings.model_copy(
        update={
            "triage_max_total_chat_tokens": 150,
            "triage_max_output_tokens": 50,
            "triage_max_input_tokens": 100,
        }
    )
    assert TokenBudget(settings).reserve(100)
    assert not TokenBudget(settings).reserve(101)
    assert not TokenBudget(
        settings.model_copy(update={"triage_max_total_chat_tokens": 149})
    ).reserve(100)


def test_truncated_model_output_is_not_dispatched(settings, index):
    result = agent(settings, index, MeasuredModel(finish_reason="length")).triage(QUESTION)
    assert result.blocked and result.evidence_po_id is None
    assert result.tool_calls[0].result_summary == "not_executed"
    assert any(f.name == "model_output_truncated" for f in result.guardrail_flags)


def test_unknown_tool_name_and_question_do_not_enter_telemetry(
    settings, index, caplog, monkeypatch
):
    events(caplog, monkeypatch)

    class HostileModel(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return self._call("priya.raman@asos.com", {"query": "PRIVATE_TOOL_ARGUMENT"})

    result = agent(settings, index, HostileModel()).triage(QUESTION + " PRIVATE_QUESTION")
    records = log_records(caplog)
    assert records and all(r["request_id"] == result.telemetry.request_id for r in records)
    payload = json.dumps(records) + result.telemetry.model_dump_json()
    assert not any(
        s in payload
        for s in ("PRIVATE_QUESTION", "PRIVATE_TOOL_ARGUMENT", "priya.raman", "PO-10001")
    )
    assert any(r.get("operation") == "unknown" for r in records)
    assert records[-1]["event"] == "triage_request" and records[-1]["outcome"] == "blocked"


@pytest.mark.parametrize("slow", [False, True])
def test_failure_and_timeout_emit_correlated_safe_summary(
    settings, index, caplog, monkeypatch, slow
):
    events(caplog, monkeypatch)

    class FailedModel(ScriptedChatModel):
        async def _agenerate(self, messages, **kwargs):
            if slow:
                await asyncio.sleep(1)
            raise LLMError("PRIVATE_PROVIDER_BODY priya.raman@asos.com")

    service = agent(
        settings.model_copy(update={"triage_total_timeout_s": 0.03}), index, FailedModel()
    )
    with pytest.raises(TriageTimeout if slow else LLMError) as error:
        service.triage(QUESTION)
    records = log_records(caplog)
    summary = records[-1]
    assert summary["request_id"] == error.value.request_id
    assert summary["outcome"] == ("timeout" if slow else "error")
    assert not summary["usage_complete"] and summary["chat_tokens_charged"] > 0
    assert "PRIVATE_PROVIDER_BODY" not in json.dumps(records) + str(error.value)


def test_concurrent_requests_do_not_share_budget_or_measurements(settings, index):
    service = agent(settings, index)

    async def together():
        return await asyncio.gather(service.atriage(QUESTION), service.atriage(QUESTION))

    first, second = asyncio.run(together())
    assert first.telemetry.request_id != second.telemetry.request_id
    assert first.telemetry.chat_tokens_charged == second.telemetry.chat_tokens_charged == 360
    assert len(first.telemetry.stages) == len(second.telemetry.stages)
    assert first.telemetry.stages is not second.telemetry.stages


def test_caller_cancellation_keeps_reservation_and_emits_summary(
    settings, index, monkeypatch, caplog
):
    events(caplog, monkeypatch)

    async def cancel_after_call_starts():
        started = asyncio.Event()

        class WaitingModel(ScriptedChatModel):
            async def _agenerate(self, messages, **kwargs):
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(agent(settings, index, WaitingModel()).atriage(QUESTION))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_after_call_starts())
    records = log_records(caplog)
    assert records[-1]["outcome"] == "cancelled"
    assert not records[-1]["usage_complete"]
    assert records[-1]["chat_tokens_charged"] > 0
    assert any(r.get("operation") == "chat" and r["status"] == "cancelled" for r in records)


def test_interactive_cli_keeps_requests_on_one_event_loop(settings, index, monkeypatch):
    service = agent(settings, index)
    original = service.atriage
    loops = []

    async def capture_loop(question):
        loops.append(asyncio.get_running_loop())
        return await original(question)

    monkeypatch.setattr(service, "atriage", capture_loop)
    monkeypatch.setattr(cli, "_agent", lambda: service)
    questions = iter([QUESTION, QUESTION, "exit"])
    monkeypatch.setattr(cli.console, "input", lambda _: next(questions))
    monkeypatch.setattr(cli, "render", lambda *args, **kwargs: None)
    assert cli.cmd_repl(argparse.Namespace(trace=False)) == 0
    assert len(loops) == 2 and loops[0] is loops[1]


@pytest.mark.parametrize("provider", ["azure", "openai"])
def test_provider_payload_has_output_cap_and_no_hidden_chat_retry(provider):
    settings = Settings(
        _env_file=None,
        triage_llm_provider=provider,
        openai_api_key="test-key",
        azure_openai_api_key="test-key",
        azure_openai_endpoint="https://example.openai.azure.com/",
        azure_openai_chat_deployment="test-deployment",
        triage_max_output_tokens=321,
    )
    model = build_chat_model(settings)
    payload = model._get_request_payload([HumanMessage(content="hello")])
    assert payload["max_completion_tokens"] == 321 and model.max_retries == 0


def test_api_returns_request_id_on_success(settings, index, monkeypatch):
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "build_agent", lambda _: agent(settings, index))
    with TestClient(api.app) as client:
        response = client.post("/triage", json={"question": QUESTION})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == response.json()["telemetry"]["request_id"]


def test_api_failure_request_id_matches_log(settings, index, monkeypatch, caplog):
    events(caplog, monkeypatch)

    class FailedModel(ScriptedChatModel):
        async def _agenerate(self, messages, **kwargs):
            raise LLMError("PRIVATE_PROVIDER_BODY")

    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "build_agent", lambda _: agent(settings, index, FailedModel()))
    with TestClient(api.app) as client:
        response = client.post("/triage", json={"question": QUESTION})
    assert response.status_code == 502
    assert response.headers["X-Request-ID"] == log_records(caplog)[-1]["request_id"]
    assert "PRIVATE_PROVIDER_BODY" not in response.text


def test_telemetry_rejects_untrusted_operation_labels():
    with pytest.raises(ValidationError):
        StageMeasurement(stage="tool", operation="private@example.com")


@pytest.mark.parametrize(
    "field", ["triage_max_total_chat_tokens", "triage_max_input_tokens", "triage_max_output_tokens"]
)
def test_token_settings_require_positive_limits(field):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: 0})
