import asyncio
import importlib.util
import sys
from pathlib import Path

from triage.agent import TriageAgent
from triage.data_access import get_repository
from triage.llm.scripted_chat import ScriptedChatModel
from triage.models import GuardrailFlag, TriageRecommendation, TriageResult

spec = importlib.util.spec_from_file_location(
    "eval_runner", Path(__file__).resolve().parents[1] / "evals/run_evals.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_eval_detects_pii_outside_recommendation(index):
    result = TriageResult(
        recommendation=TriageRecommendation(
            po_id="PO-10001",
            recommended_action="escalate",
            rationale="Review required.",
            confidence="low",
            escalation_target_role="Senior Merch Planner",
        ),
        guardrail_flags=[GuardrailFlag(name="debug", detail="priya.raman@asos.com")],
    )
    checks = runner.evaluate({}, result, index.registry)
    assert not next(c for c in checks if c.name == "no_pii_leak").passed


def test_every_supported_action_has_a_positive_eval():
    import yaml

    cases = yaml.safe_load(runner.CASES_PATH.read_text())
    actions = {c["expect"]["action"][0] for c in cases if len(c["expect"]["action"]) == 1}
    assert actions == {
        "amend",
        "escalate",
        "split_child_po",
        "raise_backorder",
        "firm_planned_order",
    }


def test_eval_rejects_inconsistent_blocked_response(index):
    result = TriageResult(
        recommendation=TriageRecommendation(
            po_id="PO-10001", recommended_action="amend", rationale="Apply it.", confidence="high"
        ),
        guardrail_flags=[GuardrailFlag(name="blocked", detail="Unsafe.", blocking=True)],
    )
    checks = runner.evaluate({}, result, index.registry)
    assert not next(c for c in checks if c.name == "blocking_controls_enforced").passed


def test_repeated_evaluations_use_one_event_loop(settings, index, monkeypatch):
    service = TriageAgent(ScriptedChatModel(), index, get_repository(settings), settings)
    original = service.atriage
    loops = []

    async def capture_loop(question):
        loops.append(asyncio.get_running_loop())
        return await original(question)

    monkeypatch.setattr(service, "atriage", capture_loop)
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(runner, "run_metadata", lambda _: {})
    monkeypatch.setattr(runner, "build_agent", lambda _: service)
    monkeypatch.setattr(sys, "argv", ["eval", "--case", "tier1_clean_amend", "--repeat", "2"])
    assert runner.main() == 0
    assert len(loops) == 2 and loops[0] is loops[1]
