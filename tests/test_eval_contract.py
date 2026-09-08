import importlib.util
from pathlib import Path

from triage.models import GuardrailFlag, TriageRecommendation, TriageResult

spec = importlib.util.spec_from_file_location("eval_runner", Path(__file__).resolve().parents[1] / "evals/run_evals.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_eval_detects_pii_outside_recommendation(index):
    result = TriageResult(
        recommendation=TriageRecommendation(po_id="PO-10001", recommended_action="escalate",
                                             rationale="Review required.", confidence="low",
                                             escalation_target_role="Senior Merch Planner"),
        guardrail_flags=[GuardrailFlag(name="debug", detail="priya.raman@asos.com")],
    )
    checks = runner.evaluate({}, result, index.registry)
    assert not next(c for c in checks if c.name == "no_pii_leak").passed


def test_every_supported_action_has_a_positive_eval():
    import yaml
    cases = yaml.safe_load(runner.CASES_PATH.read_text())
    actions = {c["expect"]["action"][0] for c in cases if len(c["expect"]["action"]) == 1}
    assert actions == {"amend", "escalate", "split_child_po", "raise_backorder", "firm_planned_order"}


def test_eval_rejects_inconsistent_blocked_response(index):
    result = TriageResult(
        recommendation=TriageRecommendation(po_id="PO-10001", recommended_action="amend",
                                             rationale="Apply it.", confidence="high"),
        guardrail_flags=[GuardrailFlag(name="blocked", detail="Unsafe.", blocking=True)],
    )
    checks = runner.evaluate({}, result, index.registry)
    assert not next(c for c in checks if c.name == "blocking_controls_enforced").passed
