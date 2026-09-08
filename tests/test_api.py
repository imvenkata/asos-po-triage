import pytest
from fastapi.testclient import TestClient

from triage import api
from triage.agent import TriageAgent
from triage.data_access import get_repository
from triage.llm.base import LLMError, TriageTimeout
from triage.llm.scripted_chat import ScriptedChatModel


def test_api_serializes_safe_envelope_and_review_state(monkeypatch, settings, index):
    class LeakingModel(ScriptedChatModel):
        def _decide(self, po, messages):
            return super()._decide(po, messages) | {"rationale": "Contact priya.raman@asos.com."}

    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "build_agent", lambda _: TriageAgent(LeakingModel(), index, get_repository(settings), settings))
    with TestClient(api.app) as client:
        response = client.post("/triage", json={"question": "Amend the minor variance on PO-10001?"})
        assert response.status_code == 200
        assert "priya.raman" not in response.text
        assert response.json()["recommendation"]["recommended_action"] == "escalate"
        assert response.json()["review_required"]


@pytest.mark.parametrize("error,status", [(LLMError("secret provider body"), 502),
                                        (TriageTimeout("secret provider body"), 504)])
def test_errors_do_not_expose_provider_bodies(monkeypatch, settings, error, status, caplog):
    class FailedAgent:
        async def atriage(self, question):
            raise error

    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "build_agent", lambda _: FailedAgent())
    with TestClient(api.app) as client:
        response = client.post("/triage", json={"question": "PO-10001"})
    assert response.status_code == status
    assert "secret provider body" not in response.text + caplog.text
