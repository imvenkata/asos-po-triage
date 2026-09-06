import pytest

from triage.config import Settings
from triage.llm.scripted import ScriptedLLMClient
from triage.retrieval.index import build_index


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(triage_llm_provider="scripted")


@pytest.fixture(scope="session")
def index(settings):
    return build_index(settings, ScriptedLLMClient())
