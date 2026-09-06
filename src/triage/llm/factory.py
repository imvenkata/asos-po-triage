from __future__ import annotations

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from .base import LLMClient

log = get_logger("llm.factory")


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.triage_llm_provider == "scripted":
        from .scripted import ScriptedLLMClient

        log.warning(
            "Using the SCRIPTED offline double. Results exercise orchestration and "
            "guardrails only - they are not evidence of model reasoning quality."
        )
        return ScriptedLLMClient()

    from .openai_compatible import OpenAICompatibleClient

    return OpenAICompatibleClient(settings)
