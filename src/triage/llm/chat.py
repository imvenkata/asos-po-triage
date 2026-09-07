"""Chat model construction.

Returns a LangChain `BaseChatModel`, which is what the agent graph binds tools
to. Provider selection is the only thing that varies.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from .base import LLMError

log = get_logger("llm.chat")


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()

    if settings.triage_llm_provider == "scripted":
        from .scripted_chat import ScriptedChatModel

        log.warning(
            "Using the SCRIPTED offline double. Results exercise orchestration and "
            "guardrails only - they are not evidence of model reasoning quality."
        )
        return ScriptedChatModel()

    if settings.triage_llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI

        missing = [
            k for k, v in {
                "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
                "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
            }.items() if not v
        ]
        if missing:
            raise LLMError(
                f"Azure OpenAI selected but {', '.join(missing)} not set. Copy "
                ".env.example to .env, or run with TRIAGE_LLM_PROVIDER=scripted."
            )
        # `temperature` is left unset deliberately. Reasoning-family deployments
        # reject any explicit value, and langchain-openai omits the parameter
        # when it is None.
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_chat_deployment,
            timeout=settings.triage_request_timeout_s,
        )

    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise LLMError("OPENAI_API_KEY not set.")
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        timeout=settings.triage_request_timeout_s,
    )


def describe(model: BaseChatModel, settings: Settings) -> str:
    """Short provider label for the result envelope."""
    if settings.triage_llm_provider == "scripted":
        return "scripted:offline-double"
    name = getattr(model, "deployment_name", None) or getattr(model, "model_name", "?")
    return f"{settings.triage_llm_provider}:{name}"
