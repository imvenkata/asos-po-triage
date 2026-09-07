"""Adapter for Azure OpenAI and OpenAI. Both speak the same Chat Completions
surface, so one adapter covers both; only client construction differs."""
from __future__ import annotations

import json
from typing import Any

from ..config import Settings
from ..logging_setup import get_logger
from .base import LLMError, LLMResponse, ToolCall

log = get_logger("llm.openai")


class OpenAICompatibleClient:
    provides_semantic_embeddings = True

    def __init__(self, settings: Settings) -> None:
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("openai package not installed - run: pip install -e .") from exc

        self._settings = settings
        # Reasoning-family deployments reject any temperature but the default.
        # Probed on first use rather than assumed from the deployment name,
        # because deployment names are arbitrary and carry no model metadata.
        self._temperature_supported = True
        if settings.triage_llm_provider == "azure":
            missing = [
                k
                for k, v in {
                    "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
                    "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
                }.items()
                if not v
            ]
            if missing:
                raise LLMError(
                    f"Azure OpenAI selected but {', '.join(missing)} not set. "
                    "Copy .env.example to .env, or run with TRIAGE_LLM_PROVIDER=scripted."
                )
            self._client = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                timeout=settings.triage_request_timeout_s,
            )
            self._chat_model = settings.azure_openai_chat_deployment
            self._embed_model = settings.azure_openai_embedding_deployment
            self.name = f"azure:{self._chat_model}"
            self._provider_label = "azure"
        else:
            if not settings.openai_api_key:
                raise LLMError("OPENAI_API_KEY not set.")
            self._client = OpenAI(
                api_key=settings.openai_api_key, timeout=settings.triage_request_timeout_s
            )
            self._chat_model = settings.openai_chat_model
            self._embed_model = settings.openai_embedding_model
            self.name = f"openai:{self._chat_model}"
            self._provider_label = "openai"

    @staticmethod
    def _is_temperature_rejection(exc: Exception) -> bool:
        text = str(exc).lower()
        return "temperature" in text and (
            "unsupported_value" in text or "does not support" in text
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self._chat_model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if self._temperature_supported and self._settings.triage_temperature is not None:
            kwargs["temperature"] = self._settings.triage_temperature

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._temperature_supported and self._is_temperature_rejection(exc):
                # Capability drift, not a config error: this deployment pins
                # temperature to its default. Retry once without it and remember.
                # Recorded loudly because it weakens run-to-run determinism, which
                # the eval suite depends on - see WRITEUP.md.
                log.warning(
                    "%s rejects an explicit temperature; retrying without it. "
                    "Sampling is now at the model default, so runs are less "
                    "reproducible than temperature=0 would give.",
                    self.name,
                )
                self._temperature_supported = False
                kwargs.pop("temperature", None)
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as retry_exc:
                    raise LLMError(
                        f"{self.name} chat call failed: {retry_exc}"
                    ) from retry_exc
            else:
                # Surface, never swallow. A silent fallback here would produce a
                # confident-looking recommendation with no model behind it.
                raise LLMError(f"{self.name} chat call failed: {exc}") from exc

        choice = resp.choices[0]
        calls: list[ToolCall] = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                log.warning("tool_call %s had unparseable arguments; passing empty dict", tc.function.name)
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return LLMResponse(
            text=choice.message.content,
            tool_calls=calls,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._embed_model:
            raise LLMError(
                "No embedding deployment configured. Set "
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT, or retrieval will run lexical-only."
            )
        try:
            resp = self._client.embeddings.create(model=self._embed_model, input=texts)
        except Exception as exc:
            raise LLMError(
                f"{self._provider_label} embedding deployment "
                f"'{self._embed_model}' failed: {exc}"
            ) from exc
        return [d.embedding for d in resp.data]
