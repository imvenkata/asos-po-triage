"""Embedding clients.

`SopIndex` depends on this narrow interface rather than a vendor SDK, so the
retrieval layer is unaffected by which provider is configured.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import Settings, get_settings
from ..logging_setup import get_logger
from .base import LLMError
from .local_embeddings import embed_texts

log = get_logger("llm.embeddings")


@runtime_checkable
class Embedder(Protocol):
    name: str
    # False where the vectors are a deterministic hashing trick rather than a
    # semantic model. The grounding guardrail reads this and reports itself
    # inactive instead of emitting a similarity it cannot actually measure.
    provides_semantic_embeddings: bool

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    """Deterministic hashed bag-of-words. For tests and offline runs only."""

    name = "local:hashed-bow"
    provides_semantic_embeddings = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)


class AzureEmbedder:
    provides_semantic_embeddings = True

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import AzureOpenAIEmbeddings

        if not settings.azure_openai_embedding_deployment:
            raise LLMError(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT not set. Retrieval will run "
                "lexical-only and the grounding guardrail will be inactive."
            )
        self._client = AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_embedding_deployment,
        )
        self.name = f"azure:{settings.azure_openai_embedding_deployment}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._client.embed_documents(texts)
        except Exception as exc:
            raise LLMError(f"{self.name} embedding call failed: {exc}") from exc


class OpenAIEmbedder:
    provides_semantic_embeddings = True

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY not set.")
        self._client = OpenAIEmbeddings(
            api_key=settings.openai_api_key, model=settings.openai_embedding_model
        )
        self.name = f"openai:{settings.openai_embedding_model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._client.embed_documents(texts)
        except Exception as exc:
            raise LLMError(f"{self.name} embedding call failed: {exc}") from exc


def build_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.triage_llm_provider == "azure":
        return AzureEmbedder(settings)
    if settings.triage_llm_provider == "openai":
        return OpenAIEmbedder(settings)
    return LocalEmbedder()
