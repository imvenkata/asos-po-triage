"""Typed configuration. Single source of truth for every tunable value.

Nothing in this codebase reads os.environ directly; everything goes through
Settings so that configuration is discoverable, typed, and overridable in tests.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

Provider = Literal["azure", "openai", "scripted"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- provider ---------------------------------------------------------
    triage_llm_provider: Provider = "azure"

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str | None = None

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- paths ------------------------------------------------------------
    corpus_dir: Path = REPO_ROOT / "corpus"
    data_dir: Path = REPO_ROOT / "data"

    # --- retrieval --------------------------------------------------------
    triage_retrieval_top_k: int = 6
    triage_rrf_k: int = 60
    # Minimum cosine similarity for the best retrieved chunk. Applies ONLY where
    # a real embedding model produced the vectors. UNCALIBRATED: a plausible
    # starting point for text-embedding-3-small, to be set from a labelled
    # retrieval set rather than by intuition. See WRITEUP.md.
    triage_min_semantic_similarity: float = 0.30

    # --- agent ------------------------------------------------------------
    triage_max_agent_steps: int = 6
    triage_temperature: float = 0.0
    triage_request_timeout_s: float = 60.0

    log_level: str = Field(default="INFO")

    @property
    def embeddings_available(self) -> bool:
        """Whether SOME embedder will be available - including the scripted
        double's non-semantic one. Use `semantic_retrieval_configured` when the
        question is whether embeddings actually carry meaning."""
        if self.triage_llm_provider == "azure":
            return bool(self.azure_openai_embedding_deployment and self.azure_openai_api_key)
        if self.triage_llm_provider == "openai":
            return bool(self.openai_api_key)
        return True  # deterministic local embedder, not a semantic model

    @property
    def semantic_retrieval_configured(self) -> bool:
        return (
            self.triage_llm_provider in ("azure", "openai") and self.embeddings_available
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
