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
    # Fused RRF scores are small by construction (1/(60+rank) summed over two
    # rankers), so this floor is calibrated against that scale, not cosine.
    triage_min_retrieval_score: float = 0.012
    triage_rrf_k: int = 60

    # --- agent ------------------------------------------------------------
    triage_max_agent_steps: int = 6
    triage_temperature: float = 0.0
    triage_request_timeout_s: float = 60.0

    log_level: str = Field(default="INFO")

    @property
    def embeddings_available(self) -> bool:
        if self.triage_llm_provider == "azure":
            return bool(self.azure_openai_embedding_deployment and self.azure_openai_api_key)
        if self.triage_llm_provider == "openai":
            return bool(self.openai_api_key)
        return True  # scripted provider has a deterministic local embedder


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
