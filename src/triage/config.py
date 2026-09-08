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
    # Deployment NAME, not model name - they are unrelated on Azure.
    azure_openai_chat_deployment: str = "gpt-5.6-luna"
    azure_openai_embedding_deployment: str | None = None

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- paths ------------------------------------------------------------
    corpus_dir: Path = REPO_ROOT / "corpus"
    data_dir: Path = REPO_ROOT / "data"

    # --- retrieval --------------------------------------------------------
    triage_retrieval_top_k: int = Field(default=6, ge=1, le=30)
    triage_rrf_k: int = Field(default=60, ge=1)
    triage_allow_lexical_only: bool = False
    # Minimum cosine similarity for the best chunk retrieved for the user's
    # question. Applies ONLY where a real embedding model produced the vectors.
    # Calibrated on 15 in-scope + 10 out-of-scope questions against
    # text-embedding-3-small (`python evals/calibrate_threshold.py`).
    #
    # The classes OVERLAP by 0.060 once the in-scope set includes how planners
    # actually type ("why is PO-10600 stuck"), so no threshold is error-free.
    # 0.38 refuses 10/10 out-of-scope and falsely refuses 2/15 in-scope, both
    # terse fragments. That operating point is chosen deliberately: a false
    # refusal costs a planner one re-phrase, a false acceptance ships an
    # ungrounded recommendation on a six-figure PO. The errors are not
    # symmetric, so the threshold should not sit at the symmetric optimum.
    # Re-derive for any other embedding model - the scale is model-specific.
    triage_min_semantic_similarity: float = 0.38

    # --- agent ------------------------------------------------------------
    triage_max_agent_steps: int = Field(default=6, ge=1, le=20)
    triage_request_timeout_s: float = Field(default=60.0, gt=0, le=180)
    triage_total_timeout_s: float = Field(default=120.0, gt=0, le=600)
    triage_max_context_chunks: int = Field(default=28, ge=1, le=200)
    triage_max_tool_calls_per_turn: int = Field(default=4, ge=1, le=8)

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
