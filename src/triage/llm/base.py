"""Shared LLM error type."""
from __future__ import annotations


class LLMError(RuntimeError):
    """Provider failures the caller is expected to surface, not swallow."""
