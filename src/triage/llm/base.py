"""Shared LLM error type."""

from __future__ import annotations


class LLMError(RuntimeError):
    """Provider failures the caller is expected to surface, not swallow."""

    def __init__(self, message: str, *, request_id: str | None = None):
        super().__init__(message)
        self.request_id = request_id


class TriageTimeout(LLMError):
    """The end-to-end request deadline expired."""
