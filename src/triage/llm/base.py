"""Provider-neutral LLM interface.

The agent depends on this Protocol, never on a vendor SDK. Swapping Azure OpenAI
for OpenAI, Bedrock, or a local model is a new adapter and one config value - no
change to the agent, the guardrails, or the evals. This is the "build once,
scale many times" seam.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    """Raised for provider failures the caller is expected to surface, not swallow."""


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
