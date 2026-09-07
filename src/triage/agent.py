"""The triage agent: bounded tool-calling loop plus a deterministic policy gate.

Division of responsibility, which is the core design decision in this codebase:

  the model      reads retrieved policy, decides, cites, explains
  Python         computes every number, detects policy conflicts, validates every
                 citation, strips PII, and enforces the invariants that must hold
                 regardless of what the model said

The model proposes; the policy gate disposes. A guardrail implemented only as a
prompt instruction is a request, not a control - so each guardrail here has a
deterministic enforcement step that runs after the model has spoken.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .config import Settings, get_settings
from .data_access import PoRepository, get_repository
from .guardrails.contradiction import detect_threshold_conflicts
from .llm.base import LLMClient, LLMError
from .logging_setup import get_logger
from .models import (
    RetrievedChunk,
    ToolInvocation,
    TriageRecommendation,
    TriageResult,
)
from .policy_gate import apply_policy_gate
from .prompts import SYSTEM_PROMPT, format_conflict_block, format_context_block
from .retrieval.index import SopIndex, build_index
from .tools.po_tools import compute_variance
from .tools.registry import ToolRegistry, build_tool_registry

log = get_logger("agent")

TERMINAL_TOOL = "submit_recommendation"


class TriageAgent:
    def __init__(
        self,
        llm: LLMClient,
        index: SopIndex,
        repo: PoRepository,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm
        self._index = index
        self._repo = repo
        self._settings = settings or get_settings()

    @property
    def semantic_retrieval_available(self) -> bool:
        """Whether the grounding guardrail can actually run in this configuration."""
        return self._index.semantic_scores_meaningful

    @property
    def pii_registry(self):
        """Exposed so the eval harness can assert on leakage using the same
        registry the guardrail uses, rather than a second hand-written list."""
        return self._index.registry

    # -- public API --------------------------------------------------------
    def triage(self, question: str) -> TriageResult:
        settings = self._settings
        registry = build_tool_registry(self._repo, self._index, settings.triage_retrieval_top_k)

        # Resolve the PO up front so policy conflicts can be judged against this
        # PO's actual figures rather than treated as universally blocking.
        observations = self._observations_for(question)

        seed = self._index.search(question)
        registry.exposed_chunk_ids |= {h.chunk.chunk_id for h in seed}
        registry.retrieved.extend(seed)

        messages = self._initial_messages(question, seed, observations)
        raw, tool_log, steps, usage = self._run_loop(messages, registry)
        return self._finalise(
            question, raw, registry, tool_log, steps, usage, observations, seed
        )

    def _observations_for(self, question: str) -> dict[str, float] | None:
        po_id = _po_id_from_question(question)
        if not po_id:
            return None
        po = self._repo.get_po(po_id)
        if po is None:
            return None
        return {
            k: v for k, v in compute_variance(po).items() if isinstance(v, (int, float))
        }

    # -- loop --------------------------------------------------------------
    def _initial_messages(
        self,
        question: str,
        seed: list[RetrievedChunk],
        observations: dict[str, float] | None,
    ) -> list[dict[str, Any]]:
        context = format_context_block(seed)
        conflicts = detect_threshold_conflicts([h.chunk for h in seed], observations)
        if conflicts:
            context += "\n\n" + format_conflict_block([c.describe() for c in conflicts])
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]

    def _run_loop(
        self, messages: list[dict[str, Any]], registry: ToolRegistry
    ) -> tuple[dict[str, Any] | None, list[ToolInvocation], int, dict[str, int]]:
        tool_log: list[ToolInvocation] = []
        usage: dict[str, int] = {}
        submitted: dict[str, Any] | None = None
        step = 0

        for step in range(1, self._settings.triage_max_agent_steps + 1):
            response = self._llm.chat(messages, tools=registry.schemas)
            for key, value in (response.usage or {}).items():
                usage[key] = usage.get(key, 0) + value

            if not response.wants_tools:
                # The model answered in prose instead of calling the terminal
                # tool. Push it back onto the contract rather than parsing prose.
                log.warning("step %d: no tool call; nudging back to the schema", step)
                messages.append({"role": "assistant", "content": response.text or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Respond by calling submit_recommendation with the "
                            "structured object. Do not answer in prose."
                        ),
                    }
                )
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            for call in response.tool_calls:
                if registry.is_terminal(call.name):
                    parsed, error = self._validate_submission(call.arguments)
                    if parsed is not None:
                        submitted = parsed
                        messages.append(
                            {"role": "tool", "tool_call_id": call.id, "content": "accepted"}
                        )
                        tool_log.append(
                            ToolInvocation(
                                step=step,
                                name=call.name,
                                arguments=call.arguments,
                                result_summary="accepted",
                            )
                        )
                        break
                    # One bounded repair attempt: hand the validation error back.
                    log.warning("step %d: submission rejected: %s", step, error)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": f"REJECTED - your object failed validation: {error}. "
                            "Call submit_recommendation again with a corrected object.",
                        }
                    )
                    tool_log.append(
                        ToolInvocation(
                            step=step,
                            name=call.name,
                            arguments=call.arguments,
                            result_summary="rejected",
                            error=error,
                        )
                    )
                    continue

                result, error = registry.dispatch(call.name, call.arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
                tool_log.append(
                    ToolInvocation(
                        step=step,
                        name=call.name,
                        arguments=call.arguments,
                        result_summary=_summarise(result),
                        error=error,
                    )
                )

            if submitted is not None:
                break

        return submitted, tool_log, step, usage

    @staticmethod
    def _validate_submission(
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            TriageRecommendation(**arguments)
        except ValidationError as exc:
            return None, "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
        return arguments, None

    # -- deterministic policy gate ----------------------------------------
    def _finalise(
        self,
        question: str,
        raw: dict[str, Any] | None,
        registry: ToolRegistry,
        tool_log: list[ToolInvocation],
        steps: int,
        usage: dict[str, int],
        observations: dict[str, float] | None = None,
        seed: list[RetrievedChunk] | None = None,
    ) -> TriageResult:
        """Hand off to the shared policy gate.

        The gate lives in `policy_gate.py` rather than here because it is not a
        property of this orchestration strategy - the LangGraph implementation in
        `alternatives/` calls the same function with the same arguments.
        """
        return apply_policy_gate(
            question=question,
            raw=raw,
            index=self._index,
            settings=self._settings,
            exposed_chunk_ids=registry.exposed_chunk_ids,
            retrieved=registry.retrieved,
            seed=seed,
            observations=observations,
            tool_log=tool_log,
            steps=steps,
            usage=usage,
            provider=getattr(self._llm, "name", "unknown"),
            po_id=_guess_po_id(question, tool_log),
        )


def _summarise(payload: str, limit: int = 160) -> str:
    flat = " ".join(payload.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _po_id_from_question(question: str) -> str | None:
    match = re.search(r"PO-\d+", question or "", re.IGNORECASE)
    return match.group(0).upper() if match else None


def _guess_po_id(question: str, tool_log: list[ToolInvocation]) -> str:
    for call in tool_log:
        if call.name == "get_po" and call.arguments.get("po_id"):
            return str(call.arguments["po_id"])
    return _po_id_from_question(question) or "UNKNOWN"


def build_agent(settings: Settings | None = None, llm: LLMClient | None = None) -> TriageAgent:
    settings = settings or get_settings()
    if llm is None:
        from .llm.factory import build_llm_client

        llm = build_llm_client(settings)
    index = build_index(settings, llm)
    return TriageAgent(llm, index, get_repository(settings), settings)


__all__ = ["TriageAgent", "build_agent", "LLMError"]
