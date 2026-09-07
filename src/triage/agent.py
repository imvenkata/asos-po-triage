"""The triage agent: a LangGraph tool-calling loop wrapped in a policy gate.

Division of responsibility, which is the core design decision in this codebase:

  the model      reads retrieved policy, decides, cites, explains
  Python         computes every number, detects policy conflicts, validates every
                 citation, strips PII, and enforces the invariants that must hold
                 regardless of what the model said

The model proposes; the policy gate disposes. A guardrail implemented only as a
prompt instruction is a request, not a control - so each guardrail has a
deterministic enforcement step in `policy_gate.py` that runs after the model has
spoken.

The graph itself is deliberately small:

    START -> agent -> (submitted?) -> END
               ^         |
               |      (tools)
               +---------+

Retrieval is seeded on the user's question before the graph runs, so every
recommendation has a grounding floor, and `search_sops` is also exposed as a tool
so the agent can search again once it discovers what kind of PO it is dealing
with.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError
from typing_extensions import TypedDict

from .config import Settings, get_settings
from .data_access import PoRepository, get_repository
from .guardrails.contradiction import detect_threshold_conflicts
from .llm.base import LLMError
from .llm.chat import build_chat_model, describe
from .logging_setup import get_logger
from .models import RetrievedChunk, ToolInvocation, TriageRecommendation, TriageResult
from .policy_gate import apply_policy_gate
from .prompts import SYSTEM_PROMPT, format_conflict_block, format_context_block
from .retrieval.index import SopIndex, build_index
from .tools.po_tools import compute_variance
from .tools.registry import TERMINAL_TOOL, RetrievalRecorder, build_tools

log = get_logger("agent")


class TriageState(TypedDict):
    """State threaded through the graph.

    `add_messages` is the reducer: each node returns only the messages it wants
    appended and the reducer merges them into the running transcript.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_graph(model: BaseChatModel, tools: list[StructuredTool]):
    """Compile the agent graph. Three nodes, one conditional edge."""
    bound = model.bind_tools(tools)

    def agent_node(state: TriageState) -> dict[str, list[BaseMessage]]:
        return {"messages": [bound.invoke(state["messages"])]}

    def route(state: TriageState) -> str:
        """Where to go after the model speaks.

        - called submit_recommendation -> done
        - called any other tool        -> execute it, loop back
        - answered in prose            -> push it back onto the contract
        """
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return "nudge"
        if any(call["name"] == TERMINAL_TOOL for call in last.tool_calls):
            return "submitted"
        return "tools"

    def nudge_node(state: TriageState) -> dict[str, list[BaseMessage]]:
        log.warning("model answered in prose; nudging back to the schema")
        return {
            "messages": [
                HumanMessage(
                    content="Respond by calling submit_recommendation with the "
                    "structured object. Do not answer in prose."
                )
            ]
        }

    graph = StateGraph(TriageState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("nudge", nudge_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route, {"tools": "tools", "nudge": "nudge", "submitted": END}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("nudge", "agent")
    return graph.compile()


class TriageAgent:
    def __init__(
        self,
        model: BaseChatModel,
        index: SopIndex,
        repo: PoRepository,
        settings: Settings | None = None,
    ) -> None:
        self._model = model
        self._index = index
        self._repo = repo
        self._settings = settings or get_settings()
        self.name = describe(model, self._settings)

    @property
    def semantic_retrieval_available(self) -> bool:
        """Whether the grounding guardrail can actually run in this configuration."""
        return self._index.semantic_scores_meaningful

    @property
    def pii_registry(self):
        """Exposed so the eval harness asserts leakage against the same registry
        the guardrail uses, rather than a second hand-written list."""
        return self._index.registry

    def triage(self, question: str) -> TriageResult:
        settings = self._settings
        recorder = RetrievalRecorder()
        tools = build_tools(
            self._repo, self._index, recorder, settings.triage_retrieval_top_k
        )
        graph = build_graph(self._model, tools)

        # Resolve the PO up front so policy conflicts can be judged against this
        # PO's actual figures rather than treated as universally blocking.
        observations = self._observations_for(question)

        seed = self._index.search(question)
        recorder.record(seed)

        context = format_context_block(seed)
        conflicts = detect_threshold_conflicts([h.chunk for h in seed], observations)
        if conflicts:
            context += "\n\n" + format_conflict_block([c.describe() for c in conflicts])

        initial: TriageState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                SystemMessage(content=context),
                HumanMessage(content=question),
            ]
        }

        # recursion_limit bounds the loop. Each agent->tools->agent round trip
        # costs two steps, hence the doubling.
        try:
            final = graph.invoke(
                initial,
                config={"recursion_limit": settings.triage_max_agent_steps * 2 + 4},
            )
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            raise LLMError(f"{self.name} graph execution failed: {exc}") from exc

        raw, tool_log, steps, usage = self._harvest(final["messages"])

        return apply_policy_gate(
            question=question,
            raw=raw,
            index=self._index,
            settings=settings,
            exposed_chunk_ids=recorder.exposed_chunk_ids,
            retrieved=recorder.retrieved,
            seed=seed,
            observations=observations,
            tool_log=tool_log,
            steps=steps,
            usage=usage,
            provider=self.name,
            po_id=_resolve_po_id(question, tool_log),
        )

    def _observations_for(self, question: str) -> dict[str, float] | None:
        po_id = _po_id_from_question(question)
        po = self._repo.get_po(po_id) if po_id else None
        if po is None:
            return None
        return {
            k: v for k, v in compute_variance(po).items() if isinstance(v, (int, float))
        }

    @staticmethod
    def _harvest(
        messages: Sequence[BaseMessage],
    ) -> tuple[dict[str, Any] | None, list[ToolInvocation], int, dict[str, int]]:
        """Recover the submission and the trace from the finished transcript."""
        raw: dict[str, Any] | None = None
        tool_log: list[ToolInvocation] = []
        usage: dict[str, int] = {}
        step = 0

        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            step += 1
            meta = msg.usage_metadata or {}
            if meta:
                usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + meta.get("input_tokens", 0)
                usage["completion_tokens"] = usage.get("completion_tokens", 0) + meta.get("output_tokens", 0)
                usage["total_tokens"] = usage.get("total_tokens", 0) + meta.get("total_tokens", 0)

            for call in msg.tool_calls or []:
                args = call.get("args") or {}
                summary = "executed"
                if call["name"] == TERMINAL_TOOL:
                    try:
                        TriageRecommendation(**args)
                        raw = args
                        summary = "accepted"
                    except ValidationError as exc:
                        summary = f"rejected: {exc.error_count()} field error(s)"
                        log.warning("submission rejected by schema: %s", summary)
                tool_log.append(
                    ToolInvocation(
                        step=step,
                        name=call["name"],
                        arguments=args,
                        result_summary=summary,
                    )
                )
        return raw, tool_log, step, usage


def _po_id_from_question(question: str) -> str | None:
    match = re.search(r"PO-\d+", question or "", re.IGNORECASE)
    return match.group(0).upper() if match else None


def _resolve_po_id(question: str, tool_log: list[ToolInvocation]) -> str:
    for call in tool_log:
        if call.name == "get_po" and call.arguments.get("po_id"):
            return str(call.arguments["po_id"]).upper()
    return _po_id_from_question(question) or "UNKNOWN"


def build_agent(
    settings: Settings | None = None, model: BaseChatModel | None = None
) -> TriageAgent:
    settings = settings or get_settings()
    from .llm.embeddings import build_embedder

    index = build_index(settings, build_embedder(settings))
    return TriageAgent(
        model or build_chat_model(settings), index, get_repository(settings), settings
    )


__all__ = ["TriageAgent", "build_agent", "build_graph", "LLMError"]
