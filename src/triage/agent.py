"""Bounded, single-agent investigation followed by deterministic validation."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Annotated, Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError
from typing_extensions import TypedDict

from .config import Settings, get_settings
from .data_access import PoRepository, get_repository
from .llm.base import LLMError, TriageTimeout
from .llm.chat import build_chat_model, describe
from .models import GuardrailFlag, ToolInvocation, TriageRecommendation, TriageResult
from .policy_gate import apply_policy_gate
from .prompts import SYSTEM_PROMPT, format_context_block
from .retrieval.index import SopIndex, build_index
from .tools.registry import TERMINAL_TOOL, RetrievalRecorder, build_tools


class TriageState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    steps: int
    repairs: int
    done: bool
    raw: dict[str, Any] | None
    flags: list[GuardrailFlag]


def build_graph(model: BaseChatModel, tools, settings: Settings | None = None):
    settings = settings or get_settings()
    bound = model.bind_tools(tools)
    by_name = {tool.name: tool for tool in tools}

    async def agent_node(state):
        message = await bound.ainvoke(state["messages"])
        return {"messages": [message], "steps": state.get("steps", 0) + 1}

    async def dispatch(state):
        message = state["messages"][-1]
        calls = message.tool_calls if isinstance(message, AIMessage) else []
        terminal = [c for c in calls if c["name"] == TERMINAL_TOOL]
        if not calls:
            return {"messages": [HumanMessage(content="Call submit_recommendation using the schema, after collecting evidence.")]}
        if len(calls) > settings.triage_max_tool_calls_per_turn or (terminal and len(calls) != 1):
            return {
                "messages": [ToolMessage(content="Rejected: submit alone; tool batch not executed.",
                                         tool_call_id=c["id"], name=c["name"], status="error") for c in calls],
                "flags": state.get("flags", []) + [GuardrailFlag(
                    name="invalid_tool_batch", detail="A terminal or oversized tool batch was rejected.", blocking=True)],
                "done": True,
            }
        if terminal:
            call = terminal[0]
            try:
                rec = TriageRecommendation.model_validate(call["args"])
            except ValidationError as exc:
                # Locations and types only. Never echo invalid input values.
                errors = [{"field": str(e["loc"][0]) if e["loc"] else "object",
                           "type": e["type"]} for e in exc.errors(include_input=False)]
                return {
                    "messages": [ToolMessage(
                        content=json.dumps({"error": "Invalid recommendation schema", "fields": errors}),
                        name=TERMINAL_TOOL, tool_call_id=call["id"], status="error")],
                    "repairs": state.get("repairs", 0) + 1,
                    "done": state.get("repairs", 0) >= 1,
                }
            return {
                "messages": [ToolMessage(content="accepted", name=TERMINAL_TOOL,
                                         tool_call_id=call["id"], status="success")],
                "raw": rec.model_dump(), "done": True,
            }
        results = []
        for call in calls:
            tool = by_name.get(call["name"])
            try:
                if tool is None:
                    raise ValueError("Unknown tool")
                content = await tool.ainvoke(call["args"])
                failed = isinstance(content, str) and isinstance(json.loads(content), dict) and "error" in json.loads(content)
            except LLMError:
                raise
            except Exception as exc:
                content = json.dumps({"error": "Tool execution failed", "category": type(exc).__name__})
                failed = True
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"],
                                       status="error" if failed else "success"))
        return {"messages": results}

    def route(state):
        return END if state.get("done") or state["steps"] >= settings.triage_max_agent_steps else "agent"

    graph = StateGraph(TriageState)
    graph.add_node("agent", agent_node)
    graph.add_node("dispatch", dispatch)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "dispatch")
    graph.add_conditional_edges("dispatch", route)
    return graph.compile()


class TriageAgent:
    def __init__(self, model: BaseChatModel, index: SopIndex, repo: PoRepository,
                 settings: Settings | None = None) -> None:
        self._model, self._index, self._repo = model, index, repo
        self._settings = settings or get_settings()
        self.name = describe(model, self._settings)

    @property
    def semantic_retrieval_available(self) -> bool:
        return self._index.semantic_scores_meaningful

    @property
    def pii_registry(self):
        return self._index.registry

    def triage(self, question: str) -> TriageResult:
        """Synchronous CLI entry point. Async callers should await atriage."""
        return asyncio.run(self.atriage(question))

    async def atriage(self, question: str) -> TriageResult:
        try:
            return await asyncio.wait_for(self._investigate(question),
                                          timeout=self._settings.triage_total_timeout_s)
        except TimeoutError as exc:
            raise TriageTimeout("Triage request deadline exceeded.") from exc
        except LLMError:
            raise
        except Exception as exc:
            # Provider exception text can contain prompts, tokens or contact data.
            raise LLMError(f"Triage dependency failed ({type(exc).__name__}).") from exc

    async def _investigate(self, question: str) -> TriageResult:
        settings = self._settings
        po_ids = list(dict.fromkeys(re.findall(r"\bPO-\d+\b", question, re.IGNORECASE)))
        po_ids = list(dict.fromkeys(p.upper() for p in po_ids))
        requested = po_ids[0] if len(po_ids) == 1 else "UNKNOWN"
        recorder = RetrievalRecorder(settings.triage_max_context_chunks,
                                     requested if requested != "UNKNOWN" else None)
        seed = await self._index.asearch(question)
        recorder.record(seed)
        flags = []
        raw, trace, steps, usage = None, [], 0, {}
        model_metadata = {}
        if len(po_ids) > 1:
            flags.append(GuardrailFlag(name="multiple_po_ids",
                                      detail="Ask about one PO at a time.", blocking=True))
        else:
            graph = build_graph(self._model, build_tools(
                self._repo, self._index, recorder, settings.triage_retrieval_top_k), settings)
            final = await graph.ainvoke(
                {"messages": [SystemMessage(content=SYSTEM_PROMPT),
                              HumanMessage(content=question),
                              HumanMessage(content=format_context_block(seed))],
                 "steps": 0, "repairs": 0, "raw": None, "done": False, "flags": []},
                config={"recursion_limit": settings.triage_max_agent_steps * 2 + 2},
            )
            raw = final.get("raw")
            for msg in final["messages"]:
                if isinstance(msg, AIMessage):
                    model_metadata.update({k: str(v) for k, v in msg.response_metadata.items()
                                           if k in {"model_name", "model", "system_fingerprint"} and v is not None})
            _, trace, steps, usage = self._harvest(final["messages"])
            flags.extend(final.get("flags", []))
            if not final.get("done") and steps >= settings.triage_max_agent_steps:
                flags.append(GuardrailFlag(name="agent_step_budget",
                                          detail="The model turn budget was exhausted.", blocking=True))
        return apply_policy_gate(
            question=question, raw=raw, index=self._index, settings=settings,
            exposed_chunk_ids=recorder.exposed_chunk_ids, retrieved=recorder.retrieved,
            seed=seed, observations=None, tool_log=trace, steps=steps, usage=usage,
            provider=self.name, po_id=requested, po=recorder.po,
            po_payload=recorder.po_payload, forecasts=recorder.forecasts, initial_flags=flags,
            model_metadata=model_metadata,
        )

    @staticmethod
    def _harvest(messages: Sequence[BaseMessage]):
        raw, trace, usage, step = None, [], {}, 0
        # Unique provider call IDs bind completed results to their requests.
        outcomes = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            step += 1
            for source, target in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
                                   ("total_tokens", "total_tokens")):
                usage[target] = usage.get(target, 0) + (msg.usage_metadata or {}).get(source, 0)
            for call in msg.tool_calls or []:
                outcome = outcomes.get(call["id"])
                summary = "not_executed" if outcome is None else (
                    "failed" if outcome.status == "error" else "executed")
                if call["name"] == TERMINAL_TOOL:
                    try:
                        rec = TriageRecommendation.model_validate(call.get("args") or {})
                        if outcome is not None and outcome.status == "success" and outcome.content == "accepted":
                            raw, summary = rec.model_dump(), "accepted"
                    except ValidationError:
                        summary = "rejected_schema"
                trace.append(ToolInvocation(step=step, name=call["name"],
                                            arguments=call.get("args") or {}, result_summary=summary))
        return raw, trace, step, usage


def build_agent(settings: Settings | None = None, model: BaseChatModel | None = None) -> TriageAgent:
    settings = settings or get_settings()
    from .llm.embeddings import build_embedder

    try:
        embedder = build_embedder(settings)
    except LLMError:
        if not settings.triage_allow_lexical_only:
            raise
        embedder = None
    index = build_index(settings, embedder)
    if settings.triage_llm_provider != "scripted" and not index.semantic_scores_meaningful and not settings.triage_allow_lexical_only:
        raise LLMError("A semantic index is required. Lexical-only mode must be explicitly enabled.")
    return TriageAgent(model or build_chat_model(settings), index, get_repository(settings), settings)


__all__ = ["TriageAgent", "build_agent", "build_graph", "LLMError"]
