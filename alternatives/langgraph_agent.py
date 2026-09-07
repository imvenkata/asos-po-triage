#!/usr/bin/env python3
"""The same triage agent, orchestrated with LangGraph instead of a hand-written loop.

WHY THIS FILE EXISTS
--------------------
WRITEUP.md argues that for this system a framework would have replaced the least
interesting part and given nothing for the most interesting part. This file is
that argument made checkable rather than asserted. It produces the same
`TriageResult` from the same corpus, tools and guardrails, and can be run against
the same eval suite:

    python evals/run_evals.py --agent langgraph

WHAT CHANGED, AND WHAT DID NOT
------------------------------
Changed - orchestration only:
    triage/agent.py::_run_loop      (~105 lines, hand-written while-loop)
    -> a LangGraph StateGraph with two nodes and one conditional edge (below)

Unchanged - imported here verbatim, not reimplemented:
    triage/policy_gate.py           the whole guardrail layer
    triage/retrieval/               hybrid retrieval + conflict closure
    triage/tools/po_tools.py        get_po / get_forecast, including the
                                    pre-computed variance block
    triage/models.py                the output contract
    triage/prompts.py               the system prompt

That asymmetry is the entire point. The framework owns the loop; it has no
opinion about whether a citation is real, whether two SOP sections contradict
each other, or whether a name leaked into the answer. Those 177 lines get
written either way.

WHAT LANGGRAPH GENUINELY BUYS
-----------------------------
Honest accounting, because the interview answer is weaker if it only lists wins:
  + The loop becomes a declarative graph. Adding a node (a re-ranker, a
    human-approval pause) is an edge change rather than surgery on a while-loop.
  + `ToolNode` handles tool dispatch, argument parsing and error-to-message
    conversion - all of which triage/tools/registry.py does by hand.
  + Checkpointing, streaming, and interrupt/resume come free. For a triage
    workflow that should pause for planner approval, `interrupt_before` is a
    one-line change; in the hand-written loop it is a redesign.
  + LangSmith tracing with no instrumentation work.
  - You inherit the abstraction. When the graph misbehaves you are debugging
    through someone else's control flow.
  - Three dependencies (langgraph, langchain-core, langchain-openai) and their
    version churn, for a loop that is ~100 lines.

WHEN I WOULD PICK IT
--------------------
For one agent in a take-home: no - it hides the thing being assessed.
For a fleet of agents across squads: yes - one loop implementation, one tracing
story, one place to change a shared safety policy. That is what "build once,
scale many times" actually requires.

RUNNING IT
----------
    pip install -e ".[langgraph]"
    python alternatives/langgraph_agent.py "PO-10342 is 12% under. Can I amend it?"
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import StructuredTool  # noqa: E402
from langchain_openai import AzureChatOpenAI, ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from pydantic import BaseModel, Field, ValidationError  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from triage.config import Settings, get_settings  # noqa: E402
from triage.data_access import PoRepository, get_repository  # noqa: E402
from triage.guardrails.contradiction import detect_threshold_conflicts  # noqa: E402
from triage.llm.base import LLMError  # noqa: E402
from triage.logging_setup import get_logger  # noqa: E402
from triage.models import (  # noqa: E402
    RetrievedChunk,
    ToolInvocation,
    TriageRecommendation,
    TriageResult,
)
from triage.policy_gate import apply_policy_gate  # noqa: E402
from triage.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    format_conflict_block,
    format_context_block,
)
from triage.retrieval.index import SopIndex, build_index  # noqa: E402
from triage.tools.po_tools import compute_variance, get_forecast, get_po  # noqa: E402

log = get_logger("langgraph")

TERMINAL_TOOL = "submit_recommendation"


# ---------------------------------------------------------------------------
# 1. Graph state
# ---------------------------------------------------------------------------
class TriageState(TypedDict):
    """State threaded through the graph.

    `add_messages` is LangGraph's reducer: each node returns only the messages it
    wants to append and the reducer merges them into the running transcript. In
    the hand-written loop this is the `messages.append(...)` calls scattered
    through `_run_loop` - the same bookkeeping, moved into the framework.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# 2. Tool argument schemas
# ---------------------------------------------------------------------------
# LangChain derives the JSON schema the model sees from these Pydantic models.
# triage/tools/registry.py hand-writes the same schemas as dicts. This is a
# genuine ergonomic win - the schema cannot drift from the function signature.


class GetPoArgs(BaseModel):
    po_id: str = Field(description="Purchase order id, e.g. PO-10342")


class GetForecastArgs(BaseModel):
    sku: str = Field(description="SKU, e.g. SKU-WW-8820")


class SearchSopsArgs(BaseModel):
    query: str = Field(description="Natural-language policy question.")


class SubmitRecommendationArgs(BaseModel):
    """Mirrors TriageRecommendation exactly - this is how structured output is
    forced. The model cannot answer except by calling this tool with a payload
    that validates."""

    po_id: str
    recommended_action: str = Field(
        description="One of: split_child_po, amend, firm_planned_order, "
        "raise_backorder, escalate"
    )
    rationale: str = Field(
        description="Why. Put citations inline in square brackets at the point "
        "they support the claim."
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Citation strings copied verbatim from search_sops results.",
    )
    confidence: str = Field(description="high, medium or low")
    escalation_target_role: str | None = Field(
        default=None,
        description="The ROLE only - never a person's name or email address. "
        "Null when no escalation is required.",
    )


# ---------------------------------------------------------------------------
# 3. Retrieval bookkeeping
# ---------------------------------------------------------------------------
class RetrievalRecorder:
    """Records every chunk the model has actually been shown.

    The policy gate validates citations by exact set membership against this, so
    something has to track it. `ToolNode` executes tools but keeps no
    domain-specific record, so this is the same closure-over-mutable-state
    pattern as ToolRegistry - the framework does not remove the need for it.
    """

    def __init__(self) -> None:
        self.exposed_chunk_ids: set[str] = set()
        self.retrieved: list[RetrievedChunk] = []

    def record(self, hits: list[RetrievedChunk]) -> None:
        self.exposed_chunk_ids |= {h.chunk.chunk_id for h in hits}
        self.retrieved.extend(hits)


def build_tools(
    repo: PoRepository, index: SopIndex, recorder: RetrievalRecorder, top_k: int
) -> list[StructuredTool]:
    """Wrap the existing tool functions as LangChain tools.

    Note what is NOT happening: no tool logic is reimplemented. `get_po` still
    returns the pre-computed variance block from triage/tools/po_tools.py, so
    the model still never does arithmetic.
    """

    def _get_po(po_id: str) -> str:
        return json.dumps(get_po(repo, po_id), default=str)

    def _get_forecast(sku: str) -> str:
        return json.dumps(get_forecast(repo, sku), default=str)

    def _search_sops(query: str) -> str:
        hits = index.search(query, top_k)
        recorder.record(hits)
        return json.dumps(
            {
                "query": query,
                "retrieval_mode": index.mode,
                "results": [
                    {
                        "citation": h.chunk.chunk_id,
                        "heading": h.chunk.heading,
                        "score": round(h.score, 5),
                        "text": h.chunk.text,
                    }
                    for h in hits
                ],
            },
            default=str,
        )

    def _submit(**kwargs: Any) -> str:
        # Validation happens in the policy gate, not here - a rejected payload
        # should reach the gate as "no valid submission" rather than raising
        # inside the graph.
        return "accepted"

    return [
        StructuredTool.from_function(
            func=_get_po,
            name="get_po",
            description="Look up a purchase order by id. Returns the PO record plus "
            "pre-computed quantity/value/ETA variance. Always use these computed "
            "figures rather than calculating percentages yourself.",
            args_schema=GetPoArgs,
        ),
        StructuredTool.from_function(
            func=_get_forecast,
            name="get_forecast",
            description="Look up forecast versus actual units for a SKU.",
            args_schema=GetForecastArgs,
        ),
        StructuredTool.from_function(
            func=_search_sops,
            name="search_sops",
            description="Search the merchandising SOPs. Call again with a more "
            "specific query whenever you discover a fact about the PO (channel, "
            "status, parent) that brings a different policy into play. Every result "
            "carries the citation string you must use.",
            args_schema=SearchSopsArgs,
        ),
        StructuredTool.from_function(
            func=_submit,
            name=TERMINAL_TOOL,
            description="Submit the final structured triage recommendation. Call "
            "this exactly once, as your last action.",
            args_schema=SubmitRecommendationArgs,
        ),
    ]


# ---------------------------------------------------------------------------
# 4. The graph
# ---------------------------------------------------------------------------
def build_graph(model, tools: list[StructuredTool], max_steps: int):
    """Two nodes, one conditional edge.

        START -> agent -> (submitted?) -> END
                   ^         |
                   |      (tools)
                   +---------+

    This replaces `_run_loop`. The step ceiling that the hand-written loop
    enforces with `range(1, max_steps + 1)` becomes LangGraph's
    `recursion_limit` at invoke time.
    """
    bound = model.bind_tools(tools)

    def agent_node(state: TriageState) -> dict[str, list[BaseMessage]]:
        """Call the model. Returns only what to append; the reducer merges."""
        return {"messages": [bound.invoke(state["messages"])]}

    def route(state: TriageState) -> str:
        """Decide where to go after the model speaks.

        Three cases, matching the hand-written loop's branches exactly:
          - called submit_recommendation -> done
          - called any other tool        -> execute it, then loop back
          - answered in prose            -> nudge back onto the contract
        """
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return "nudge"
        if any(tc["name"] == TERMINAL_TOOL for tc in last.tool_calls):
            return "submitted"
        return "tools"

    def nudge_node(state: TriageState) -> dict[str, list[BaseMessage]]:
        """The model answered in prose. Push it back onto the schema rather than
        parsing prose - the same choice the hand-written loop makes."""
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
        "agent",
        route,
        {"tools": "tools", "nudge": "nudge", "submitted": END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("nudge", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# 5. The agent
# ---------------------------------------------------------------------------
class LangGraphTriageAgent:
    """Drop-in alternative to triage.agent.TriageAgent.

    Same constructor shape, same `.triage(question) -> TriageResult`, so the
    eval harness can run either without knowing which it has.
    """

    def __init__(
        self,
        index: SopIndex,
        repo: PoRepository,
        settings: Settings | None = None,
    ) -> None:
        self._index = index
        self._repo = repo
        self._settings = settings or get_settings()
        self._model = _build_chat_model(self._settings)
        self.name = f"langgraph:{getattr(self._model, 'deployment_name', None) or getattr(self._model, 'model_name', '?')}"

    # Parity with TriageAgent so the eval runner can treat them identically.
    @property
    def semantic_retrieval_available(self) -> bool:
        return self._index.semantic_scores_meaningful

    @property
    def pii_registry(self):
        return self._index.registry

    def triage(self, question: str) -> TriageResult:
        settings = self._settings
        recorder = RetrievalRecorder()
        tools = build_tools(
            self._repo, self._index, recorder, settings.triage_retrieval_top_k
        )
        graph = build_graph(self._model, tools, settings.triage_max_agent_steps)

        # --- identical pre-flight to the hand-written agent ------------------
        observations = self._observations_for(question)
        seed = self._index.search(question)
        recorder.record(seed)

        context = format_context_block(seed)
        conflicts = detect_threshold_conflicts([h.chunk for h in seed], observations)
        if conflicts:
            context += "\n\n" + format_conflict_block([c.describe() for c in conflicts])

        initial = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                SystemMessage(content=context),
                HumanMessage(content=question),
            ]
        }

        # --- run the graph ---------------------------------------------------
        # recursion_limit is LangGraph's equivalent of the loop's step ceiling.
        # Each agent->tools->agent round trip costs 2, hence the doubling.
        try:
            final = graph.invoke(
                initial,
                config={"recursion_limit": settings.triage_max_agent_steps * 2 + 4},
            )
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            raise LLMError(f"{self.name} graph execution failed: {exc}") from exc

        raw, tool_log, steps, usage = self._harvest(final["messages"])

        # --- identical policy gate, imported not reimplemented ---------------
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
            po_id=_po_id_from(question, tool_log),
        )

    def _observations_for(self, question: str) -> dict[str, float] | None:
        po_id = _po_id_from(question, [])
        po = self._repo.get_po(po_id) if po_id != "UNKNOWN" else None
        if po is None:
            return None
        return {
            k: v for k, v in compute_variance(po).items() if isinstance(v, (int, float))
        }

    @staticmethod
    def _harvest(
        messages: Sequence[BaseMessage],
    ) -> tuple[dict[str, Any] | None, list[ToolInvocation], int, dict[str, int]]:
        """Reconstruct the trace the policy gate expects from the message list.

        The hand-written loop builds this as it goes. Here it is recovered
        afterwards, which is a real cost of delegating the loop: the framework
        owns the transcript, so anything domain-specific has to be read back out
        of it rather than recorded at the point it happened.
        """
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
                if call["name"] == TERMINAL_TOOL:
                    try:
                        TriageRecommendation(**args)
                        raw = args
                        summary = "accepted"
                    except ValidationError as exc:
                        summary = f"rejected: {exc.error_count()} field error(s)"
                    tool_log.append(
                        ToolInvocation(
                            step=step,
                            name=call["name"],
                            arguments=args,
                            result_summary=summary,
                        )
                    )
                else:
                    tool_log.append(
                        ToolInvocation(
                            step=step,
                            name=call["name"],
                            arguments=args,
                            result_summary="executed",
                        )
                    )
        return raw, tool_log, step, usage


def _po_id_from(question: str, tool_log: list[ToolInvocation]) -> str:
    for call in tool_log:
        if call.name == "get_po" and call.arguments.get("po_id"):
            return str(call.arguments["po_id"]).upper()
    match = re.search(r"PO-\d+", question or "", re.IGNORECASE)
    return match.group(0).upper() if match else "UNKNOWN"


def _build_chat_model(settings: Settings):
    """Construct the LangChain chat model from the same Settings object.

    `temperature` is deliberately left unset. It defaults to None in
    langchain-openai and is therefore not sent - which is what the deployment
    used here requires, since it rejects any explicit temperature. The
    hand-written adapter has to probe for that and retry; here it falls out of
    the default. A small, real win.
    """
    if settings.triage_llm_provider == "azure":
        if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
            raise LLMError(
                "Azure OpenAI not configured. Copy .env.example to .env. "
                "The LangGraph agent has no offline mode - it needs a real model."
            )
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_chat_deployment,
            timeout=settings.triage_request_timeout_s,
        )
    if settings.triage_llm_provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY not set.")
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            timeout=settings.triage_request_timeout_s,
        )
    raise LLMError(
        f"Provider '{settings.triage_llm_provider}' has no LangChain adapter here. "
        "The scripted offline double only drives the hand-written agent; this file "
        "exists to compare real orchestration, so it requires a real provider."
    )


def build_langgraph_agent(settings: Settings | None = None) -> LangGraphTriageAgent:
    """Mirror of triage.agent.build_agent."""
    settings = settings or get_settings()
    from triage.llm.factory import build_llm_client

    # The embedding client is still the project's own - only the CHAT
    # orchestration moves to LangGraph. Retrieval is untouched.
    index = build_index(settings, build_llm_client(settings))
    return LangGraphTriageAgent(index, get_repository(settings), settings)


if __name__ == "__main__":
    from rich.console import Console

    from triage.cli import render
    from triage.logging_setup import setup_logging

    setup_logging("WARNING")
    console = Console()
    question = " ".join(sys.argv[1:]) or (
        "PO-10342 has come in about 12% under the original order value. "
        "Can I amend it in place?"
    )
    console.print(f"[dim]LangGraph orchestration - {question}[/]\n")
    agent = build_langgraph_agent()
    render(agent.triage(question), show_trace=True)
