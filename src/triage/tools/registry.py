"""Tool definitions and retrieval bookkeeping.

Argument schemas are Pydantic models, so the JSON schema the model sees is
derived from the function signature and cannot drift from it.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..data_access import PoRepository
from ..models import RetrievedChunk
from ..retrieval.index import SopIndex
from .po_tools import get_forecast, get_po

TERMINAL_TOOL = "submit_recommendation"

ACTIONS = ["split_child_po", "amend", "firm_planned_order", "raise_backorder", "escalate"]
ROLES = [
    "Merch Planner",
    "Senior Merch Planner",
    "Head of Buying",
    "Wholesale Planning Lead",
    "Supply Chain Risk Lead",
]


class GetPoArgs(BaseModel):
    po_id: str = Field(description="Purchase order id, e.g. PO-10342")


class GetForecastArgs(BaseModel):
    sku: str = Field(description="SKU, e.g. SKU-WW-8820")


class SearchSopsArgs(BaseModel):
    query: str = Field(description="Natural-language policy question.")


class SubmitRecommendationArgs(BaseModel):
    """Mirrors TriageRecommendation. Structured output is forced by making this
    the only way the agent can finish."""

    po_id: str
    recommended_action: str = Field(description=f"One of: {', '.join(ACTIONS)}")
    rationale: str = Field(
        description="Why, referencing the policy you relied on. Put each citation "
        "inline in square brackets at the point it supports the claim."
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Citation strings copied verbatim from search_sops results, "
        "e.g. 'po_amendment_policy.md §3'.",
    )
    confidence: str = Field(description="high, medium or low")
    escalation_target_role: str | None = Field(
        default=None,
        description=f"The ROLE only, one of: {', '.join(ROLES)}. Never a person's "
        "name or email address. Null when no escalation is required.",
    )


class RetrievalRecorder:
    """Every chunk the model has been shown.

    The policy gate validates citations by exact set membership against this, so
    the search tool records as it goes.
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
        # Validation belongs to the policy gate: a malformed payload should
        # arrive there as "no valid submission" rather than raise inside the graph.
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
            description="Look up forecast versus actual units for a SKU. Use when the "
            "question concerns demand impact or re-forecasting.",
            args_schema=GetForecastArgs,
        ),
        StructuredTool.from_function(
            func=_search_sops,
            name="search_sops",
            description="Search the merchandising SOPs. Call this again with a more "
            "specific query whenever you discover a fact about the PO (its channel, "
            "status, or parent) that brings a different policy into play. Every "
            "result carries the citation string you must use.",
            args_schema=SearchSopsArgs,
        ),
        StructuredTool.from_function(
            func=_submit,
            name=TERMINAL_TOOL,
            description="Submit the final structured triage recommendation. Call this "
            "exactly once, as your last action.",
            args_schema=SubmitRecommendationArgs,
        ),
    ]
