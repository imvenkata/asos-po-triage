"""Domain and contract schemas.

`TriageRecommendation` is the exact contract the assessment specifies - no extra
fields. Everything operational (guardrail flags, retrieval telemetry, the agent
trace) lives on the `TriageResult` envelope so the published contract stays
clean and stable while the diagnostics can evolve.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RecommendedAction = Literal[
    "split_child_po", "amend", "firm_planned_order", "raise_backorder", "escalate"
]
Confidence = Literal["high", "medium", "low"]

# Closed set from merch_escalation_matrix.md §4. Making this a Literal rather
# than a free string means a person's name or email address cannot validate into
# the field at all - the PII guardrail for this field is the type system, not a
# regex applied after the fact.
EscalationRole = Literal[
    "Merch Planner",
    "Senior Merch Planner",
    "Head of Buying",
    "Wholesale Planning Lead",
    "Supply Chain Risk Lead",
]


class TriageRecommendation(BaseModel):
    """The structured output contract. Matches the assessment spec exactly."""

    model_config = ConfigDict(extra="forbid")

    po_id: str
    recommended_action: RecommendedAction
    rationale: str
    citations: list[str] = Field(default_factory=list)
    confidence: Confidence
    escalation_target_role: EscalationRole | None = None


# --- Source data ---------------------------------------------------------


class PurchaseOrder(BaseModel):
    model_config = ConfigDict(extra="ignore")

    po_id: str
    supplier: str
    channel: Literal["retail", "wholesale"]
    category: str
    sku: str
    ordered_qty: int
    confirmed_qty: int
    original_eta: date
    eta: date
    original_value_gbp: float
    value_gbp: float
    status: str
    parent_po_id: str | None = None
    remainder_eta: date | None = None
    supplier_note: str | None = None


class Forecast(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sku: str
    category: str
    forecast_units: int
    actual_units: int
    variance_units: int
    variance_pct: float
    forecast_period: str
    linked_po_ids: list[str] = Field(default_factory=list)


# --- Retrieval -----------------------------------------------------------


class Chunk(BaseModel):
    """A retrievable unit of SOP text.

    `chunk_id` is the citation string ("po_amendment_policy.md §3"). Because the
    chunker splits on section headings, every citation the agent can legitimately
    emit already exists as a chunk_id - which is what makes citation validation
    an exact set-membership test rather than fuzzy matching.
    """

    chunk_id: str
    doc: str
    section: str
    heading: str
    text: str
    redacted: bool = False


class RetrievedChunk(BaseModel):
    """`score` is the fused RRF score - use it to ORDER results, never to judge
    relevance. RRF is a function of rank, so it says nothing about how good the
    match is. `dense_score` (cosine) and `lexical_score` (BM25) are the raw
    signals that do carry relevance information."""

    chunk: Chunk
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    dense_score: float | None = None


# --- Guardrail + envelope -------------------------------------------------


class GuardrailFlag(BaseModel):
    """A guardrail that fired. `blocking` flags force the final action."""

    name: str
    detail: str
    blocking: bool = False


class ToolInvocation(BaseModel):
    step: int
    name: str
    arguments: dict[str, Any]
    result_summary: str


class TriageResult(BaseModel):
    """Operational envelope. The API returns this; `recommendation` is the contract."""

    recommendation: TriageRecommendation
    guardrail_flags: list[GuardrailFlag] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    # Citations naming a real corpus section that was not itself retrieved.
    resolved_citations: list[str] = Field(default_factory=list)
    # Citations matching no section in the corpus. Invented, and dropped.
    dropped_citations: list[str] = Field(default_factory=list)
    tool_calls: list[ToolInvocation] = Field(default_factory=list)
    # Max cosine similarity, or None where no semantic embedding model was used.
    semantic_similarity: float | None = None
    grounding_reason: str = ""
    steps_used: int = 0
    provider: str = "unknown"
    usage: dict[str, int] = Field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return any(f.blocking for f in self.guardrail_flags)
