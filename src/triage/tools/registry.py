"""Tool registry: schemas the model sees, and the dispatcher that runs them.

One registry owns both halves so a tool cannot be advertised without being
executable, or executed without being advertised. Dispatch is whitelist-only -
a hallucinated tool name is rejected and fed back to the model as an error
result rather than raising, which lets the loop self-correct within its budget.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..data_access import PoRepository
from ..logging_setup import get_logger
from ..models import RetrievedChunk
from ..retrieval.index import SopIndex
from .po_tools import get_forecast, get_po

log = get_logger("tools")

ACTIONS = ["split_child_po", "amend", "firm_planned_order", "raise_backorder", "escalate"]
ROLES = [
    "Merch Planner",
    "Senior Merch Planner",
    "Head of Buying",
    "Wholesale Planning Lead",
    "Supply Chain Risk Lead",
]


@dataclass
class ToolSpec:
    name: str
    schema: dict[str, Any]
    handler: Callable[..., Any]
    terminal: bool = False


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs = {s.name: s for s in specs}
        # Every chunk the model has actually been shown, across all searches.
        # This set is what citation validation is checked against.
        self.exposed_chunk_ids: set[str] = set()
        self.retrieved: list[RetrievedChunk] = []

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [s.schema for s in self._specs.values()]

    def is_terminal(self, name: str) -> bool:
        spec = self._specs.get(name)
        return bool(spec and spec.terminal)

    def dispatch(self, name: str, arguments: dict[str, Any]) -> tuple[str, str | None]:
        """Run a tool. Returns (json_result, error_message)."""
        spec = self._specs.get(name)
        if spec is None:
            msg = f"Unknown tool '{name}'. Available tools: {sorted(self._specs)}."
            log.warning(msg)
            return json.dumps({"error": msg}), msg
        try:
            result = spec.handler(**arguments)
        except TypeError as exc:
            msg = f"Invalid arguments for '{name}': {exc}"
            log.warning(msg)
            return json.dumps({"error": msg}), msg
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, and logged
            msg = f"Tool '{name}' failed: {exc}"
            log.exception(msg)
            return json.dumps({"error": msg}), msg
        return json.dumps(result, default=str), None


def _fn(name: str, description: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": params},
    }


def build_tool_registry(repo: PoRepository, index: SopIndex, top_k: int) -> ToolRegistry:
    registry_ref: dict[str, ToolRegistry] = {}

    def search_sops(query: str, top_k_override: int | None = None) -> dict[str, Any]:
        hits = index.search(query, top_k_override or top_k)
        reg = registry_ref["reg"]
        reg.exposed_chunk_ids |= {h.chunk.chunk_id for h in hits}
        reg.retrieved.extend(hits)
        return {
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
        }

    specs = [
        ToolSpec(
            name="get_po",
            schema=_fn(
                "get_po",
                "Look up a purchase order by id. Returns the PO record plus "
                "pre-computed quantity/value/ETA variance. Always use these computed "
                "figures rather than calculating percentages yourself.",
                {
                    "type": "object",
                    "properties": {
                        "po_id": {"type": "string", "description": "e.g. PO-10342"}
                    },
                    "required": ["po_id"],
                },
            ),
            handler=lambda po_id: get_po(repo, po_id),
        ),
        ToolSpec(
            name="get_forecast",
            schema=_fn(
                "get_forecast",
                "Look up forecast versus actual units for a SKU. Use when the "
                "question concerns demand impact or re-forecasting.",
                {
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string", "description": "e.g. SKU-WW-8820"}
                    },
                    "required": ["sku"],
                },
            ),
            handler=lambda sku: get_forecast(repo, sku),
        ),
        ToolSpec(
            name="search_sops",
            schema=_fn(
                "search_sops",
                "Search the merchandising SOPs. Call this again with a more "
                "specific query whenever you discover a fact about the PO (its "
                "channel, status, or parent) that brings a different policy into "
                "play. Every result carries the citation string you must use.",
                {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language policy question.",
                        }
                    },
                    "required": ["query"],
                },
            ),
            handler=search_sops,
        ),
        ToolSpec(
            name="submit_recommendation",
            schema=_fn(
                "submit_recommendation",
                "Submit the final structured triage recommendation. Call this "
                "exactly once, as your last action.",
                {
                    "type": "object",
                    "properties": {
                        "po_id": {"type": "string"},
                        "recommended_action": {"type": "string", "enum": ACTIONS},
                        "rationale": {
                            "type": "string",
                            "description": "Why, referencing the policy you relied on.",
                        },
                        "citations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Citation strings copied verbatim from "
                            "search_sops results, e.g. 'po_amendment_policy.md §3'.",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "escalation_target_role": {
                            "type": ["string", "null"],
                            "enum": ROLES + [None],
                            "description": "The ROLE only. Never a person's name or "
                            "email address. Null when no escalation is required.",
                        },
                    },
                    "required": [
                        "po_id",
                        "recommended_action",
                        "rationale",
                        "citations",
                        "confidence",
                    ],
                },
            ),
            handler=lambda **kwargs: kwargs,
            terminal=True,
        ),
    ]

    registry = ToolRegistry(specs)
    registry_ref["reg"] = registry
    return registry
