"""Evidence identity and relevance checks.

A model citation is accepted only if its text was exposed to that request.
Existing but unread references are not evidence. The numerical check tests
literal provenance; neither it nor cosine similarity proves semantic entailment.
RRF ranks results and is deliberately not used as a relevance threshold.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from ..models import RetrievedChunk

_NORMALISE = re.compile(r"\b(?:section|sec\.?|s\.)\s*(\d+)\b", re.IGNORECASE)


@dataclass
class CitationVerdict:
    verified: list[str] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)
    unexposed: list[str] = field(default_factory=list)
    fabricated: list[str] = field(default_factory=list)

    @property
    def kept(self) -> list[str]:
        return self.verified


def _normalise(raw: str) -> str:
    cite = " ".join(str(raw).split())
    cite = _NORMALISE.sub(r"§\1", cite)
    return re.sub(r"§\s+(\d)", r"§\1", cite)


def validate_citations(emitted: list[str], exposed: set[str], corpus: set[str]) -> CitationVerdict:
    """Partition citations into exposed, existing-but-unread, and nonexistent."""
    verdict = CitationVerdict()
    for raw in emitted or []:
        cite = _normalise(raw)
        if cite in exposed:
            if cite not in verdict.verified:
                verdict.verified.append(cite)
        elif cite in corpus:
            if cite not in verdict.unexposed:
                verdict.unexposed.append(cite)
        elif cite not in verdict.fabricated:
            verdict.fabricated.append(cite)
    return verdict


def inline_citations(rationale: str) -> list[str]:
    """Include malformed bracketed file references so they cannot evade checks."""
    return [_normalise(s) for s in re.findall(r"\[([^\]\n]*\.md[^\]\n]*)\]", rationale, re.I)]


def normalize_citations(citations: list[str]) -> set[str]:
    return {_normalise(c) for c in citations}


def _numbers(text: str) -> set[Decimal]:
    text = re.sub(r"\[[^\]]*\.md[^\]]*\]|\b(?:PO|SKU)-[\w-]+|§\s*\d+", "", text, flags=re.I)
    values = set()
    for token in re.findall(r"(?<!\w)[+-]?\d[\d,]*(?:\.\d+)?", text):
        try:
            values.add(Decimal(token.replace(",", "")))
        except InvalidOperation:
            pass
    return values


def unsupported_numbers(rationale: str, cited_text: str, facts: dict | None) -> bool:
    """Literal-number provenance, NOT entailment or general fact checking.

    Compare absolute and displayed (two-decimal) values because the calculation
    retains precision for threshold decisions while the prose may round it.
    """
    supplied = _numbers(cited_text)
    if facts:
        # Free-text supplier notes are untrusted, not numerical authority.
        structured = {k: v for k, v in facts.items() if k not in ("supplier_note", "supplier")}
        for n in _numbers(json.dumps(structured, default=str)):
            supplied.update(
                (n, abs(n), n.quantize(Decimal("0.01")), abs(n).quantize(Decimal("0.01")))
            )
    return not _numbers(rationale).issubset(supplied)


@dataclass
class GroundingAssessment:
    grounded: bool
    reason: str
    matched_terms: int
    total_terms: int
    semantic_similarity: float | None  # None => not measurable with this provider
    active: bool = True  # False => the check could not run at all

    def as_detail(self) -> str:
        semantic = (
            "unmeasurable"
            if self.semantic_similarity is None
            else f"{self.semantic_similarity:.3f}"
        )
        return (
            f"{self.reason} (max cosine {semantic}, "
            f"content-word overlap {self.matched_terms}/{self.total_terms})"
        )


def assess_grounding(
    retrieved: list[RetrievedChunk],
    matched_terms: int,
    total_terms: int,
    semantic_scores_meaningful: bool,
    min_semantic_similarity: float,
) -> GroundingAssessment:
    if not retrieved:
        # Valid in every mode: nothing was retrieved, so nothing grounds an answer.
        return GroundingAssessment(False, "nothing retrieved", matched_terms, total_terms, None)

    if not semantic_scores_meaningful:
        return GroundingAssessment(
            True,
            "not measurable - no semantic embedding model configured, so the "
            "grounding gate is INACTIVE",
            matched_terms,
            total_terms,
            None,
            active=False,
        )

    scores = [r.dense_score for r in retrieved if r.dense_score is not None]
    similarity = max(scores) if scores else 0.0
    if similarity < min_semantic_similarity:
        return GroundingAssessment(
            False,
            f"best semantic match {similarity:.3f} is below the floor {min_semantic_similarity}",
            matched_terms,
            total_terms,
            similarity,
        )
    return GroundingAssessment(True, "grounded", matched_terms, total_terms, similarity)
