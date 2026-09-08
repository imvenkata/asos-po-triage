"""Grounding guardrails: citation validity and retrieval grounding.

Citation validation resolves against two sets, not one, because "not retrieved"
and "does not exist" are different failures and deserve different handling:

  verified    the citation was in the context window. Trusted.
  resolved    the citation names a real corpus section that was NOT retrieved.
              This happens legitimately - SOP sections cross-reference each
              other, and a model that follows "see po_amendment_policy.md §3"
              is behaving correctly. Kept, but recorded.
  fabricated  the citation matches no section in the corpus. Invention.
              Dropped, and blocking.

--------------------------------------------------------------------------
Grounding assessment - and a bug worth recording, because it is easy to repeat

The first version thresholded the fused RRF score. That cannot work. RRF scores
1/(k + rank): rank 1 scores 1/61 whether the hit is a bullseye or garbage, and a
dense index always returns k results however bad they are. Measured on this
corpus, "what is the best recipe for sourdough bread" scored 0.03200 - identical
to the correct policy query. The guardrail could not fire.

Relevance has to be read from the raw signals fusion consumed, not from fusion's
output. Two are used here, and they answer different questions:

  semantic similarity max cosine over retrieved chunks. The gate.

  lexical overlap     how many content words the question shares with the
                      corpus. Reported as a diagnostic, deliberately NOT used to
                      block - see below.

Why there is no lexical fallback, having tried to build one: "PO-10001 came back
slightly short" shares zero content words with the corpus ("came", "back",
"slightly", "short" appear nowhere in it) and is a perfectly good question.
"What is the best recipe for sourdough bread" also shares zero. Lexically the
two are indistinguishable, because a planner describes the symptom in their own
words while the SOPs use policy vocabulary - which is the vocabulary-mismatch
problem dense retrieval exists to solve in the first place. A lexical gate
therefore cannot separate off-topic from on-topic-but-differently-phrased, and
shipping one produces confident false refusals on legitimate questions.

So this guardrail genuinely depends on semantic retrieval. Where no semantic
embedding model is configured it reports itself INACTIVE rather than silently
passing everything or blocking real questions. The eval suite skips the
out-of-scope case in that mode instead of pretending to have tested it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import json

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


def validate_citations(
    emitted: list[str], exposed: set[str], corpus: set[str]
) -> CitationVerdict:
    """Partition emitted citations into verified / resolved / fabricated."""
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
            supplied.update((n, abs(n), n.quantize(Decimal("0.01")), abs(n).quantize(Decimal("0.01"))))
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
            "unmeasurable" if self.semantic_similarity is None
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
        return GroundingAssessment(
            False, "nothing retrieved", matched_terms, total_terms, None
        )

    if not semantic_scores_meaningful:
        return GroundingAssessment(
            True,
            "not measurable - no semantic embedding model configured, so the "
            "grounding gate is INACTIVE",
            matched_terms, total_terms, None, active=False,
        )

    scores = [r.dense_score for r in retrieved if r.dense_score is not None]
    similarity = max(scores) if scores else 0.0
    if similarity < min_semantic_similarity:
        return GroundingAssessment(
            False,
            f"best semantic match {similarity:.3f} is below the floor "
            f"{min_semantic_similarity}",
            matched_terms, total_terms, similarity,
        )
    return GroundingAssessment(True, "grounded", matched_terms, total_terms, similarity)
