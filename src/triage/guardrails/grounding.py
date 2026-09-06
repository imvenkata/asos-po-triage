"""Grounding guardrails: citation validity and retrieval confidence.

Citation validation resolves against two sets, not one, because "not retrieved"
and "does not exist" are different failures and deserve different handling:

  verified    the citation was in the context window. Trusted.
  resolved    the citation names a real corpus section that was NOT retrieved.
              This happens legitimately - SOP sections cross-reference each
              other, and a model that follows "see po_amendment_policy.md §3"
              is behaving correctly. Kept, but recorded, because the model is
              asserting something about text it did not actually read.
  fabricated  the citation matches no section in the corpus. This is invention.
              Dropped, and treated as blocking - a recommendation resting on a
              made-up policy reference must not be auto-actioned.

Collapsing `resolved` into `fabricated` was the first version of this guardrail
and it failed four of six eval cases on correct behaviour. Precision matters as
much in guardrails as in the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import RetrievedChunk

# "po_amendment_policy.md §3", tolerating "Section 3" / "sec 3" / missing space.
_NORMALISE = re.compile(r"\b(?:section|sec\.?|s\.)\s*(\d+)\b", re.IGNORECASE)


@dataclass
class CitationVerdict:
    verified: list[str] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)
    fabricated: list[str] = field(default_factory=list)

    @property
    def kept(self) -> list[str]:
        return self.verified + self.resolved


def _normalise(raw: str) -> str:
    cite = " ".join(str(raw).split())
    cite = _NORMALISE.sub(r"§\1", cite)
    cite = re.sub(r"§\s+(\d)", r"§\1", cite)
    return cite


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
            if cite not in verdict.resolved:
                verdict.resolved.append(cite)
        elif cite not in verdict.fabricated:
            verdict.fabricated.append(cite)
    return verdict


def assess_retrieval_confidence(
    retrieved: list[RetrievedChunk], min_score: float
) -> tuple[float, bool]:
    """Return (top score, is_sufficiently_grounded).

    Nothing retrieved, or nothing above the floor, means the corpus does not
    cover the question. The correct behaviour is to say so and escalate - not to
    answer from parametric knowledge about how procurement usually works.
    """
    if not retrieved:
        return 0.0, False
    top = max(r.score for r in retrieved)
    return top, top >= min_score
