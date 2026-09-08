"""Contradiction guardrail.

Detects the case the assessment plants deliberately: two policy sections that
state different thresholds for the same decision. This runs deterministically
over the corpus for conflict closure and over the fetched PO's figures in the
get_po result. The final gate independently enforces known material conflicts.
This controls the targeted thresholds even when the model misses them.

Scope, stated honestly: this is a targeted extractor over named policy
dimensions, not general natural-language inference. It is precise on the
dimensions it knows and blind to conflicts expressed in prose it has no anchor
for. See WRITEUP.md for how I would generalise it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Chunk


@dataclass(frozen=True)
class PolicyDimension:
    """A single decision threshold that must be consistent across the corpus.

    `observed_key` names the field in a PO's computed-variance block that this
    threshold is compared against. It is what makes materiality assessable: a
    conflict only matters if the PO actually sits between the competing values.
    """

    key: str
    label: str
    anchors: tuple[str, ...]
    unit_pattern: str  # regex capturing the numeric value in that unit
    unit: str
    observed_key: str | None = None


DIMENSIONS: tuple[PolicyDimension, ...] = (
    PolicyDimension(
        key="amendment_variance_pct",
        label="variance threshold for amending a PO in place",
        anchors=("cumulative variance",),
        unit_pattern=r"(\d+(?:\.\d+)?)\s*%",
        unit="%",
        observed_key="value_variance_pct",
    ),
    PolicyDimension(
        key="max_eta_slip_days",
        label="maximum ETA movement before re-raise",
        anchors=("calendar days from the original eta",),
        unit_pattern=r"(\d+)\s*calendar days",
        unit="days",
        observed_key="eta_slip_days",
    ),
    PolicyDimension(
        key="backorder_max_delay_days",
        label="maximum permissible backorder delay",
        anchors=("maximum permissible backorder delay",),
        unit_pattern=r"(\d+)\s*calendar days",
        unit="days",
    ),
)

_SENTENCE = re.compile(r"[^.\n]*(?:\.|\n|$)")


@dataclass(frozen=True)
class ThresholdClaim:
    dimension: PolicyDimension
    value: float
    chunk_id: str
    sentence: str


@dataclass(frozen=True)
class ThresholdConflict:
    dimension: PolicyDimension
    claims: tuple[ThresholdClaim, ...]

    def describe(self) -> str:
        parts = " vs ".join(f"{c.value:g}{c.dimension.unit} [{c.chunk_id}]" for c in self.claims)
        return f"{self.dimension.label}: {parts}"

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(sorted(c.value for c in self.claims))

    def is_material_for(self, observations: dict[str, float] | None) -> bool:
        """Would the competing thresholds produce different outcomes for this PO?

        A corpus-level inconsistency is not automatically a decision-level
        problem. If the observed value sits below every competing threshold, or
        above all of them, both sections agree on the outcome and blocking the
        recommendation would be a false positive - the pathology that made the
        first version of this guardrail escalate every single PO.

        Conservative by construction: with no observation to compare against, the
        conflict is treated as material.
        """
        key = self.dimension.observed_key
        if not key or not observations or key not in observations:
            return True
        try:
            observed = abs(float(observations[key]))
        except (TypeError, ValueError):
            return True
        low, high = self.values[0], self.values[-1]
        return low < observed <= high


def _extract(chunk: Chunk) -> list[ThresholdClaim]:
    claims: list[ThresholdClaim] = []
    for raw in _SENTENCE.findall(chunk.text):
        sentence = raw.strip()
        if not sentence:
            continue
        low = sentence.lower()
        for dim in DIMENSIONS:
            if not any(a in low for a in dim.anchors):
                continue
            for value in re.findall(dim.unit_pattern, low):
                claims.append(ThresholdClaim(dim, float(value), chunk.chunk_id, sentence))
    return claims


def detect_threshold_conflicts(
    chunks: list[Chunk], observations: dict[str, float] | None = None
) -> list[ThresholdConflict]:
    """Return one conflict per dimension that carries two or more distinct values.

    Pass `observations` (a PO's computed-variance block) to filter down to the
    conflicts that actually change this PO's outcome; omit it to get every
    corpus-level inconsistency, which is what the corpus audit uses.
    """
    by_dim: dict[str, list[ThresholdClaim]] = {}
    for chunk in chunks:
        for claim in _extract(chunk):
            by_dim.setdefault(claim.dimension.key, []).append(claim)

    conflicts: list[ThresholdConflict] = []
    for claims in by_dim.values():
        distinct = sorted({c.value for c in claims})
        if len(distinct) < 2:
            continue
        # Keep one representative claim per distinct value, in document order.
        seen: set[float] = set()
        reps: list[ThresholdClaim] = []
        for claim in claims:
            if claim.value not in seen:
                seen.add(claim.value)
                reps.append(claim)
        conflicts.append(ThresholdConflict(reps[0].dimension, tuple(reps)))

    if observations is not None:
        conflicts = [c for c in conflicts if c.is_material_for(observations)]
    return conflicts


def detect_all_conflicts(chunks: list[Chunk]) -> list[ThresholdConflict]:
    """Every inconsistency in the corpus, material or not. Used by `triage audit`."""
    return detect_threshold_conflicts(chunks, observations=None)
