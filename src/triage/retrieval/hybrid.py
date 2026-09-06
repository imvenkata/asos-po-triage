"""Reciprocal Rank Fusion.

RRF over raw score blending because BM25 scores and cosine similarities live on
incomparable scales; fusing ranks needs no per-corpus weight tuning, which is
exactly what you want when you cannot yet measure retrieval quality on real
traffic. score(d) = sum over rankers of 1/(k + rank(d)), k=60 as per Cormack et al.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusedHit:
    index: int
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None


def reciprocal_rank_fusion(
    lexical: list[tuple[int, float]],
    dense: list[tuple[int, float]],
    k: int = 60,
    top_k: int = 6,
) -> list[FusedHit]:
    hits: dict[int, FusedHit] = {}

    for rank, (idx, _score) in enumerate(lexical, start=1):
        hit = hits.setdefault(idx, FusedHit(index=idx, score=0.0))
        hit.score += 1.0 / (k + rank)
        hit.lexical_rank = rank

    for rank, (idx, _score) in enumerate(dense, start=1):
        hit = hits.setdefault(idx, FusedHit(index=idx, score=0.0))
        hit.score += 1.0 / (k + rank)
        hit.dense_rank = rank

    ordered = sorted(hits.values(), key=lambda h: (-h.score, h.index))
    return ordered[:top_k]
