"""Assembles the searchable SOP index: chunk -> redact -> embed -> fuse."""
from __future__ import annotations

from pathlib import Path

from ..config import Settings, get_settings
from ..guardrails.contradiction import detect_all_conflicts
from ..guardrails.pii import PiiRegistry, build_pii_registry
from ..ingest.chunker import chunk_corpus
from ..llm.base import LLMError
from ..llm.embeddings import Embedder
from ..logging_setup import get_logger
from ..models import Chunk, RetrievedChunk
from .dense import DenseIndex
from .hybrid import reciprocal_rank_fusion
from .lexical import BM25

log = get_logger("retrieval.index")


class SopIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        registry: PiiRegistry,
        settings: Settings,
        embedder: Embedder | None = None,
    ) -> None:
        self.chunks = chunks
        self.registry = registry
        self._settings = settings
        self._embedder = embedder

        # Embed the heading alongside the body: headings carry most of the topic
        # signal in a policy corpus and are short enough to be free.
        self._texts = [f"{c.doc} — {c.heading}\n{c.text}" for c in chunks]
        self._bm25 = BM25(self._texts)
        self._dense: DenseIndex | None = None
        self._by_id = {c.chunk_id: i for i, c in enumerate(chunks)}
        self._conflict_partners = self._build_conflict_graph()

        if embedder is not None:
            try:
                self._dense = DenseIndex(embedder.embed(self._texts))
                log.info("Dense index built over %d chunks (%s)", len(chunks), embedder.name)
            except LLMError as exc:
                # Degrade to lexical-only, but say so at WARNING and record it, so
                # a half-working retriever never looks like a healthy one.
                log.warning("Dense index unavailable, running lexical-only: %s", exc)

    def _build_conflict_graph(self) -> dict[str, set[str]]:
        """Precompute which sections contradict which, once, at index time.

        A contradiction is only detectable if BOTH conflicting sections reach the
        context window, and relevance ranking does not guarantee that: the two
        halves of a conflict are phrased differently, so a query matching one may
        rank the other well outside top-k. Measured on this corpus, "can I amend
        PO-10342 with a 12% variance" retrieved NEITHER threshold section - it
        surfaced the definitions and sign-off sections instead.

        So closure is keyed on the DOCUMENT, not the chunk: an unresolved
        conflict is a property of the policy as a whole, and if the agent is
        consulting that policy at all it must see the parts of it that disagree.
        Cost is bounded - conflicts are defects, so they are rare, and each adds
        at most a couple of sections.
        """
        graph: dict[str, set[str]] = {}
        for conflict in detect_all_conflicts(self.chunks):
            ids = {claim.chunk_id for claim in conflict.claims}
            for doc in {cid.split(" §")[0] for cid in ids}:
                graph.setdefault(doc, set()).update(ids)
        if graph:
            log.info(
                "Conflict graph: %s carry unresolved threshold conflicts", sorted(graph)
            )
        return graph

    @property
    def semantic_scores_meaningful(self) -> bool:
        """True only when a real embedding model built the dense index."""
        return self._dense is not None and bool(
            getattr(self._embedder, "provides_semantic_embeddings", False)
        )

    def informative_overlap(self, query: str) -> tuple[int, int]:
        return self._bm25.informative_overlap(query)

    @property
    def mode(self) -> str:
        return "hybrid (BM25 + dense, RRF)" if self._dense else "lexical-only (BM25)"

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self._settings.triage_retrieval_top_k
        pool = max(top_k * 3, 12)

        lexical = self._bm25.search(query, pool)
        dense: list[tuple[int, float]] = []
        if self._dense is not None and self._embedder is not None:
            try:
                dense = self._dense.search(self._embedder.embed([query])[0], pool)
            except LLMError as exc:
                log.warning("Query embedding failed, this query is lexical-only: %s", exc)

        fused = reciprocal_rank_fusion(
            lexical, dense, k=self._settings.triage_rrf_k, top_k=top_k
        )
        results = [
            RetrievedChunk(
                chunk=self.chunks[h.index],
                score=h.score,
                lexical_rank=h.lexical_rank,
                dense_rank=h.dense_rank,
                lexical_score=h.lexical_score,
                dense_score=h.dense_score,
            )
            for h in fused
        ]
        return self._enforce_conflict_closure(results)

    def _enforce_conflict_closure(
        self, results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Never return one half of a known contradiction without the other."""
        present = {r.chunk.chunk_id for r in results}
        # Score each pulled-in section at the score of the hit that pulled it, so
        # the retrieval-confidence gate is not skewed by these additions.
        forced: list[RetrievedChunk] = []
        for hit in results:
            for partner_id in self._conflict_partners.get(hit.chunk.doc, ()):
                if partner_id in present:
                    continue
                present.add(partner_id)
                forced.append(
                    RetrievedChunk(chunk=self.chunks[self._by_id[partner_id]], score=hit.score)
                )
        if forced:
            log.info(
                "Conflict closure pulled in %s", [f.chunk.chunk_id for f in forced]
            )
        return results + forced

    @property
    def chunk_ids(self) -> set[str]:
        return {c.chunk_id for c in self.chunks}


def build_index(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    corpus_dir: Path | None = None,
) -> SopIndex:
    settings = settings or get_settings()
    corpus_dir = corpus_dir or settings.corpus_dir
    registry = build_pii_registry(corpus_dir)
    if registry.is_empty:
        log.warning("No contact details found in corpus - PII registry is empty.")
    chunks = chunk_corpus(corpus_dir, registry)
    return SopIndex(chunks, registry, settings, embedder)
