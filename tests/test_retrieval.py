from triage.llm.local_embeddings import embed_texts
from triage.retrieval.hybrid import reciprocal_rank_fusion
from triage.retrieval.lexical import tokenize


def test_tokenizer_keeps_the_tokens_that_carry_the_policy():
    """Thresholds are the meaning here. A tokenizer that drops % or £ loses it."""
    assert "15%" in tokenize("does not exceed 15% of the original value")
    assert "£50000" in tokenize("above £50,000 requires Head of Buying")


def test_hybrid_retrieval_surfaces_both_halves_of_the_contradiction(index):
    """Detecting the conflict is impossible unless retrieval returns both sections."""
    hits = {h.chunk.chunk_id for h in index.search("can I amend PO-10342 with a 12% variance")}
    assert "po_amendment_policy.md §2" in hits
    assert "po_amendment_policy.md §4" in hits


def test_wholesale_query_reaches_the_channel_restriction(index):
    hits = {h.chunk.chunk_id for h in index.search("can I backorder a wholesale PO shortfall")}
    assert "backorder_reconciliation.md §2" in hits


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion(lexical=[(1, 9.0), (2, 8.0)], dense=[(2, 0.9), (3, 0.8)], top_k=3)
    assert fused[0].index == 2  # ranked by both, so it wins despite topping neither
    assert fused[0].lexical_rank == 2 and fused[0].dense_rank == 1


def test_local_embedder_is_deterministic():
    assert embed_texts(["amend the purchase order"]) == embed_texts(["amend the purchase order"])


def test_conflict_closure_fires_even_when_neither_half_ranks(index):
    """The measured failure this invariant exists for: this exact query ranks the
    definitions and sign-off sections above both threshold sections."""
    query = "can I amend PO-10342 with a 12% variance"
    fused_only = {h.chunk.chunk_id for h in index.search(query)}
    assert {"po_amendment_policy.md §2", "po_amendment_policy.md §4"} <= fused_only


def test_closure_is_a_no_op_for_documents_without_conflicts(index, settings):
    from triage.retrieval.index import SopIndex
    chunks = [c for c in index.chunks if c.doc == "backorder_reconciliation.md"]
    clean = SopIndex(chunks, index.registry, settings)
    hits = clean.search("maximum backorder delay", top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "backorder_reconciliation.md §3"
