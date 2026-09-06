from triage.guardrails.grounding import assess_grounding, validate_citations

EXPOSED = {"po_amendment_policy.md §2"}
CORPUS = {"po_amendment_policy.md §2", "po_amendment_policy.md §3"}


def test_retrieved_citation_is_verified():
    v = validate_citations(["po_amendment_policy.md §2"], EXPOSED, CORPUS)
    assert v.verified == ["po_amendment_policy.md §2"] and not v.fabricated


def test_real_but_unretrieved_citation_is_resolved_not_dropped():
    v = validate_citations(["po_amendment_policy.md §3"], EXPOSED, CORPUS)
    assert v.resolved == ["po_amendment_policy.md §3"]
    assert v.kept == ["po_amendment_policy.md §3"] and not v.fabricated


def test_invented_section_is_fabricated_and_dropped():
    v = validate_citations(["po_amendment_policy.md §99"], EXPOSED, CORPUS)
    assert v.fabricated == ["po_amendment_policy.md §99"] and v.kept == []


def test_invented_document_is_fabricated():
    v = validate_citations(["procurement_handbook.md §1"], EXPOSED, CORPUS)
    assert v.fabricated == ["procurement_handbook.md §1"]


def test_alternative_section_syntax_is_normalised():
    v = validate_citations(["po_amendment_policy.md Section 2"], EXPOSED, CORPUS)
    assert v.verified == ["po_amendment_policy.md §2"]


def _hit(cosine):
    from triage.models import Chunk, RetrievedChunk

    return RetrievedChunk(
        chunk=Chunk(chunk_id="d.md §1", doc="d.md", section="1", heading="h", text="t"),
        score=0.03,
        dense_score=cosine,
    )


def test_empty_retrieval_is_never_grounded_in_any_mode():
    assert not assess_grounding([], 3, 3, True, 0.30).grounded
    assert not assess_grounding([], 3, 3, False, 0.30).grounded


def test_weak_semantic_match_is_refused():
    a = assess_grounding([_hit(0.11)], 2, 3, True, 0.30)
    assert not a.grounded and a.active
    assert a.semantic_similarity == 0.11


def test_strong_semantic_match_is_grounded():
    assert assess_grounding([_hit(0.62)], 2, 3, True, 0.30).grounded


def test_gate_reports_itself_inactive_without_a_semantic_model():
    """It must never look like a passing check when it cannot run at all."""
    a = assess_grounding([_hit(0.99)], 0, 4, False, 0.30)
    assert a.grounded and not a.active
    assert a.semantic_similarity is None
    assert "INACTIVE" in a.reason


def test_zero_lexical_overlap_alone_never_blocks():
    """'came back slightly short' shares no content word with the corpus and is
    still a legitimate question. Lexical overlap is a diagnostic, not a gate."""
    assert assess_grounding([_hit(0.55)], 0, 4, True, 0.30).grounded
