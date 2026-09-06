from triage.guardrails.grounding import assess_retrieval_confidence, validate_citations

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


def test_empty_retrieval_is_never_considered_grounded():
    assert assess_retrieval_confidence([], 0.01) == (0.0, False)
