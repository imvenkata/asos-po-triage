"""The PII guardrail is the one that must not have a bad day."""
import pytest
from pydantic import ValidationError

from triage.guardrails.pii import build_pii_registry, redact, scan_for_pii
from triage.models import TriageRecommendation


def test_every_corpus_contact_is_discovered(settings):
    registry = build_pii_registry(settings.corpus_dir)
    assert len(registry.emails) == 8
    assert "Priya Raman" in registry.names
    assert "James Fenwick" in registry.names


def test_no_contact_survives_index_time_redaction(index):
    """The load-bearing claim: PII cannot reach the context window."""
    for chunk in index.chunks:
        assert scan_for_pii(chunk.text, index.registry) == [], chunk.chunk_id


def test_redaction_preserves_the_routing_roles(index):
    matrix = next(c for c in index.chunks if c.chunk_id == "merch_escalation_matrix.md §1")
    assert "Senior Merch Planner" in matrix.text
    assert "Head of Buying" in matrix.text
    assert "@asos.com" not in matrix.text


def test_output_scan_catches_a_leak_arriving_by_another_route(settings):
    registry = build_pii_registry(settings.corpus_dir)
    leaked = "Escalate to Marcus Whitfield at marcus.whitfield@asos.com"
    findings = scan_for_pii(leaked, registry)
    assert len(findings) == 2
    cleaned, changed = redact(leaked, registry)
    assert changed and scan_for_pii(cleaned, registry) == []


def test_schema_makes_a_person_name_unrepresentable_as_a_role():
    """The strongest guardrail is the one the type system enforces."""
    with pytest.raises(ValidationError):
        TriageRecommendation(
            po_id="PO-1",
            recommended_action="escalate",
            rationale="x",
            confidence="low",
            escalation_target_role="Marcus Whitfield",
        )
