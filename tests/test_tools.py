import json

from triage.data_access import get_repository
from triage.tools.registry import build_tool_registry


def test_computed_variance_is_exact_not_estimated(settings):
    repo = get_repository(settings)
    registry = build_tool_registry(repo, index=None, top_k=3)  # no retrieval needed
    payload, error = registry.dispatch("get_po", {"po_id": "PO-10342"})
    assert error is None
    variance = json.loads(payload)["computed_variance"]
    assert variance["qty_variance_pct"] == 12.0
    assert variance["value_variance_pct"] == 12.0


def test_lookup_is_case_insensitive(settings):
    repo = get_repository(settings)
    assert repo.get_po("po-10001") is not None


def test_unknown_tool_is_rejected_not_raised(settings, index):
    registry = build_tool_registry(get_repository(settings), index, top_k=3)
    payload, error = registry.dispatch("drop_all_purchase_orders", {})
    assert "Unknown tool" in error
    assert "error" in json.loads(payload)


def test_bad_arguments_are_returned_to_the_model_as_an_error(settings, index):
    registry = build_tool_registry(get_repository(settings), index, top_k=3)
    payload, error = registry.dispatch("get_po", {"wrong_kwarg": "PO-10001"})
    assert "Invalid arguments" in error
    assert "error" in json.loads(payload)


def test_search_records_every_chunk_the_model_was_shown(settings, index):
    registry = build_tool_registry(get_repository(settings), index, top_k=4)
    registry.dispatch("search_sops", {"query": "backorder wholesale"})
    assert len(registry.exposed_chunk_ids) == 4
