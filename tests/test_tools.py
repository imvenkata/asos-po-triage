import json

from triage.data_access import get_repository
from triage.models import TriageRecommendation
from triage.tools.registry import TERMINAL_TOOL, RetrievalRecorder, build_tools


def _tools(settings, index, top_k=6):
    return build_tools(get_repository(settings), index, RetrievalRecorder(), top_k)


def test_the_four_tools_are_exposed(settings, index):
    assert {t.name for t in _tools(settings, index)} == {
        "get_po", "get_forecast", "search_sops", TERMINAL_TOOL
    }


def test_computed_variance_is_exact_not_estimated(settings, index):
    """The model must never do arithmetic; the tool hands it the figures."""
    tool = next(t for t in _tools(settings, index) if t.name == "get_po")
    variance = json.loads(tool.invoke({"po_id": "PO-10342"}))["computed_variance"]
    assert variance["qty_variance_pct"] == 12.0
    assert variance["value_variance_pct"] == 12.0


def test_lookup_is_case_insensitive(settings):
    assert get_repository(settings).get_po("po-10001") is not None


def test_unknown_po_returns_an_error_payload_not_an_exception(settings, index):
    tool = next(t for t in _tools(settings, index) if t.name == "get_po")
    assert "error" in json.loads(tool.invoke({"po_id": "PO-00000"}))


def test_search_records_every_chunk_the_model_was_shown(settings, index):
    recorder = RetrievalRecorder()
    tools = build_tools(get_repository(settings), index, recorder, 4)
    next(t for t in tools if t.name == "search_sops").invoke({"query": "backorder wholesale"})
    assert len(recorder.exposed_chunk_ids) == 4
    assert len(recorder.retrieved) == 4


def test_submit_schema_mirrors_the_output_contract(settings, index):
    """If these drift, structured output stops matching the published contract."""
    submit = next(t for t in _tools(settings, index) if t.name == TERMINAL_TOOL)
    assert submit.args_schema.model_json_schema() == TriageRecommendation.model_json_schema()
