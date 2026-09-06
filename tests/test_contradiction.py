from triage.guardrails.contradiction import detect_all_conflicts, detect_threshold_conflicts


def test_detects_the_planted_conflict(index):
    conflicts = detect_all_conflicts(index.chunks)
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.dimension.key == "amendment_variance_pct"
    assert conflict.values == (10.0, 15.0)
    cited = {c.chunk_id for c in conflict.claims}
    assert cited == {"po_amendment_policy.md §2", "po_amendment_policy.md §4"}


def test_does_not_over_fire_on_consistent_thresholds(index):
    """Both §2 and §4 state a 20-day ETA limit. Agreement must not read as conflict."""
    keys = {c.dimension.key for c in detect_all_conflicts(index.chunks)}
    assert "max_eta_slip_days" not in keys


def test_conflict_is_immaterial_below_both_thresholds(index):
    """4.5% variance: 10% and 15% agree, so the recommendation must not be blocked."""
    assert detect_threshold_conflicts(index.chunks, {"value_variance_pct": 4.5}) == []


def test_conflict_is_material_between_the_thresholds(index):
    """12% variance: §2 permits amendment, §4 forbids it. Genuine disagreement."""
    assert len(detect_threshold_conflicts(index.chunks, {"value_variance_pct": 12.0})) == 1


def test_conflict_is_immaterial_above_both_thresholds(index):
    assert detect_threshold_conflicts(index.chunks, {"value_variance_pct": 35.0}) == []


def test_missing_observation_is_treated_conservatively(index):
    assert len(detect_threshold_conflicts(index.chunks, {})) == 1
