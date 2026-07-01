from __future__ import annotations

from bench.smqe_temporal_window_invariant import generate_cases, run_eval


def test_temporal_window_cases_cover_window_types():
    cases = generate_cases(seed=20260630, cases=22)
    counts = {}
    for case in cases:
        counts[case.case_type] = counts.get(case.case_type, 0) + 1

    assert counts == {
        "fortnight_count": 2,
        "most_recent_latest": 2,
        "past_days_count": 2,
        "past_few_months_count": 2,
        "past_week_count": 2,
        "past_week_list": 2,
        "recent_count": 2,
        "recent_hours_sum": 2,
        "recent_list": 2,
        "source_action_variant_window": 2,
        "source_location_window": 2,
    }


def test_temporal_window_invariant_passes_both_backends():
    report = run_eval(seed=20260630, cases=22)

    assert report["pass"] is True
    assert report["checks"] == 44
    assert report["record_backend_correct"] == 22
    assert report["claim_backend_correct"] == 22
    assert report["backend_counts"] == {"claim": 22, "record": 22}
