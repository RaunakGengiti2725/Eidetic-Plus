from __future__ import annotations

from bench.smqe_attribution_invariant import generate_cases, run_eval


def test_attribution_cases_cover_actor_types():
    cases = generate_cases(seed=20260630, cases=16)
    counts = {}
    for case in cases:
        counts[case.case_type] = counts.get(case.case_type, 0) + 1

    assert counts == {
        "gave_actor": 4,
        "recommend_actor": 4,
        "shared_actor": 4,
        "told_actor": 4,
    }


def test_attribution_invariant_passes_both_backends():
    report = run_eval(seed=20260630, cases=16)

    assert report["pass"] is True
    assert report["checks"] == 32
    assert report["record_backend_correct"] == 16
    assert report["claim_backend_correct"] == 16
    assert report["backend_counts"] == {"claim": 16, "record": 16}
