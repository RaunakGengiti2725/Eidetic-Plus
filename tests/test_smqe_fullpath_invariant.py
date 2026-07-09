from __future__ import annotations

import json

import bench.smqe_fullpath_invariant as fullpath


def test_fullpath_invariant_runs_adapter_without_reader_calls():
    report = fullpath.run_eval(seed=12345, cases=23)

    # P0 fail-closed (2026-07-09): DERIVED count/sum ops no longer verify in the structured
    # adapter (eidetic/smqe/verify.py); they fall through to the reader and abstain. Those cases
    # still PASS (correct-or-silent), but they are not verified/structured/claim-backed, and they
    # consult the reader. Everything below is computed from the aggregate case count so the
    # invariant stays exact rather than hard-coding the pre-fix "all cases verify".
    agg = (report["case_operator_counts"].get("count_aggregate", 0)
           + report["case_operator_counts"].get("multi_session_sum", 0))
    non_agg = 23 - agg
    assert agg > 0  # the fix is under test only if aggregate cases are present
    assert report["pass"] is True
    assert report["seed_mode"] == "fixed"
    assert report["correct"] == 23           # aggregates pass by abstaining
    assert report["verified"] == non_agg
    assert report["structured_recall"] == non_agg
    assert report["reader_calls"] == 0       # zero UNEXPECTED reader calls (no failures)
    assert report["reader_consults"] == agg  # aggregates reach the reader tier and abstain
    assert report["proof_link_checks"] == non_agg
    assert report["backend_counts"] == {"claim": non_agg}
    assert sum(report["case_operator_counts"].values()) == 23
    assert report["claims_extracted"] >= non_agg
    assert report["avg_proof_tokens"] < 80
    assert report["avg_context_tokens"] < 80
    assert report["latency_budget_checks"] == 23
    assert report["p95_latency_ms"] < report["latency_budget_ms"]


def test_fullpath_invariant_cli_writes_report(tmp_path, monkeypatch):
    out = tmp_path / "smqe_fullpath_invariant.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "smqe_fullpath_invariant",
            "--seed", "777",
            "--cases", "6",
            "--out", str(out),
        ],
    )

    assert fullpath.main() == 0
    report = json.loads(out.read_text())

    agg = (report["case_operator_counts"].get("count_aggregate", 0)
           + report["case_operator_counts"].get("multi_session_sum", 0))
    assert report["pass"] is True
    assert report["seed"] == 777
    assert report["seed_mode"] == "fixed"
    assert report["reader_calls"] == 0       # zero UNEXPECTED reader calls (no failures)
    assert report["reader_consults"] == agg  # derived aggregates fall to the reader and abstain
    assert report["correct"] == 6
    assert report["proof_link_checks"] == 6 - agg
    assert sum(report["case_operator_counts"].values()) == 6
    assert report["latency_budget_checks"] == 6
