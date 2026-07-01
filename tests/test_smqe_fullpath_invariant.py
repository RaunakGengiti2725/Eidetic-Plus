from __future__ import annotations

import json

import bench.smqe_fullpath_invariant as fullpath


def test_fullpath_invariant_runs_adapter_without_reader_calls():
    report = fullpath.run_eval(seed=12345, cases=23)

    assert report["pass"] is True
    assert report["seed_mode"] == "fixed"
    assert report["correct"] == 23
    assert report["verified"] == 23
    assert report["structured_recall"] == 23
    assert report["reader_calls"] == 0
    assert report["proof_link_checks"] == 23
    assert report["claim_backend_correct"] == 23
    assert report["backend_counts"] == {"claim": 23}
    assert sum(report["case_operator_counts"].values()) == 23
    assert report["claims_extracted"] >= 23
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

    assert report["pass"] is True
    assert report["seed"] == 777
    assert report["seed_mode"] == "fixed"
    assert report["reader_calls"] == 0
    assert report["correct"] == 6
    assert report["proof_link_checks"] == 6
    assert sum(report["case_operator_counts"].values()) == 6
    assert report["latency_budget_checks"] == 6
