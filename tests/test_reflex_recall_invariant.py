from __future__ import annotations

import json
import subprocess
import sys

from bench.reflex_recall_invariant import run_eval


def test_reflex_recall_invariant_passes_on_rotating_seed():
    report = run_eval(seed=12345, cases=4)

    assert report["pass"] is True
    assert report["cases"] == 4
    assert report["checks"] == 48
    assert report["correct"] == 48
    assert report["direct_hit_checks"] == 8
    assert report["coactivation_checks"] == 8
    assert report["active_scope_filter_checks"] == 16
    assert report["proof_link_checks"] == 4
    assert report["score_contract_checks"] == 8
    assert report["latency_budget_checks"] == 4
    assert report["p95_latency_ms"] < report["latency_budget_ms"]
    assert report["case_type_counts"] == {"reflex_recall_proof_surface": 4}


def test_reflex_recall_invariant_cli_writes_report(tmp_path):
    out = tmp_path / "reflex_recall_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.reflex_recall_invariant",
            "--seed",
            "111111",
            "--cases",
            "3",
            "--out",
            str(out),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    report = json.loads(out.read_text())
    assert report["pass"] is True
    assert report["seed"] == 111111
    assert report["cases"] == 3
    assert report["checks"] == 36
    assert report["correct"] == 36
