from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_claim_coverage import run_eval


def test_smqe_claim_coverage_passes_rotating_seed():
    report = run_eval(seed=707070, cases=46)

    assert report["pass"] is True
    assert report["correct"] == 46
    assert report["claim_backend_correct"] == 46
    assert report["backend_counts"] == {"claim": 46}
    assert report["claim_backend_operator_counts"] == report["operator_counts"]
    assert set(report["operator_counts"]) == {
        "count_aggregate",
        "latest_value",
        "multi_session_sum",
        "open_inference",
        "preference_synth",
        "relative_temporal",
        "speaker_fact",
        "table_lookup",
        "temporal_delta",
    }
    assert min(report["operator_counts"].values()) >= 2
    assert report["claims_extracted"] >= 24
    assert report["avg_proof_tokens"] < 80


def test_smqe_claim_coverage_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_claim_coverage.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_claim_coverage",
            "--seed",
            "818181",
            "--cases",
            "12",
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
    assert report["seed"] == 818181
    assert report["cases"] == 12
    assert report["backend_counts"] == {"claim": 12}
    assert report["claim_backend_operator_counts"] == report["operator_counts"]
