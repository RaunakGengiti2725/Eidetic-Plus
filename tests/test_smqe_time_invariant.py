from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_time_invariant import run_eval


def test_smqe_time_eval_passes_before_after_and_backends_on_rotating_seed():
    report = run_eval(seed=989898, cases=24)

    assert report["pass"] is True
    assert report["correct"] == 96
    assert report["record_backend_correct"] == 48
    assert report["claim_backend_correct"] == 48
    assert report["backend_counts"] == {"claim": 48, "record": 48}
    assert report["operator_counts"] == {
        "count_aggregate": 3,
        "latest_value": 3,
        "multi_session_sum": 3,
        "preference_synth": 3,
        "relative_temporal": 3,
        "speaker_fact": 3,
        "table_lookup": 3,
        "temporal_delta": 3,
    }
    assert report["avg_proof_tokens"] < 80


def test_smqe_time_eval_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_time_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_time_invariant",
            "--seed",
            "999999",
            "--cases",
            "16",
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
    assert report["seed"] == 999999
    assert report["cases"] == 16
    assert report["checks"] == 64
    assert report["record_backend_correct"] == 32
    assert report["claim_backend_correct"] == 32
