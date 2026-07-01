from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_invalidation_invariant import run_eval


def test_smqe_invalidation_eval_passes_before_after_and_backends_on_rotating_seed():
    report = run_eval(seed=101010, cases=27)

    assert report["pass"] is True
    assert report["correct"] == 108
    assert report["record_backend_correct"] == 54
    assert report["claim_backend_correct"] == 54
    assert report["backend_counts"] == {"claim": 54, "record": 54}
    assert report["operator_counts"] == {
        "count_aggregate": 3,
        "latest_value": 6,
        "multi_session_sum": 3,
        "preference_synth": 3,
        "relative_temporal": 3,
        "speaker_fact": 3,
        "table_lookup": 3,
        "temporal_delta": 3,
    }
    assert report["case_type_counts"]["preference_supersession"] == 3
    assert report["preference_supersession_pass"] is True
    assert report["preference_supersession_cases"] == 3
    assert report["preference_supersession_checks"] == 12
    assert report["preference_supersession_correct"] == 12
    assert report["preference_supersession_record_correct"] == 6
    assert report["preference_supersession_claim_correct"] == 6
    assert report["avg_proof_tokens"] < 80


def test_smqe_invalidation_eval_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_invalidation_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_invalidation_invariant",
            "--seed",
            "111111",
            "--cases",
            "18",
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
    assert report["cases"] == 18
    assert report["checks"] == 72
    assert report["record_backend_correct"] == 36
    assert report["claim_backend_correct"] == 36
    assert report["preference_supersession_pass"] is True
    assert report["preference_supersession_cases"] == 2
    assert report["preference_supersession_checks"] == 8
