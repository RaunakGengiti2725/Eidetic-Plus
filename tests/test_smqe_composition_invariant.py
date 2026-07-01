from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_composition_invariant import run_eval


def test_smqe_composition_eval_passes_record_and_claim_backends_on_rotating_seed():
    report = run_eval(seed=767676, cases=24)

    assert report["pass"] is True
    assert report["checks"] == 48
    assert report["correct"] == 24
    assert report["record_backend_correct"] == 24
    assert report["claim_backend_correct"] == 24
    assert report["backend_counts"] == {"claim": 24, "record": 24}
    assert report["case_type_counts"] == {
        "event_order": 4,
        "relative_event_time": 4,
        "shared_value": 16,
    }
    assert report["avg_proof_tokens"] < 80


def test_smqe_composition_eval_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_composition_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_composition_invariant",
            "--seed",
            "777777",
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
    assert report["seed"] == 777777
    assert report["cases"] == 12
    assert report["checks"] == 24
    assert report["record_backend_correct"] == 12
    assert report["claim_backend_correct"] == 12
