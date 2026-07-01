from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_relative_phrase_invariant import run_eval


def test_smqe_relative_phrase_eval_passes_record_and_claim_backends_on_rotating_seed():
    report = run_eval(seed=666666, cases=24)

    assert report["pass"] is True
    assert report["checks"] == 48
    assert report["correct"] == 24
    assert report["record_backend_correct"] == 24
    assert report["claim_backend_correct"] == 24
    assert report["backend_counts"] == {"claim": 24, "record": 24}
    assert report["case_type_counts"] == {
        "ago_days": 4,
        "ago_weeks": 4,
        "fortnight_ago": 4,
        "in_days": 4,
        "next_month": 4,
        "next_week": 4,
    }
    assert report["avg_proof_tokens"] < 80


def test_smqe_relative_phrase_eval_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_relative_phrase_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_relative_phrase_invariant",
            "--seed",
            "676767",
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
    assert report["seed"] == 676767
    assert report["cases"] == 12
    assert report["checks"] == 24
    assert report["record_backend_correct"] == 12
    assert report["claim_backend_correct"] == 12
