from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_abstention_invariant import run_eval


def test_smqe_abstention_eval_passes_both_backends_on_rotating_seed():
    report = run_eval(seed=949494, cases=24)

    assert report["pass"] is True
    assert report["abstained"] == 24
    assert report["record_only_abstained"] == 24
    assert report["claims_present_abstained"] == 24
    assert report["checks"] == 48
    assert report["case_type_counts"] == {
        "count_neutral_quantity": 3,
        "count_target_mismatch": 3,
        "latest_future_only": 3,
        "latest_missing_subject": 3,
        "preference_no_positive": 3,
        "speaker_crossed_support": 3,
        "table_missing_row": 3,
        "temporal_missing_anchor": 3,
    }


def test_smqe_abstention_eval_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_abstention_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_abstention_invariant",
            "--seed",
            "959595",
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
    assert report["seed"] == 959595
    assert report["cases"] == 16
    assert report["record_only_abstained"] == 16
    assert report["claims_present_abstained"] == 16
