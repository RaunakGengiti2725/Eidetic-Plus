from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_lacuna_invariant import run_eval


def test_smqe_lacuna_eval_passes_on_rotating_seed():
    report = run_eval(seed=424242, cases=24)

    assert report["pass"] is True
    assert report["correct"] == 24
    assert set(report["case_type_counts"]) == {
        "positive_confirmation",
        "negative_assertion",
        "retraction_order",
        "absent_proposition",
    }
    assert min(report["case_type_counts"].values()) >= 6


def test_smqe_lacuna_eval_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_lacuna_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_lacuna_invariant",
            "--seed",
            "999",
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
    assert report["cases"] == 12
    assert report["seed"] == 999
    assert report["seed_mode"] == "fixed"
