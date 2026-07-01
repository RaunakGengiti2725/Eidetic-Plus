from __future__ import annotations

import json
import subprocess
import sys

from bench.affect_salience_invariant import run_eval


def test_affect_salience_invariant_passes_on_rotating_seed():
    report = run_eval(seed=12345, cases=4)

    assert report["pass"] is True
    assert report["cases"] == 4
    assert report["checks"] == 28
    assert report["correct"] == 28
    assert report["flip_checks"] == 8
    assert report["age_free_checks"] == 4
    assert report["bounded_checks"] == 4
    assert report["case_type_counts"] == {"affect_salience_retrieval": 4}
    assert report["max_boost_ratio"] <= 0.5
    assert report["min_age_gap_seconds"] >= 2_592_000


def test_affect_salience_invariant_cli_writes_report(tmp_path):
    out = tmp_path / "affect_salience_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.affect_salience_invariant",
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
    assert report["checks"] == 21
    assert report["correct"] == 21
