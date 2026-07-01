from __future__ import annotations

import json
import subprocess
import sys

from bench.scratchpad_invariant import run_eval


def test_scratchpad_invariant_passes_on_rotating_seed():
    report = run_eval(seed=12345, cases=6)

    assert report["pass"] is True
    assert report["cases"] == 6
    assert report["checks"] == 66
    assert report["correct"] == 66
    assert report["ordering_checks"] == 6
    assert report["active_scope_filter_checks"] == 6
    assert report["proof_link_checks"] == 24
    assert report["top_k_checks"] == 6
    assert report["retrieval_channel_checks"] == 24
    assert report["case_type_counts"] == {"scratchpad_active_proof_surface": 6}


def test_scratchpad_invariant_cli_writes_report(tmp_path):
    out = tmp_path / "scratchpad_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.scratchpad_invariant",
            "--seed",
            "111111",
            "--cases",
            "4",
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
    assert report["cases"] == 4
    assert report["checks"] == 44
    assert report["correct"] == 44
