from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_planner_invariant import run_eval


def test_smqe_planner_invariant_passes_on_rotating_seed():
    report = run_eval(seed=12345, cases=10)

    assert report["pass"] is True
    assert report["cases"] == 10
    assert report["correct"] == report["checks"]
    assert report["generic_term_checks"] == 10
    assert report["case_type_counts"] == {"smqe_planner_generic_shape": 10}
    assert set(report["operator_counts"]) == {
        "count_aggregate",
        "event_order",
        "latest_value",
        "multi_session_sum",
        "open_inference",
        "preference_synth",
        "relative_temporal",
        "speaker_fact",
        "table_lookup",
        "temporal_delta",
    }


def test_smqe_planner_invariant_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_planner_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_planner_invariant",
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
    assert report["correct"] == report["checks"]
