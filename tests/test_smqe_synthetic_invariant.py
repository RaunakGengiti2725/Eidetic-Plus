from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_synthetic_invariant import generate_cases, run_eval


def test_smqe_synthetic_eval_passes_rotating_seed():
    report = run_eval(seed=424242, cases=46)

    assert report["pass"] is True
    assert report["correct"] == 46
    assert report["failures"] == []
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
    assert report["backend_counts"]["claim"] >= 1
    assert report["backend_counts"]["record"] >= 1
    assert report["avg_proof_tokens"] < 80


def test_smqe_synthetic_eval_changes_with_seed():
    first = [(c.op, c.question, c.expected) for c in generate_cases(1001, 12)]
    second = [(c.op, c.question, c.expected) for c in generate_cases(1002, 12)]

    assert first != second


def test_smqe_synthetic_eval_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_synth.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_synthetic_invariant",
            "--seed",
            "515151",
            "--cases",
            "10",
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
    assert report["seed"] == 515151
    assert report["cases"] == 10
