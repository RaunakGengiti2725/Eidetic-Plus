from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_paraphrase_invariant import run_eval


def test_smqe_paraphrase_eval_passes_both_backends_on_rotating_seed():
    report = run_eval(seed=909090, cases=24)

    assert report["pass"] is True
    assert report["correct"] == 24
    assert report["record_backend_correct"] == 24
    assert report["claim_backend_correct"] == 24
    assert report["backend_counts"] == {"claim": 24, "record": 24}
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
    assert report["avg_proof_tokens"] < 80


def test_smqe_paraphrase_eval_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_paraphrase_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_paraphrase_invariant",
            "--seed",
            "919191",
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
    assert report["seed"] == 919191
    assert report["cases"] == 12
    assert report["backend_counts"] == {"claim": 12, "record": 12}
