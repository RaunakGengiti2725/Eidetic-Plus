from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_conflict_invariant import run_eval
from bench.smqe_synthetic_invariant import _proof_excludes_terms


def test_forbidden_proof_matching_uses_phrase_boundaries():
    assert _proof_excludes_terms("User: My current studio fund is 290 usd.", ["90 usd"])
    assert not _proof_excludes_terms("User: My current studio fund is 90 usd.", ["90 usd"])
    assert _proof_excludes_terms("User: My current studio fund is $290.", ["$90"])
    assert not _proof_excludes_terms("User: My current studio fund is $90.", ["$90"])


def test_smqe_conflict_eval_passes_both_backends_on_rotating_seed():
    report = run_eval(seed=929292, cases=24)

    assert report["pass"] is True
    assert report["correct"] == 24
    assert report["record_backend_correct"] == 24
    assert report["claim_backend_correct"] == 24
    assert report["backend_counts"] == {"claim": 24, "record": 24}
    assert report["value_type_counts"] == {"amount": 8, "location": 8, "status": 8}
    assert report["avg_proof_tokens"] < 80


def test_smqe_conflict_eval_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_conflict_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_conflict_invariant",
            "--seed",
            "939393",
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
    assert report["seed"] == 939393
    assert report["cases"] == 12
    assert report["backend_counts"] == {"claim": 12, "record": 12}
