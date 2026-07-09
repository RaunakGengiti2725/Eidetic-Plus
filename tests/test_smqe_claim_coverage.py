from __future__ import annotations

import json
import subprocess
import sys

from bench.smqe_claim_coverage import run_eval


# P0 fail-closed (2026-07-09): DERIVED count/sum ops no longer verify via the claim backend
# (eidetic/smqe/verify.py) -- they abstain. Such cases still PASS (correct-or-silent) and still
# EXTRACT claims, but they are not claim-backed verified answers, so they drop out of
# backend_counts / claim_backend_operator_counts while remaining in the case-level operator_counts.
_ABSTAINING_AGG_OPS = {"count_aggregate", "multi_session_sum"}


def test_smqe_claim_coverage_passes_rotating_seed():
    report = run_eval(seed=707070, cases=46)

    oc = report["operator_counts"]
    agg_cases = sum(oc.get(op, 0) for op in _ABSTAINING_AGG_OPS)
    assert agg_cases > 0  # the fix is under test only if aggregate cases are present

    assert report["pass"] is True
    assert report["correct"] == 46                 # aggregates pass by abstaining
    assert report["claim_backend_correct"] == 46
    assert report["backend_counts"] == {"claim": 46 - agg_cases}
    # Every NON-aggregate op is fully claim-backed; the two aggregate ops are absent from the
    # claim-backend tally because they fail closed.
    assert set(report["claim_backend_operator_counts"]) == set(oc) - _ABSTAINING_AGG_OPS
    assert all(report["claim_backend_operator_counts"][op] == oc[op]
               for op in report["claim_backend_operator_counts"])
    assert set(oc) == {
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
    assert min(oc.values()) >= 2
    assert report["claims_extracted"] >= 24
    assert report["avg_proof_tokens"] < 80


def test_smqe_claim_coverage_cli_writes_report(tmp_path):
    out = tmp_path / "smqe_claim_coverage.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.smqe_claim_coverage",
            "--seed",
            "818181",
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
    oc = report["operator_counts"]
    agg_cases = sum(oc.get(op, 0) for op in _ABSTAINING_AGG_OPS)
    assert report["pass"] is True
    assert report["seed"] == 818181
    assert report["cases"] == 12
    assert report["backend_counts"] == {"claim": 12 - agg_cases}
    assert set(report["claim_backend_operator_counts"]) == set(oc) - _ABSTAINING_AGG_OPS
