from __future__ import annotations

import json
import subprocess
import sys

from bench.region_routing_invariant import run_eval


def test_region_routing_invariant_passes_on_rotating_seed():
    report = run_eval(seed=12345, cases=4)

    assert report["pass"] is True
    assert report["cases"] == 4
    assert report["checks"] == 48
    assert report["correct"] == 48
    assert report["dense_miss_recovery_checks"] == 12
    assert report["active_scope_filter_checks"] == 8
    assert report["nested_cocoon_checks"] == 8
    assert report["proof_link_checks"] == 8
    assert report["telemetry_trace_checks"] == 8
    assert report["route_only_context_checks"] == 4
    assert report["case_type_counts"] == {"region_routing_cocoon_proof": 4}


def test_region_routing_invariant_cli_writes_report(tmp_path):
    out = tmp_path / "region_routing_invariant.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.region_routing_invariant",
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
    assert report["checks"] == 36
    assert report["correct"] == 36
