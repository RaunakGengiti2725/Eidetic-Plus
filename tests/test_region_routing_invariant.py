from __future__ import annotations

import hashlib
import json
import subprocess
import sys

from bench.region_routing_invariant import run_eval
from eidetic.models import DerivedRecord, MemoryRecord, RetrievalCandidate, Scope
from eidetic.retrieval import _memory_region_hints
from eidetic.store import RecordStore


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


def test_region_hints_require_discriminative_query_overlap(tmp_path):
    store = RecordStore(tmp_path / "regions.sqlite")
    scope = Scope(namespace="region-discriminative")

    def rec(mid: str, text: str) -> MemoryRecord:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out = MemoryRecord(
            memory_id=mid,
            text=text,
            source="user",
            scope=scope,
            valid_at=100.0,
            content_hash=h,
            raw_uri=f"cas://{h}",
        )
        store.upsert_record(out)
        return out

    target = rec("target", "Jamie's family roadtrip plan used the west route.")
    decoy = rec("decoy", "Jamie's family camping memory was about a mountain trail.")
    store.add_derived(DerivedRecord(
        cid="target-region",
        kind="gist",
        namespace=scope.namespace,
        level=1,
        text="family roadtrip route",
        member_ids=[target.memory_id],
    ))
    store.add_derived(DerivedRecord(
        cid="decoy-region",
        kind="gist",
        namespace=scope.namespace,
        level=1,
        text="family camping mountain",
        member_ids=[decoy.memory_id],
    ))

    hints = _memory_region_hints(
        store,
        "Where did Jamie's family go on a roadtrip?",
        [
            RetrievalCandidate(record=target, dense_score=0.8, fused_score=0.8),
            RetrievalCandidate(record=decoy, dense_score=0.7, fused_score=0.7),
        ],
        scope,
        at=150.0,
        limit=3,
    )

    assert [hint["region_id"] for hint in hints] == ["target-region"]
