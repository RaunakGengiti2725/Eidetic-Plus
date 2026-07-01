from __future__ import annotations

import json

from bench.claim_scope import build_claim_scope, write_claim_scope


def test_claim_scope_reports_limited_measured_harness_scope(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "split": "test",
        "subset": 3,
        "sample_count": 3,
        "runs": 1,
    }))
    (tmp_path / "eidetic-plus-full__run0.jsonl").write_text(json.dumps({
        "system": "eidetic-plus-full",
        "dataset": "longmemeval",
        "sample_id": "s0",
        "run_idx": 0,
        "correct": True,
    }) + "\n")
    (tmp_path / "rag-full__run0.jsonl").write_text(json.dumps({
        "system": "rag-full",
        "dataset": "longmemeval",
        "sample_id": "s0",
        "run_idx": 0,
        "correct": False,
    }) + "\n")

    report = build_claim_scope(tmp_path)
    out = write_claim_scope(tmp_path, report)

    assert out.name == "claim_scope.json"
    assert report["public_claim_scope"] == "measured-harness-only"
    assert report["measured_harness_systems"] == ["eidetic-plus-full", "rag-full"]
    assert report["external_system_evidence"] == []
    assert report["datasets"] == ["longmemeval"]
    assert any("Only 1 run" in item for item in report["limitations"])
    assert any("Not a SOTA/best-in-world claim" in item for item in report["limitations"])
    assert any("mem0 was not measured" in item for item in report["limitations"])


def test_claim_scope_warns_from_logs_when_manifest_sample_count_missing(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "split": "test",
        "subset": 0,
        "runs": 10,
    }))
    rows = [
        {
            "system": system,
            "dataset": "longmemeval",
            "category": category,
            "sample_id": sample_id,
            "run_idx": run_idx,
            "correct": True,
        }
        for system in ("eidetic-plus-full", "rag-vector")
        for run_idx in (0, 1)
        for sample_id, category in (("s0", "single-hop"), ("s1", "multi-hop"))
    ]
    (tmp_path / "eidetic-plus-full__run0.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )

    report = build_claim_scope(tmp_path)

    assert report["manifest_sample_count"] is None
    assert report["log_sample_count"] == 2
    assert report["effective_sample_count"] == 2
    assert any("Only 2 unique benchmark sample" in item for item in report["limitations"])


def test_claim_scope_requires_external_evidence_for_top_system_coverage(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "split": "test",
        "subset": 100,
        "sample_count": 100,
        "runs": 10,
    }))
    (tmp_path / "eidetic-plus__run0.jsonl").write_text(json.dumps({
        "system": "eidetic-plus",
        "dataset": "longmemeval",
        "sample_id": "s0",
        "run_idx": 0,
        "correct": True,
    }) + "\n")

    names_only = build_claim_scope(
        tmp_path,
        public_claim_scope="best-in-world",
        measured_external_systems=["chronos", "mastra", "byterover", "hindsight"],
    )

    assert any("missing top-system measurements" in item for item in names_only["limitations"])

    evidence = [
        {
            "system": "chronos",
            "dataset": "longmemeval",
            "split": "test",
            "n": 500,
            "runs": 10,
            "score": 0.956,
            "date": "2026-06-29",
            "source": "published paper",
            "artifact_fingerprint": "sha256:chronos",
        },
        {
            "system": "mastra",
            "dataset": "longmemeval",
            "split": "test",
            "n": 500,
            "runs": 10,
            "score": 0.954,
            "date": "2026-06-29",
            "source": "published paper",
            "artifact_fingerprint": "sha256:mastra",
        },
        {
            "system": "byterover",
            "dataset": "longmemeval",
            "split": "test",
            "n": 500,
            "runs": 10,
            "score": 0.961,
            "date": "2026-06-29",
            "source": "published paper",
            "artifact_fingerprint": "sha256:byterover",
        },
        {
            "system": "hindsight",
            "dataset": "longmemeval",
            "split": "test",
            "n": 500,
            "runs": 10,
            "score": 0.940,
            "date": "2026-06-29",
            "source": "published paper",
            "artifact_fingerprint": "sha256:hindsight",
        },
    ]
    with_evidence = build_claim_scope(
        tmp_path,
        public_claim_scope="best-in-world",
        external_system_evidence=evidence,
    )

    assert not any("missing top-system measurements" in item for item in with_evidence["limitations"])
    assert {item["system"] for item in with_evidence["external_system_evidence"]} == {
        "chronos",
        "mastra",
        "byterover",
        "hindsight",
    }
