"""Offline tests for the Benchmark Theater report (Track 5.4): bundles accuracy-by-category,
cost/latency, paired significance, AND the product-specific metrics (verified recall rate,
abstention rate, age-flatness slope) from the raw logs' `extra`/`age_days` fields. Renders
ONLY from real logs -- a pending placeholder when there are none, never invented numbers."""
from __future__ import annotations

import json

import pytest


def test_report_pending_without_logs(tmp_path):
    from bench.report import build_report
    p = build_report(tmp_path)
    assert p.exists()
    assert "Pending" in p.read_text()


def _row(sysname, sid, correct, *, abstained=False, verified=False, age=None, cat="single-hop"):
    return {"system": sysname, "dataset": "locomo", "category": cat, "sample_id": sid,
            "question": "q", "gold": "g", "predicted": "p", "correct": correct,
            "write_tokens": 100, "query_tokens": 50, "search_ms": 10.0, "e2e_ms": 120.0,
            "abstained": abstained, "run_idx": 0, "age_days": age,
            "extra": {"verified": verified}}


def test_report_computes_product_metrics(tmp_path):
    from bench.report import build_report
    rows = [
        _row("eidetic-plus-full", "c_q0", True, verified=True, age=0.0),
        _row("eidetic-plus-full", "c_q1", True, verified=True, age=365.0),
        _row("eidetic-plus-full", "c_q2", False, abstained=True, verified=False, age=100.0),
        _row("eidetic-plus-full", "c_q3", True, verified=False, age=200.0),
        _row("rag-vector", "c_q0", True, verified=False, age=0.0),
        _row("rag-vector", "c_q1", False, verified=False, age=365.0),
    ]
    rows[0]["extra"]["consolidate"] = {"consolidate_pending": {
        "pending_processed": 4,
        "facts_extracted": 2,
        "events_indexed": 1,
        "extraction_timed_out": 1,
        "extraction_deferred": 0,
    }}
    rows[0]["extra"].update({
        "structured_recall": True,
        "smqe_operator": "latest_value",
        "smqe_backend": "claim",
        "smqe_policy": "smqe:latest_value:claim",
        "policy": "smqe:latest_value:claim",
    })
    rows[1]["extra"].update({
        "structured_recall": True,
        "smqe_operator": "count_aggregate",
        "smqe_backend": "record",
        "smqe_policy": "smqe:count_aggregate:record",
        "policy": "smqe:count_aggregate:record",
    })
    rows[2]["extra"]["policy"] = "fixed-reader + verify+abstain+proof"
    rows[3]["extra"]["policy"] = "legacy source-scan rescue"
    (tmp_path / "eidetic-plus-full__run0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows if r["system"] == "eidetic-plus-full"))
    (tmp_path / "rag-vector__run0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows if r["system"] == "rag-vector"))

    p = build_report(tmp_path, {"judge_model": "qwen3-max", "judge_backend": "dashscope"})
    assert p.exists()
    text = p.read_text()
    assert "eidetic-plus-full" in text and "rag-vector" in text
    assert "Consolidation health" in text
    assert "SMQE Backend Mix" in text

    data = json.loads((tmp_path / "report.json").read_text())
    pm = data["product_metrics"]["eidetic-plus-full"]
    assert pm["n"] == 4
    assert pm["accuracy"] == pytest.approx(0.75)
    assert pm["abstention_rate"] == pytest.approx(0.25)
    assert pm["verified_rate"] == pytest.approx(0.5)
    assert pm["accuracy_on_answered"] == pytest.approx(1.0)   # the 3 answered were all correct
    assert "age_flatness_slope_per_year" in pm
    assert data["consolidation"]["eidetic-plus-full"]["extraction_timed_out"] == 1
    assert data["smqe"]["eidetic-plus-full"]["structured"] == 2
    assert data["smqe"]["eidetic-plus-full"]["claim"] == 1
    assert data["smqe"]["eidetic-plus-full"]["record"] == 1
    assert data["smqe"]["eidetic-plus-full"]["fallback"] == 2
    assert data["smqe"]["eidetic-plus-full"]["legacy_policy_rows"] == 1
    assert data["smqe"]["eidetic-plus-full"]["operators"]["latest_value"] == 1
    # baseline has no verified/abstention signal -> rates 0, still reported.
    assert data["product_metrics"]["rag-vector"]["verified_rate"] == pytest.approx(0.0)


def test_report_summarizes_region_routing_telemetry(tmp_path):
    from bench.report import build_report, region_metrics
    rows = [
        _row("eidetic-plus-full", "c_q0", True),
        _row("eidetic-plus-full", "c_q1", True),
        _row("eidetic-plus-full", "c_q2", False),
        _row("eidetic-plus-full", "c_q3", True),
    ]
    rows[0]["extra"].update({
        "region_hint_count": 2,
        "region_ids": ["r1", "r2"],
        "region_member_ids": ["m1", "m2", "m1"],
    })
    rows[1]["extra"].update({
        "region_hint_count": 0,
        "region_ids": [],
        "region_member_ids": [],
    })
    rows[2]["extra"].update({
        "region_hint_count": "bad",
        "region_ids": "r3",
        "region_member_ids": [],
    })
    (tmp_path / "eidetic-plus-full__run0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))

    direct = region_metrics(rows)["eidetic-plus-full"]
    assert direct["rows"] == 4
    assert direct["telemetry_rows"] == 3
    assert direct["missing_rows"] == 1
    assert direct["hint_rows"] == 1
    assert direct["hint_row_rate"] == pytest.approx(0.25)
    assert direct["total_hints"] == 2
    assert direct["unique_region_ids"] == 2
    assert direct["unique_region_member_ids"] == 2
    assert direct["malformed_rows"] == 1
    assert direct["malformed_samples"] == ["c_q2"]

    build_report(tmp_path)
    text = (tmp_path / "report.md").read_text()
    assert "Region Routing Telemetry" in text
    assert "| eidetic-plus-full | 4 | 3 | 1 | 25.0 | 2 | 2 | 2 | 1 | 1 |" in text
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["region_telemetry"]["eidetic-plus-full"] == direct


def test_report_references_manifest_and_logs(tmp_path):
    from bench.report import build_report
    (tmp_path / "eidetic-plus-full__run0.jsonl").write_text(
        json.dumps(_row("eidetic-plus-full", "c_q0", True, verified=True, age=1.0)) + "\n")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"systems": "eidetic-full", "split": "test"}))
    build_report(tmp_path)
    text = (tmp_path / "report.md").read_text()
    assert "run_manifest.json" in text          # points at the reproduction manifest
    assert "false-premise" in text.lower()      # documents the metric that needs a labeled set
