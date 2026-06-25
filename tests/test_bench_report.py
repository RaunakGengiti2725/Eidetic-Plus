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
    (tmp_path / "eidetic-plus-full__run0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows if r["system"] == "eidetic-plus-full"))
    (tmp_path / "rag-vector__run0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows if r["system"] == "rag-vector"))

    p = build_report(tmp_path, {"judge_model": "qwen3-max", "judge_backend": "dashscope"})
    assert p.exists()
    text = p.read_text()
    assert "eidetic-plus-full" in text and "rag-vector" in text

    data = json.loads((tmp_path / "report.json").read_text())
    pm = data["product_metrics"]["eidetic-plus-full"]
    assert pm["n"] == 4
    assert pm["accuracy"] == pytest.approx(0.75)
    assert pm["abstention_rate"] == pytest.approx(0.25)
    assert pm["verified_rate"] == pytest.approx(0.5)
    assert pm["accuracy_on_answered"] == pytest.approx(1.0)   # the 3 answered were all correct
    assert "age_flatness_slope_per_year" in pm
    # baseline has no verified/abstention signal -> rates 0, still reported.
    assert data["product_metrics"]["rag-vector"]["verified_rate"] == pytest.approx(0.0)


def test_report_references_manifest_and_logs(tmp_path):
    from bench.report import build_report
    (tmp_path / "eidetic-plus-full__run0.jsonl").write_text(
        json.dumps(_row("eidetic-plus-full", "c_q0", True, verified=True, age=1.0)) + "\n")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"systems": "eidetic-full", "split": "test"}))
    build_report(tmp_path)
    text = (tmp_path / "report.md").read_text()
    assert "run_manifest.json" in text          # points at the reproduction manifest
    assert "false-premise" in text.lower()      # documents the metric that needs a labeled set
