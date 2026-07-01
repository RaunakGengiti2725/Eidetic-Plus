"""Offline tests for benchmark failure forensics."""
from __future__ import annotations

from bench.forensics import analyze, bucket_failure, render_markdown


def test_bucket_failure_uses_logged_signals():
    assert bucket_failure({"error": "ModelCallError: quota"}) == "infra"
    assert bucket_failure({"abstained": True}) == "abstention"
    assert bucket_failure({"correct": False, "extra": {"verified": True, "coverage": 0.9}}) == "reader_error"
    assert bucket_failure({"correct": False, "extra": {"coverage": 0.1}}) == "retrieval_miss"


def test_analyze_counts_buckets_and_filters_system(tmp_path):
    rows = [
        {"system": "eidetic-plus-full", "category": "temporal", "sample_id": "q1",
         "question": "when", "gold": "g", "predicted": "p", "correct": False,
         "dataset": "locomo", "run_idx": 0,
         "extra": {"coverage": 0.1, "consolidate": {"consolidate_pending": {
             "pending_processed": 3,
             "facts_extracted": 1,
             "events_indexed": 1,
             "extraction_timed_out": 2,
             "extraction_deferred": 0,
         }}}},
        {"system": "eidetic-plus-full", "category": "temporal", "sample_id": "q2",
         "question": "when", "gold": "g", "predicted": "p", "correct": True,
         "dataset": "locomo", "run_idx": 0,
         "extra": {"coverage": 0.9}},
        {"system": "rag-full", "category": "temporal", "sample_id": "q3",
         "question": "when", "gold": "g", "predicted": "p", "correct": False,
         "dataset": "locomo", "run_idx": 0,
         "extra": {"coverage": 0.9}},
    ]
    report = analyze(rows, system="eidetic-plus-full")
    assert report["rows"] == 2
    assert report["failures"] == 1
    assert report["bucket_counts"] == {"retrieval_miss": 1}
    assert report["consolidation"]["eidetic-plus-full"]["extraction_timed_out"] == 2
    out = render_markdown(report, tmp_path / "report.md")
    text = out.read_text()
    assert "Benchmark Failure Forensics" in text
    assert "Consolidation Health" in text
