"""Offline tests for ABSTENTION_V2_TAU calibration plumbing."""
from __future__ import annotations

import json

from bench.calibrate import (
    attach_calibration_metadata,
    calibrate_abstention_v2,
    _v2_samples_from_logs,
)


def test_calibrate_abstention_v2_meets_precision_target():
    samples = (
        [{"confidence": 0.9, "correct": True}] * 10
        + [{"confidence": 0.7, "correct": True}] * 2
        + [{"confidence": 0.6, "correct": False}] * 3
    )
    res = calibrate_abstention_v2(samples, precision_target=0.95)
    assert res["ok"]
    assert res["tau"] > 0.6
    assert res["precision_at_tau"] >= 0.95
    assert 0.0 < res["coverage_at_tau"] < 1.0


def test_calibrate_abstention_v2_abstains_all_when_unreachable():
    res = calibrate_abstention_v2(
        [{"confidence": 0.9, "correct": False}, {"confidence": 0.8, "correct": True}],
        precision_target=0.95,
    )
    assert res["ok"]
    assert res["tau"] > 0.9
    assert res["coverage_at_tau"] == 0.0


def test_v2_samples_filter_split_and_answerable(monkeypatch):
    # Force deterministic split labels without depending on hash partition details.
    import bench.calibrate as calibrate

    monkeypatch.setattr(calibrate, "split_of", lambda sid: "dev" if sid.startswith("dev") else "test")
    rows = [
        {"system": "eidetic-plus-full", "sample_id": "dev-1", "category": "temporal",
         "correct": True, "extra": {"confidence": 0.8}},
        {"system": "eidetic-plus-full", "sample_id": "test-1", "category": "temporal",
         "correct": False, "extra": {"confidence": 0.9}},
        {"system": "eidetic-plus-full", "sample_id": "dev-2", "category": "unanswerable",
         "correct": False, "extra": {"confidence": 0.1}},
        {"system": "rag-full", "sample_id": "dev-3", "category": "temporal",
         "correct": True, "extra": {"confidence": 0.7}},
    ]
    samples, excluded = _v2_samples_from_logs(rows, "eidetic-plus-full", split="dev")
    assert samples == [{"confidence": 0.8, "correct": True}]
    assert excluded == 1


def test_attach_calibration_metadata_fingerprints_source_logs(tmp_path):
    (tmp_path / "eidetic-plus-full__run0.jsonl").write_text(json.dumps({
        "system": "eidetic-plus-full",
        "sample_id": "dev-1",
        "correct": True,
        "extra": {"confidence": 0.9},
    }) + "\n")

    report = attach_calibration_metadata(
        {"ok": True, "tau": 0.7, "n": 1, "target": 0.95},
        logs=tmp_path,
        method="abstention_v2_tau",
        system="eidetic-plus-full",
        split="dev",
        excluded_by_split=2,
    )

    assert report["method"] == "abstention_v2_tau"
    assert report["split"] == "dev"
    assert report["system"] == "eidetic-plus-full"
    assert report["excluded_by_split"] == 2
    assert report["recommended_env"]["ABSTENTION_V2_TAU"] == "0.7"
    assert report["log_fingerprint"]["file_count"] == 1
    assert report["log_fingerprint"]["combined_sha256"]
