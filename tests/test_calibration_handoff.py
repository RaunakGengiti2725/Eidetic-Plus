"""Offline tests for publishing dev calibration into public benchmark artifacts."""
from __future__ import annotations

import json

from bench.calibration_handoff import (
    CalibrationHandoffError,
    copy_abstention_v2_tau_report,
    env_vars_for_report,
    main,
    validate_abstention_v2_tau_report,
)


def _report(**overrides):
    data = {
        "ok": True,
        "method": "abstention_v2_tau",
        "split": "dev",
        "system": "eidetic-plus-full",
        "tau": 0.7,
        "n": 50,
        "target": 0.95,
        "precision_at_tau": 1.0,
        "coverage_at_tau": 0.8,
        "log_fingerprint": {
            "combined_sha256": "a" * 64,
            "file_count": 3,
            "files": [],
        },
    }
    data.update(overrides)
    return data


def test_copy_abstention_v2_tau_report_writes_release_artifact_and_env(tmp_path):
    src = tmp_path / "cal_dev" / "abstention_v2_tau.json"
    src.parent.mkdir()
    src.write_text(json.dumps(_report()))
    out = tmp_path / "bench"
    env_out = tmp_path / "guard" / "abstention.env"

    result = copy_abstention_v2_tau_report(src, out, env_out=env_out, min_samples=50)

    copied = json.loads((out / "abstention_v2_tau.json").read_text())
    assert copied["method"] == "abstention_v2_tau"
    assert copied["split"] == "dev"
    assert result["dest"] == out / "abstention_v2_tau.json"
    assert result["env_vars"] == {"ABSTENTION_V2": "1", "ABSTENTION_V2_TAU": "0.7"}
    assert env_out.read_text() == "ABSTENTION_V2=1\nABSTENTION_V2_TAU=0.7\n"


def test_validate_abstention_v2_tau_report_rejects_non_dev_report():
    try:
        validate_abstention_v2_tau_report(_report(split="test"))
    except CalibrationHandoffError as exc:
        assert "split must be dev" in str(exc)
    else:
        raise AssertionError("expected invalid split to be rejected")


def test_validate_abstention_v2_tau_report_rejects_missing_fingerprint():
    try:
        validate_abstention_v2_tau_report(_report(log_fingerprint={}))
    except CalibrationHandoffError as exc:
        assert "combined_sha256" in str(exc)
    else:
        raise AssertionError("expected missing fingerprint to be rejected")


def test_env_vars_for_report_uses_exact_report_tau_string():
    assert env_vars_for_report(_report(tau=0.7000000000000001)) == {
        "ABSTENTION_V2": "1",
        "ABSTENTION_V2_TAU": "0.7000000000000001",
    }


def test_cli_copies_report_and_writes_env_file(tmp_path):
    src = tmp_path / "abstention_v2_tau.json"
    src.write_text(json.dumps(_report()))
    out = tmp_path / "bench"
    env_out = tmp_path / "abstention.env"

    assert main([
        "--calibration", str(src),
        "--out", str(out),
        "--env-out", str(env_out),
        "--min-samples", "50",
    ]) == 0
    assert (out / "abstention_v2_tau.json").exists()
    assert "ABSTENTION_V2_TAU=0.7" in env_out.read_text()


def test_cli_returns_error_for_invalid_report(tmp_path):
    src = tmp_path / "abstention_v2_tau.json"
    src.write_text(json.dumps(_report(ok=False)))

    assert main(["--calibration", str(src), "--out", str(tmp_path / "bench")]) == 2
