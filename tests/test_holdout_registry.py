from __future__ import annotations

import json
from types import SimpleNamespace

from bench.audit_no_holdout_leakage import DEFAULT_SCAN_ROOTS, audit
from bench import build_holdout_registry


def _write_empty_registry(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "longmemeval_test_holdout.json").write_text("[]")
    (root / "locomo_test_holdout.json").write_text("[]")
    (root / "leaked_sample_ids.json").write_text("[]")
    (root / "manifest.json").write_text("{}")


def test_holdout_audit_default_roots_include_docs():
    assert DEFAULT_SCAN_ROOTS == ("eidetic", "bench", "tests", "docs")


def test_holdout_audit_checks_short_sample_ids(tmp_path):
    holdout = tmp_path / "holdout"
    _write_empty_registry(holdout)
    (holdout / "locomo_test_holdout.json").write_text(json.dumps(["z9_q7"]))
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    (safe_root / "x.py").write_text("SAFE = True\n")
    leaked_root = tmp_path / "leaked"
    leaked_root.mkdir()
    (leaked_root / "x.py").write_text('CASE = "z9_q7"\n')

    safe = audit(holdout, [safe_root], include_legacy_policy=False)
    leaked = audit(holdout, [leaked_root], include_legacy_policy=False)

    assert safe["pass"] is True
    assert safe["holdout_needles_checked"] == 1
    assert safe["legacy_policy_scan_enabled"] is False
    assert safe["forbidden_runtime_symbols_checked"] == 0
    assert leaked["pass"] is False
    assert leaked["findings"][0]["needle"] == "z9_q7"


def test_holdout_audit_reports_legacy_shortcut_scan_coverage(tmp_path):
    holdout = tmp_path / "holdout"
    _write_empty_registry(holdout)
    (holdout / "locomo_test_holdout.json").write_text(json.dumps(["z9_q7"]))
    root = tmp_path / "safe"
    root.mkdir()
    (root / "x.py").write_text("SAFE = True\n")

    report = audit(holdout, [root])

    assert report["pass"] is True
    assert report["legacy_policy_scan_enabled"] is True
    assert report["forbidden_policy_strings_checked"] > 0
    assert report["forbidden_fixed_answer_strings_checked"] > 0
    assert report["forbidden_runtime_symbols_checked"] > 0
    assert report["scan_roots"] == [str(root)]


def test_build_holdout_registry_uses_exact_requested_samples(tmp_path, monkeypatch):
    samples = [
        SimpleNamespace(dataset="locomo", sample_id="z9_q7"),
        SimpleNamespace(dataset="locomo", sample_id="y8_q6"),
        SimpleNamespace(dataset="longmemeval", sample_id="private_lme_a"),
        SimpleNamespace(dataset="longmemeval", sample_id="private_lme_b"),
    ]
    monkeypatch.setattr(build_holdout_registry, "load_samples", lambda *args, **kwargs: samples)
    samples_file = tmp_path / "samples.json"
    samples_file.write_text(json.dumps([
        {"dataset": "locomo", "sample_id": "y8_q6"},
        {"dataset": "longmemeval", "sample_id": "private_lme_a"},
    ]))

    report = build_holdout_registry.build_registry(
        out_dir=tmp_path / "holdout",
        dataset="both",
        split="test",
        samples_file=str(samples_file),
    )

    assert report["pass"] is True
    assert report["counts"] == {"longmemeval": 1, "locomo": 1}
    assert json.loads((tmp_path / "holdout" / "locomo_test_holdout.json").read_text()) == ["y8_q6"]
    assert json.loads((tmp_path / "holdout" / "longmemeval_test_holdout.json").read_text()) == ["private_lme_a"]
    manifest = json.loads((tmp_path / "holdout" / "manifest.json").read_text())
    assert manifest["registry_kind"] == "holdout_sample_id_needles"
    assert manifest["samples_file"] == str(samples_file)
