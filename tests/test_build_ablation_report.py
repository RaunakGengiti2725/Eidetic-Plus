from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from bench.build_ablation_report import build_ablation_report


SYSTEM = "eidetic-plus-full"


def _row(sample_id: str, run_idx: int, correct: bool, *, query_tokens: int = 100,
         verified_present: bool = True) -> dict:
    extra = {
        "structured_recall": True,
        "smqe_operator": "latest_value",
        "smqe_backend": "claim",
        "smqe_policy": "smqe:latest_value:claim",
        "policy": "smqe:latest_value:claim",
    }
    if verified_present:
        extra["verified"] = bool(correct)
    return {
        "system": SYSTEM,
        "dataset": "locomo",
        "category": "single-hop",
        "sample_id": sample_id,
        "question": "q",
        "gold": "g",
        "predicted": "g" if correct else "x",
        "correct": correct,
        "write_tokens": 10,
        "query_tokens": query_tokens,
        "search_ms": 1.0,
        "e2e_ms": 2.0,
        "abstained": False,
        "run_idx": run_idx,
        "age_days": 0.0,
        "n_sessions": 1,
        "extra": extra,
        "error": "",
    }


def _write_artifact(path: Path, correct_ids: set[str], *, query_tokens: int = 100,
                    split: str = "dev", verified_present: bool = True,
                    extra_sample: bool = False, affect_enabled: bool = True,
                    gist_enabled: bool = True,
                    extra_env: dict[str, str] | None = None) -> Path:
    path.mkdir(parents=True)
    sample_ids = [f"sample_{idx}" for idx in range(4)]
    if extra_sample:
        sample_ids.append("sample_extra")
    rows = []
    for run_idx in (0, 1):
        for sid in sample_ids:
            rows.append(
                _row(
                    sid,
                    run_idx,
                    sid in correct_ids,
                    query_tokens=query_tokens,
                    verified_present=verified_present,
                )
            )
    (path / f"{SYSTEM}__run0.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    env = {
        "AFFECT_SALIENCE": "1" if affect_enabled else "0",
        "GIST_CHANNEL": "1" if gist_enabled else "0",
        "DATA_DIR": str(path / "data"),
    }
    env.update(extra_env or {})
    (path / "run_manifest.json").write_text(json.dumps({
        "systems": SYSTEM,
        "dataset": "locomo",
        "split": split,
        "runs": 2,
        "run_offset": 0,
        "sample_count": len(sample_ids),
        "sample_rows": [
            {"dataset": "locomo", "category": "single-hop", "sample_id": sid}
            for sid in sample_ids
        ],
        "env": env,
    }))
    return path


def _write_three_artifacts(tmp_path: Path):
    full = _write_artifact(tmp_path / "full", {"sample_0", "sample_1", "sample_2"},
                           query_tokens=100)
    metabolism_off = _write_artifact(tmp_path / "metabolism_off", {"sample_0", "sample_1"},
                                     query_tokens=100, gist_enabled=False)
    regions_off = _write_artifact(tmp_path / "regions_off", {"sample_0", "sample_1"},
                                  query_tokens=100, gist_enabled=False)
    forgetting_off = _write_artifact(tmp_path / "forgetting_off", {"sample_0", "sample_1", "sample_2"},
                                     query_tokens=150)
    affect_off = _write_artifact(tmp_path / "affect_off", {"sample_0", "sample_1"},
                                 query_tokens=100, affect_enabled=False)
    return full, metabolism_off, regions_off, forgetting_off, affect_off


def test_build_ablation_report_from_real_logs_passes(tmp_path):
    full, metabolism_off, regions_off, forgetting_off, affect_off = _write_three_artifacts(tmp_path)

    report = build_ablation_report(
        full,
        metabolism_off,
        forgetting_off,
        affect_off,
        regions_off,
        system=SYSTEM,
        split="dev",
        min_samples=4,
        min_metabolism_accuracy_delta_pp=10.0,
        min_region_accuracy_delta_pp=10.0,
        min_affect_accuracy_delta_pp=10.0,
        min_forgetting_cost_ratio=1.2,
    )

    assert report["pass"] is True
    assert report["system"] == SYSTEM
    assert report["split"] == "dev"
    assert report["full"]["n"] == 4
    assert report["full"]["row_n"] == 8
    assert report["full"]["verified_accuracy"] == 0.75
    assert report["ablations"]["metabolism_off"]["verified_accuracy"] == 0.5
    assert report["ablations"]["regions_off"]["verified_accuracy"] == 0.5
    assert report["ablations"]["forgetting_off"]["query_tokens_median"] == 150
    assert report["ablations"]["affect_off"]["verified_accuracy"] == 0.5
    assert report["deltas"]["metabolism_delta_pp"] == 25.0
    assert report["deltas"]["region_delta_pp"] == 25.0
    assert report["deltas"]["affect_delta_pp"] == 25.0
    assert report["deltas"]["forgetting_cost_ratio"] == 1.5
    assert report["paired_coverage"]["exact_row_keys"] is True
    assert len(report["artifact_fingerprints"]) == 5
    assert all(ref["log_fingerprint"]["combined_sha256"] for ref in report["artifact_fingerprints"])


def test_build_ablation_report_rejects_test_split_for_dev_evidence(tmp_path):
    full, metabolism_off, regions_off, forgetting_off, affect_off = _write_three_artifacts(tmp_path)
    (full / "run_manifest.json").write_text(json.dumps({
        "systems": SYSTEM,
        "dataset": "locomo",
        "split": "test",
        "sample_count": 4,
        "env": {"AFFECT_SALIENCE": "1", "GIST_CHANNEL": "1", "DATA_DIR": str(full / "data")},
    }))

    report = build_ablation_report(
        full,
        metabolism_off,
        forgetting_off,
        affect_off,
        regions_off,
        system=SYSTEM,
        split="dev",
        min_samples=4,
    )

    assert report["pass"] is False
    assert any("full:split:test:expected:dev" in failure for failure in report["failures"])


def test_build_ablation_report_rejects_unpaired_rows(tmp_path):
    full, metabolism_off, regions_off, _forgetting_off, affect_off = _write_three_artifacts(tmp_path)
    forgetting_off = _write_artifact(
        tmp_path / "forgetting_extra",
        {"sample_0", "sample_1", "sample_2"},
        query_tokens=150,
        extra_sample=True,
    )

    report = build_ablation_report(
        full,
        metabolism_off,
        forgetting_off,
        affect_off,
        regions_off,
        system=SYSTEM,
        split="dev",
        min_samples=4,
    )

    assert report["pass"] is False
    assert any("forgetting_off:unpaired_rows" in failure for failure in report["failures"])


def test_build_ablation_report_requires_verified_metadata(tmp_path):
    full, metabolism_off, regions_off, _forgetting_off, affect_off = _write_three_artifacts(tmp_path)
    forgetting_off = _write_artifact(
        tmp_path / "forgetting_no_verified",
        {"sample_0", "sample_1", "sample_2"},
        query_tokens=150,
        verified_present=False,
    )

    report = build_ablation_report(
        full,
        metabolism_off,
        forgetting_off,
        affect_off,
        regions_off,
        system=SYSTEM,
        split="dev",
        min_samples=4,
    )

    assert report["pass"] is False
    assert any("verified_metadata_missing" in failure for failure in report["failures"])


def test_build_ablation_report_requires_affect_artifact(tmp_path):
    full, metabolism_off, regions_off, forgetting_off, _affect_off = _write_three_artifacts(tmp_path)

    report = build_ablation_report(
        full,
        metabolism_off,
        forgetting_off,
        regions_off_dir=regions_off,
        system=SYSTEM,
        split="dev",
        min_samples=4,
    )

    assert report["pass"] is False
    assert "affect_off:artifact_missing" in report["failures"]


def test_build_ablation_report_rejects_affect_env_drift(tmp_path):
    full, metabolism_off, regions_off, forgetting_off, _affect_off = _write_three_artifacts(tmp_path)
    affect_off = _write_artifact(
        tmp_path / "affect_drift",
        {"sample_0", "sample_1"},
        affect_enabled=False,
        extra_env={"READER_MODE": "different"},
    )

    report = build_ablation_report(
        full,
        metabolism_off,
        forgetting_off,
        affect_off,
        regions_off,
        system=SYSTEM,
        split="dev",
        min_samples=4,
    )

    assert report["pass"] is False
    assert any("affect_off:non_affect_env_drift:READER_MODE" in failure for failure in report["failures"])


def test_build_ablation_report_requires_region_artifact(tmp_path):
    full, metabolism_off, _regions_off, forgetting_off, affect_off = _write_three_artifacts(tmp_path)

    report = build_ablation_report(
        full,
        metabolism_off,
        forgetting_off,
        affect_off,
        system=SYSTEM,
        split="dev",
        min_samples=4,
    )

    assert report["pass"] is False
    assert "regions_off:artifact_missing" in report["failures"]


def test_build_ablation_report_rejects_weak_region_ablation(tmp_path):
    full, metabolism_off, regions_off, forgetting_off, affect_off = _write_three_artifacts(tmp_path)
    regions_off = _write_artifact(
        tmp_path / "regions_weak",
        {"sample_0", "sample_1", "sample_2"},
        query_tokens=100,
        gist_enabled=False,
    )

    report = build_ablation_report(
        full,
        metabolism_off,
        forgetting_off,
        affect_off,
        regions_off,
        system=SYSTEM,
        split="dev",
        min_samples=4,
        min_region_accuracy_delta_pp=10.0,
    )

    assert report["pass"] is False
    assert any("region_delta_pp" in failure for failure in report["failures"])


def test_build_ablation_report_cli_writes_sidecar(tmp_path):
    full, metabolism_off, regions_off, forgetting_off, affect_off = _write_three_artifacts(tmp_path)
    out = tmp_path / "ablation_report.json"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.build_ablation_report",
            "--full",
            str(full),
            "--metabolism-off",
            str(metabolism_off),
            "--regions-off",
            str(regions_off),
            "--forgetting-off",
            str(forgetting_off),
            "--affect-off",
            str(affect_off),
            "--system",
            SYSTEM,
            "--min-samples",
            "4",
            "--min-metabolism-accuracy-delta-pp",
            "10",
            "--min-region-accuracy-delta-pp",
            "10",
            "--min-affect-accuracy-delta-pp",
            "10",
            "--min-forgetting-cost-ratio",
            "1.2",
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["pass"] is True
    assert data["generated_by"] == "bench.build_ablation_report"
