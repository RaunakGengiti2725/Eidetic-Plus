from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.run_dev_ablation import (
    AFFECT_OFF_ENV,
    METABOLISM_OFF_ENV,
    REGIONS_OFF_ENV,
    _parse_env_pairs,
    build_run_specs,
    run_dev_ablation,
)


SYSTEM = "eidetic-plus-full"


def _row(sample_id: str, run_idx: int, correct: bool, query_tokens: int) -> dict:
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
        "extra": {
            "verified": bool(correct),
            "structured_recall": True,
            "smqe_operator": "latest_value",
            "smqe_backend": "claim",
            "smqe_policy": "smqe:latest_value:claim",
            "policy": "smqe:latest_value:claim",
        },
        "error": "",
    }


def _write_fake_artifact(path: Path, correct_ids: set[str], query_tokens: int,
                         env: dict[str, str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_idx in (0, 1):
        for idx in range(4):
            sid = f"sample_{idx}"
            rows.append(_row(sid, run_idx, sid in correct_ids, query_tokens))
    (path / f"{SYSTEM}__run0.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (path / "run_manifest.json").write_text(json.dumps({
        "systems": SYSTEM,
        "dataset": "locomo",
        "split": "dev",
        "runs": 2,
        "sample_count": 4,
        "sample_rows": [
            {"dataset": "locomo", "category": "single-hop", "sample_id": f"sample_{idx}"}
            for idx in range(4)
        ],
        "env": {
            "AFFECT_SALIENCE": env.get("AFFECT_SALIENCE", ""),
            "GIST_CHANNEL": env.get("GIST_CHANNEL", ""),
            "DATA_DIR": env.get("DATA_DIR", ""),
            "METABOLISM_MODE": env.get("METABOLISM_MODE", ""),
        },
    }))


def test_parse_env_pairs_accepts_repeated_and_comma_values():
    assert _parse_env_pairs(["A=1,B=2", "B=3", "EMPTY="]) == {
        "A": "1",
        "B": "3",
        "EMPTY": "",
    }


def test_parse_env_pairs_rejects_malformed_values():
    with pytest.raises(ValueError):
        _parse_env_pairs(["NOPE"])


def test_build_run_specs_use_one_samples_file_and_role_envs(tmp_path):
    samples = tmp_path / "dev.samples.json"
    specs = build_run_specs(
        out_root=tmp_path / "abl",
        samples_file=samples,
        systems="eidetic-full",
        dataset="both",
        variant="longmemeval_s",
        runs=3,
        overwrite=True,
        common_env={"READER_MODEL": "unit-reader"},
        full_env={"SALIENCE_PRUNE_THRESHOLD": "0.15"},
        forgetting_off_env={"SALIENCE_PRUNE_THRESHOLD": "0"},
    )

    assert [spec.role for spec in specs] == [
        "full", "metabolism_off", "regions_off", "forgetting_off", "affect_off"
    ]
    assert all(str(samples) in spec.command for spec in specs)
    assert all("--overwrite" in spec.command for spec in specs)
    assert specs[0].env_overrides["METABOLISM_MODE"] == "1"
    assert specs[0].env_overrides["AFFECT_SALIENCE"] == "1"
    assert specs[0].env_overrides["GIST_CHANNEL"] == "1"
    assert specs[0].env_overrides["SALIENCE_PRUNE_THRESHOLD"] == "0.15"
    assert specs[1].env_overrides["FULL_SLEEP"] == METABOLISM_OFF_ENV["FULL_SLEEP"]
    assert specs[2].env_overrides["GIST_CHANNEL"] == REGIONS_OFF_ENV["GIST_CHANNEL"]
    assert specs[3].env_overrides["SALIENCE_PRUNE_THRESHOLD"] == "0"
    assert specs[4].env_overrides["AFFECT_SALIENCE"] == AFFECT_OFF_ENV["AFFECT_SALIENCE"]
    assert len({spec.env_overrides["DATA_DIR"] for spec in specs}) == 5


def test_run_dev_ablation_builds_report_from_real_fake_logs(tmp_path):
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([
        {"dataset": "locomo", "sample_id": f"sample_{idx}"}
        for idx in range(4)
    ]))
    seen_roles: list[str] = []

    def fake_runner(spec, env, cwd):
        seen_roles.append(spec.role)
        assert env["DATA_DIR"].endswith(f"data_{spec.role}")
        if spec.role == "full":
            _write_fake_artifact(Path(spec.out_dir), {"sample_0", "sample_1", "sample_2"}, 100, env)
        elif spec.role == "metabolism_off":
            _write_fake_artifact(Path(spec.out_dir), {"sample_0", "sample_1"}, 100, env)
        elif spec.role == "regions_off":
            _write_fake_artifact(Path(spec.out_dir), {"sample_0", "sample_1"}, 100, env)
        elif spec.role == "forgetting_off":
            _write_fake_artifact(Path(spec.out_dir), {"sample_0", "sample_1", "sample_2"}, 150, env)
        elif spec.role == "affect_off":
            _write_fake_artifact(Path(spec.out_dir), {"sample_0", "sample_1"}, 100, env)
        return 0

    report = run_dev_ablation(
        out_root=tmp_path / "abl",
        report_out=tmp_path / "ablation_report.json",
        systems="eidetic-full",
        system_under_test=SYSTEM,
        dataset="locomo",
        samples_file=samples,
        runs=2,
        min_samples=4,
        min_metabolism_accuracy_delta_pp=10.0,
        min_affect_accuracy_delta_pp=10.0,
        min_forgetting_cost_ratio=1.2,
        command_runner=fake_runner,
    )

    assert seen_roles == ["full", "metabolism_off", "regions_off", "forgetting_off", "affect_off"]
    assert report["pass"] is True
    assert report["generated_by"] == "bench.run_dev_ablation"
    assert report["deltas"]["metabolism_delta_pp"] == 25.0
    assert report["deltas"]["region_delta_pp"] == 25.0
    assert report["deltas"]["affect_delta_pp"] == 25.0
    assert report["deltas"]["forgetting_cost_ratio"] == 1.5
    assert report["forgetting_cost_profiles"] == {
        "full": {"SALIENCE_PRUNE_THRESHOLD": 0.0, "DREAM_PRUNE_PERCENTILE": 5.0},
        "forgetting_off": {"SALIENCE_PRUNE_THRESHOLD": 0.0, "DREAM_PRUNE_PERCENTILE": 0.0},
    }
    assert report["samples_file"] == str(samples)
    assert len(report["run_specs"]) == 5
    assert json.loads((tmp_path / "ablation_report.json").read_text())["pass"] is True


def test_run_dev_ablation_rejects_disabled_full_affect_before_running(tmp_path):
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"dataset": "locomo", "sample_id": "sample_0"}]))

    def fake_runner(spec, env, cwd):
        raise AssertionError("preflight should fail before subprocess execution")

    report = run_dev_ablation(
        out_root=tmp_path / "abl",
        report_out=tmp_path / "ablation_report.json",
        samples_file=samples,
        common_env={"AFFECT_SALIENCE": "0"},
        min_samples=1,
        command_runner=fake_runner,
    )

    assert report["pass"] is False
    assert any("full:AFFECT_SALIENCE:0:expected:on" in item for item in report["failures"])


def test_run_dev_ablation_rejects_identical_forgetting_cost_profile_before_running(tmp_path):
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"dataset": "locomo", "sample_id": "sample_0"}]))
    seen_roles: list[str] = []

    def fake_runner(spec, env, cwd):
        seen_roles.append(spec.role)
        return 0

    report = run_dev_ablation(
        out_root=tmp_path / "abl",
        report_out=tmp_path / "ablation_report.json",
        samples_file=samples,
        common_env={"SALIENCE_PRUNE_THRESHOLD": "0", "DREAM_PRUNE_PERCENTILE": "0"},
        min_samples=1,
        command_runner=fake_runner,
    )

    assert seen_roles == []
    assert report["pass"] is False
    assert any("forgetting_off:identical_cost_profile" in item for item in report["failures"])
    assert report["forgetting_cost_profiles"] == {
        "full": {"SALIENCE_PRUNE_THRESHOLD": 0.0, "DREAM_PRUNE_PERCENTILE": 0.0},
        "forgetting_off": {"SALIENCE_PRUNE_THRESHOLD": 0.0, "DREAM_PRUNE_PERCENTILE": 0.0},
    }
    saved = json.loads((tmp_path / "ablation_report.json").read_text())
    assert saved["status"] == "FAIL"
    assert saved["samples_file"] == str(samples)


def test_run_dev_ablation_rejects_parent_env_identical_forgetting_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("SALIENCE_PRUNE_THRESHOLD", "0")
    monkeypatch.setenv("DREAM_PRUNE_PERCENTILE", "0")
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"dataset": "locomo", "sample_id": "sample_0"}]))

    def fake_runner(spec, env, cwd):
        raise AssertionError("preflight should fail before subprocess execution")

    report = run_dev_ablation(
        out_root=tmp_path / "abl",
        report_out=tmp_path / "ablation_report.json",
        samples_file=samples,
        min_samples=1,
        command_runner=fake_runner,
    )

    assert report["pass"] is False
    assert any("identical_cost_profile" in item for item in report["failures"])


def test_run_dev_ablation_rejects_inverted_forgetting_cost_profile(tmp_path):
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"dataset": "locomo", "sample_id": "sample_0"}]))

    def fake_runner(spec, env, cwd):
        raise AssertionError("preflight should fail before subprocess execution")

    report = run_dev_ablation(
        out_root=tmp_path / "abl",
        report_out=tmp_path / "ablation_report.json",
        samples_file=samples,
        forgetting_off_env={"DREAM_PRUNE_PERCENTILE": "6"},
        min_samples=1,
        command_runner=fake_runner,
    )

    assert report["pass"] is False
    assert any("forgetting_off:inverted_cost_profile" in item for item in report["failures"])
    assert report["forgetting_cost_profiles"]["forgetting_off"]["DREAM_PRUNE_PERCENTILE"] == 6.0


def test_run_dev_ablation_rejects_invalid_forgetting_cost_profile(tmp_path):
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"dataset": "locomo", "sample_id": "sample_0"}]))

    def fake_runner(spec, env, cwd):
        raise AssertionError("preflight should fail before subprocess execution")

    report = run_dev_ablation(
        out_root=tmp_path / "abl",
        report_out=tmp_path / "ablation_report.json",
        samples_file=samples,
        full_env={"DREAM_PRUNE_PERCENTILE": "definitely-not-a-number"},
        min_samples=1,
        command_runner=fake_runner,
    )

    assert report["pass"] is False
    assert any("forgetting_off:invalid_cost_profile" in item for item in report["failures"])
    assert report["forgetting_cost_profiles"] == {}


def test_run_dev_ablation_writes_fail_closed_report_when_a_run_fails(tmp_path):
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"dataset": "locomo", "sample_id": "sample_0"}]))

    def fake_runner(spec, env, cwd):
        return 7 if spec.role == "forgetting_off" else 0

    report = run_dev_ablation(
        out_root=tmp_path / "abl",
        report_out=tmp_path / "ablation_report.json",
        samples_file=samples,
        min_samples=1,
        command_runner=fake_runner,
    )

    assert report["pass"] is False
    assert "forgetting_off:bench.run exited 7" in report["failures"]
    assert report["forgetting_cost_profiles"]["forgetting_off"]["DREAM_PRUNE_PERCENTILE"] == 0.0
    assert json.loads((tmp_path / "ablation_report.json").read_text())["status"] == "FAIL"
