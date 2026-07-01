from __future__ import annotations

import json
from types import SimpleNamespace

import bench.run_slice_invariant_eval as slice_eval


def _samples():
    rows = []
    for category in ("alpha", "beta"):
        for idx in range(6):
            rows.append(SimpleNamespace(
                dataset="synthetic",
                sample_id=f"{category}-{idx}",
                category=category,
            ))
    return rows


def test_slice_invariant_plan_only_uses_random_seed_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(slice_eval, "load_samples", lambda *args, **kwargs: _samples())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_slice_invariant_eval",
            "--dataset", "locomo",
            "--draws", "3",
            "--subset", "4",
            "--out", str(tmp_path),
            "--plan-only",
        ],
    )

    assert slice_eval.main() == 0
    report = json.loads((tmp_path / "slice_invariant.json").read_text())

    assert report["seed_mode"] == "random"
    assert report["split"] == "test"
    assert report["holdout_profile"] == "holdout"
    assert report["pass"] is False
    assert report["failures"] == ["draw1:not_executed", "draw2:not_executed", "draw3:not_executed"]
    assert report["pool_unique_sample_ids"] == 12
    assert report["required_unique_sample_ids"] == 12
    assert report["unique_sample_ids"] == 12
    assert isinstance(report["seed"], int)
    assert len(report["draw_seeds"]) == 3
    assert len(set(report["draw_seeds"])) == 3
    assert [run["seed"] for run in report["runs"]] == report["draw_seeds"]


def test_slice_invariant_plan_only_marks_explicit_seed_as_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(slice_eval, "load_samples", lambda *args, **kwargs: _samples())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_slice_invariant_eval",
            "--dataset", "locomo",
            "--draws", "2",
            "--subset", "3",
            "--seed", "12345",
            "--out", str(tmp_path),
            "--plan-only",
        ],
    )

    assert slice_eval.main() == 0
    report = json.loads((tmp_path / "slice_invariant.json").read_text())

    assert report["seed"] == 12345
    assert report["seed_mode"] == "fixed"
    assert report["pass"] is False
    assert report["pool_unique_sample_ids"] == 12
    assert report["required_unique_sample_ids"] == 6
    assert report["unique_sample_ids"] == 6
    assert report["draw_seeds"] == [12345, 12346]


def test_slice_invariant_rejects_insufficient_unique_pool_before_running(tmp_path, monkeypatch):
    monkeypatch.setattr(slice_eval, "load_samples", lambda *args, **kwargs: _samples())

    def fake_run(cmd, text, stdout, stderr):
        raise AssertionError("preflight should fail before subprocess execution")

    monkeypatch.setattr(slice_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_slice_invariant_eval",
            "--dataset", "locomo",
            "--draws", "4",
            "--subset", "4",
            "--out", str(tmp_path),
        ],
    )

    assert slice_eval.main() == 1
    report = json.loads((tmp_path / "slice_invariant.json").read_text())

    assert report["pass"] is False
    assert report["pool_unique_sample_ids"] == 12
    assert report["required_unique_sample_ids"] == 16
    assert report["unique_sample_ids"] == 12
    assert report["runs"] == []
    assert report["failures"] == ["insufficient_pool_unique_samples:12<required:16"]
    assert not list(tmp_path.glob("draw_*.samples.json"))


def test_slice_invariant_executed_run_records_holdout_split_and_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(slice_eval, "load_samples", lambda *args, **kwargs: _samples())

    def fake_run(cmd, text, stdout, stderr):
        out_dir = tmp_path
        for idx, value in enumerate(cmd):
            if value == "--out":
                out_dir = tmp_path / cmd[idx + 1]
                break
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "scoreboard.json").write_text(json.dumps({
            "systems": ["eidetic-plus-full"],
            "integrity": {
                "eidetic-plus-full": {
                    "n": 4,
                    "verified_correct": 4,
                    "has_verify": True,
                },
            },
        }))
        return SimpleNamespace(returncode=0, stdout="ok\n")

    monkeypatch.setattr(slice_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_slice_invariant_eval",
            "--dataset", "locomo",
            "--draws", "1",
            "--subset", "4",
            "--out", str(tmp_path),
        ],
    )

    assert slice_eval.main() == 0
    report = json.loads((tmp_path / "slice_invariant.json").read_text())

    assert report["pass"] is True
    assert report["failures"] == []
    assert report["split"] == "test"
    assert report["holdout_profile"] == "holdout"
    assert report["pool_unique_sample_ids"] == 12
    assert report["required_unique_sample_ids"] == 4
    assert report["runs"][0]["executed"] is True
    assert report["runs"][0]["score"]["pass"] is True
    assert report["runs"][0]["score"]["verified_correct"] == 4
    assert report["unique_sample_ids"] == 4


def test_slice_invariant_rejects_correct_only_scoreboard(tmp_path, monkeypatch):
    monkeypatch.setattr(slice_eval, "load_samples", lambda *args, **kwargs: _samples())

    def fake_run(cmd, text, stdout, stderr):
        out_dir = tmp_path
        for idx, value in enumerate(cmd):
            if value == "--out":
                out_dir = tmp_path / cmd[idx + 1]
                break
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "scoreboard.json").write_text(json.dumps({
            "systems": [{"system": "eidetic-plus-full", "correct": 4, "total": 4}],
        }))
        return SimpleNamespace(returncode=0, stdout="ok\n")

    monkeypatch.setattr(slice_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_slice_invariant_eval",
            "--dataset", "locomo",
            "--draws", "1",
            "--subset", "4",
            "--out", str(tmp_path),
        ],
    )

    assert slice_eval.main() == 1
    report = json.loads((tmp_path / "slice_invariant.json").read_text())

    assert report["pass"] is False
    assert report["runs"][0]["score"]["verified_correct"] == 0
    assert report["failures"] == ["draw1:score:false"]
