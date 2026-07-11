from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bench.replay import replay_sources, verify_replay_artifact, write_replay_artifact
from eidetic.models import ABSTENTION_TEXT


def _row(sample_id: str, *, verified: bool, abstained: bool,
         correct: bool, valid_proof: bool = True) -> dict:
    content_hash = "a" * 64
    proof = {
        "citations": 1,
        "entailed_memory_ids": ["mem_1"],
        "entailed_content_hashes": [content_hash],
        "entailed_raw_uris": [f"cas://{content_hash}"],
        "proof_surface_tokens": 12,
    } if valid_proof else {
        "citations": 0,
        "entailed_memory_ids": [],
        "entailed_content_hashes": [],
        "entailed_raw_uris": [],
        "proof_surface_tokens": 0,
    }
    return {
        "system": "eidetic-plus-full",
        "dataset": "locomo",
        "category": "single-hop",
        "sample_id": sample_id,
        "question": f"question {sample_id}",
        "gold": f"gold {sample_id}",
        "predicted": f"answer {sample_id}" if not abstained else ABSTENTION_TEXT,
        "correct": correct,
        "abstained": abstained,
        "extra": {"verified": verified, **proof},
    }


def _write_source(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def test_replay_converts_every_unverified_delivery_and_preserves_verified_rows(tmp_path):
    rows = [
        _row("verified-correct", verified=True, abstained=False, correct=True),
        _row("verified-wrong", verified=True, abstained=False, correct=False),
        _row("unverified-correct", verified=False, abstained=False, correct=True),
        _row("unverified-wrong", verified=False, abstained=False, correct=False),
        _row("already-abstained", verified=False, abstained=True, correct=False),
    ]
    source = _write_source(tmp_path / "window" / "eidetic-plus-full__run0.jsonl", rows)

    replayed, report = replay_sources([source])

    by_id = {row["sample_id"]: row for row in replayed}
    assert report["status"] == "PASS"
    assert report["aggregate"]["original_unverified_answered"] == 2
    assert report["aggregate"]["replay_unverified_answered"] == 0
    assert report["aggregate"]["converted_unverified"] == 2
    assert report["aggregate"]["converted_unverified_correct"] == 1
    assert report["aggregate"]["converted_unverified_wrong"] == 1
    assert report["verified_correct_regressions"] == 0
    assert by_id["verified-correct"]["predicted"] == "answer verified-correct"
    assert by_id["verified-correct"]["correct"] is True
    assert by_id["verified-correct"]["status"] == "VERIFIED"
    assert by_id["verified-wrong"]["status"] == "VERIFIED"
    for sample_id in ("unverified-correct", "unverified-wrong", "already-abstained"):
        assert by_id[sample_id]["status"] == "ABSTAINED"
        assert by_id[sample_id]["predicted"] == ABSTENTION_TEXT
        assert by_id[sample_id]["correct"] is False
        assert by_id[sample_id]["extra"]["citations"] == 0
        assert by_id[sample_id]["extra"]["entailed_memory_ids"] == []


def test_replay_fails_closed_on_invalid_verified_proof_metadata(tmp_path):
    source = _write_source(
        tmp_path / "window" / "eidetic-plus-full__run0.jsonl",
        [_row("invalid-proof", verified=True, abstained=False,
              correct=True, valid_proof=False)],
    )

    replayed, report = replay_sources([source])

    assert report["status"] == "FAIL"
    assert report["checks"]["no_invalid_verified_proof_metadata"] is False
    assert report["checks"]["zero_verified_correct_regression"] is False
    assert report["aggregate"]["invalid_verified_proof_rows"] == 1
    assert replayed[0]["status"] == "ABSTAINED"
    assert replayed[0]["extra"]["replay"]["proof_metadata_issues"]


def test_replay_artifact_is_byte_deterministic_and_hash_bound(tmp_path):
    source = _write_source(
        tmp_path / "window" / "eidetic-plus-full__run0.jsonl",
        [
            _row("verified", verified=True, abstained=False, correct=True),
            _row("unverified", verified=False, abstained=False, correct=True),
        ],
    )
    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"

    manifest_a = write_replay_artifact([source], out_a)
    manifest_b = write_replay_artifact([source], out_b)

    assert manifest_a == manifest_b
    implementation = Path(__file__).parent.parent / manifest_a["implementation"]["path"]
    assert hashlib.sha256(implementation.read_bytes()).hexdigest() == \
        manifest_a["implementation"]["sha256"]
    for name in ("replay_rows.jsonl", "replay_report.json",
                 "replay_report.md", "replay_manifest.json"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    for output in manifest_a["outputs"]:
        data = (out_a / output["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == output["sha256"]
    root = Path(__file__).parent.parent
    assert verify_replay_artifact(out_a, repo_root=root)["status"] == "PASS"
    with (out_a / "replay_rows.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    verification = verify_replay_artifact(out_a, repo_root=root)
    assert verification["status"] == "FAIL"
    assert any("output:replay_rows.jsonl:hash_mismatch" in failure
               for failure in verification["failures"])


def _guard_row(sample_id: str, *, question: str, predicted: str, correct: bool,
               policy: str) -> dict:
    row = _row(sample_id, verified=True, abstained=False, correct=correct)
    row["question"] = question
    row["predicted"] = predicted
    row["extra"]["policy"] = policy
    return row


def test_guard_projection_converts_junk_and_keeps_clean_rows(tmp_path):
    from bench.replay import project_guard_policy

    rows = [
        # Junk-echo verified-wrong structured row: today's clean-fact floor rejects it.
        _guard_row("junk-echo", question="What did Priya take part in at the fair?",
                   predicted="Yeah, Priya", correct=False,
                   policy="smqe:latest_value:claim"),
        # Dangling-tail verified-wrong structured row.
        _guard_row("dangling", question="What pastimes were shared with the club?",
                   predicted="last month, garden, painting,", correct=False,
                   policy="smqe:open_inference:claim"),
        # Clean verified-correct structured row: must survive untouched.
        _guard_row("clean-right", question="Where did the ceremony take place?",
                   predicted="at the Harbor pavilion", correct=True,
                   policy="smqe:latest_value:claim"),
        # Clean verified-correct reader row.
        _guard_row("reader-right", question="What instrument does Lena practice?",
                   predicted="Lena practices the cello every evening", correct=True,
                   policy="fixed-reader + verify+abstain+proof"),
        # Wrong-but-clean value: form cannot know it is wrong -- stays residual.
        _guard_row("clean-wrong", question="When did the workshop take place?",
                   predicted="March 2021", correct=False,
                   policy="smqe:relative_temporal:claim"),
    ]
    source = _write_source(tmp_path / "window" / "eidetic-plus-full__run0.jsonl", rows)

    report = project_guard_policy([source])

    assert report["status"] == "PASS"
    agg = report["aggregate"]
    assert agg["verified_rows"] == 5
    assert agg["verified_wrong_before"] == 3
    assert agg["wrong_converted_to_abstain"] == 2
    assert agg["correct_lost_to_abstain"] == 0
    assert agg["verified_wrong_after"] == 1
    assert report["residual_wrong_by_family"] == {"relative_temporal": 1}
    converted = {item["sample_id"] for item in report["conversions"]}
    assert converted == {"junk-echo", "dangling"}
    assert report["losses"] == []


def test_guard_projection_enumerates_every_correct_loss(tmp_path):
    from bench.replay import project_guard_policy

    rows = [
        # A correct row today's floor would reject: it MUST be enumerated, never silent.
        _guard_row("lost-right", question="What did she photograph at dawn?",
                   predicted="I took", correct=True,
                   policy="smqe:latest_value:claim"),
    ]
    source = _write_source(tmp_path / "window" / "eidetic-plus-full__run0.jsonl", rows)

    report = project_guard_policy([source])

    assert report["aggregate"]["correct_lost_to_abstain"] == 1
    assert [item["sample_id"] for item in report["losses"]] == ["lost-right"]
    assert report["checks"]["all_losses_enumerated"] is True


def test_guard_projection_artifact_binds_guard_implementation_bytes(tmp_path):
    from bench.replay import write_guard_projection_artifact

    source = _write_source(
        tmp_path / "window" / "eidetic-plus-full__run0.jsonl",
        [_guard_row("clean-right", question="Where did the ceremony take place?",
                    predicted="at the Harbor pavilion", correct=True,
                    policy="smqe:latest_value:claim")],
    )
    out = tmp_path / "out"

    manifest = write_guard_projection_artifact([source], out)

    root = Path(__file__).parent.parent
    bound = {impl["path"]: impl for impl in manifest["implementations"]}
    assert set(bound) == {"bench/replay.py", "eidetic/smqe/verify.py"}
    for rel, impl in bound.items():
        data = (root / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == impl["sha256"]
        assert len(data) == impl["bytes"]
    for output in manifest["outputs"]:
        data = (out / output["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == output["sha256"]


def test_replay_sorts_sources_and_rejects_duplicate_content(tmp_path):
    row = _row("verified", verified=True, abstained=False, correct=True)
    source_b = _write_source(tmp_path / "b" / "eidetic-plus-full__run0.jsonl", [row])
    source_a = _write_source(
        tmp_path / "a" / "eidetic-plus-full__run0.jsonl",
        [_row("other", verified=True, abstained=False, correct=True)],
    )

    _, report = replay_sources([source_b, source_a])
    assert [window["source_id"] for window in report["windows"]] == [
        "a/eidetic-plus-full__run0.jsonl",
        "b/eidetic-plus-full__run0.jsonl",
    ]

    duplicate = _write_source(tmp_path / "c" / "eidetic-plus-full__run0.jsonl", [row])
    with pytest.raises(ValueError, match="duplicate source content"):
        replay_sources([source_b, duplicate])
