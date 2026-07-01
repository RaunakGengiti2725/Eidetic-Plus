"""Snap-back fidelity: a faded memory snaps back to the byte-identical original, 100% over the
corpus (forgetting-machine plan -- a falsifiable NUMBER, not a UI demo).

Forgetting lowers only the FSRS index-priority weight; the content-addressed substrate is never
mutated or deleted. engine.snap_back_audit() verifies sha256(get_raw(h)) == h for every record and
reports lossless/total so the bench can quote the guarantee. These tests run fully offline (records
are injected straight into the store + substrate -- no model call)."""
from __future__ import annotations

from eidetic import fsrs
from eidetic.models import MemoryRecord, now
from scripts.snap_back_audit import build_report


def _ingest_offline(engine, raws: list[bytes]) -> None:
    for raw in raws:
        h, uri = engine.substrate.put(raw)
        rec = MemoryRecord(content_hash=h, raw_uri=uri, text=raw.decode(),
                           fsrs=fsrs.init_state(importance=0.5, surprise=0.5))
        engine.store.upsert_record(rec)


def test_snap_back_audit_is_lossless_after_a_century_of_fade(engine):
    raws = [f"Memory {i}: a user fact, value={i * 7}, tag=alpha{i}".encode() for i in range(25)]
    _ingest_offline(engine, raws)

    # Fade EVERY record 100 years forward -- retrievability collapses toward 0.
    far = now() + 100 * 365 * 86400
    faded = 0
    for rec in engine.store.all_records():
        fsrs.decay(rec.fsrs, at=far)
        engine.store.upsert_record(rec)
        if fsrs.current_retrievability(rec.fsrs, far) < 0.2:
            faded += 1
    assert faded == len(raws), "precondition: every record must actually be heavily faded"

    audit = engine.snap_back_audit()
    assert audit["total"] == len(raws)
    assert audit["lossless"] == len(raws)
    assert audit["rate"] == 1.0
    assert set(audit["audited_content_hashes"]) == {r.content_hash for r in engine.store.all_records()}
    assert audit["failures"] == []


def test_snap_back_audit_returns_byte_identical_bytes(engine):
    raws = [b"window seats on long flights", b"deathly allergic to peanuts"]
    _ingest_offline(engine, raws)
    far = now() + 30 * 365 * 86400
    for rec in engine.store.all_records():
        fsrs.decay(rec.fsrs, at=far)
        engine.store.upsert_record(rec)
    # The faded record still returns the EXACT original bytes via get_raw.
    for rec in engine.store.all_records():
        assert engine.get_raw(rec.content_hash) in raws


def test_snap_back_audit_flags_corruption_never_hides_it(engine):
    _ingest_offline(engine, [b"a real, intact memory"])
    # Inject a record whose raw blob is absent from the substrate (simulated corruption / loss).
    ghost = MemoryRecord(content_hash="de" * 32, raw_uri="cas://de", text="ghost")
    engine.store.upsert_record(ghost)

    audit = engine.snap_back_audit()
    assert audit["total"] == 2
    assert audit["lossless"] == 1
    assert audit["rate"] < 1.0
    assert len(audit["audited_content_hashes"]) == 1
    assert len(audit["failures"]) == 1
    assert audit["failures"][0]["content_hash"] == "de" * 32


def test_records_without_a_raw_blob_are_skipped(engine):
    _ingest_offline(engine, [b"intact memory with a raw blob"])
    # Derived/profile-style record: no content_hash -> nothing to snap back, excluded from total.
    derived = MemoryRecord(text="derived gist, no raw bytes", content_hash="")
    engine.store.upsert_record(derived)

    audit = engine.snap_back_audit()
    assert audit["total"] == 1
    assert audit["rate"] == 1.0


def test_snap_back_script_report_fails_empty_or_corrupt(tmp_path):
    empty = build_report({"total": 0, "lossless": 0, "rate": 1.0, "failures": []}, tmp_path)
    assert empty["status"] == "FAIL"
    assert empty["records_with_raw_blob"] == 0

    corrupt = build_report(
        {"total": 2, "lossless": 1, "rate": 0.5,
         "audited_content_hashes": ["ab" * 32],
         "failures": [{"memory_id": "m1", "content_hash": "bad", "error": "missing"}]},
        tmp_path,
    )
    assert corrupt["status"] == "FAIL"
    assert corrupt["rate_pct"] == 50.0
    assert corrupt["audited_content_hashes"] == ["ab" * 32]
