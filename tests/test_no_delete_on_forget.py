"""Required test: forgetting only down-weights the index; it never deletes a raw record.

Even after decades of FSRS decay (priority -> ~0), the record is still in the store
and its raw bytes are byte-for-byte intact in the immutable substrate. A strong cue
(reawakening) restores priority in O(1)."""
from __future__ import annotations

from eidetic import fsrs
from eidetic.models import MemoryRecord, now
from eidetic.store import RecordStore
from eidetic.substrate import ImmutableViolation, LocalCASSubstrate


def test_forgetting_downweights_but_never_deletes(tmp_path):
    sub = LocalCASSubstrate(tmp_path / "cas")
    store = RecordStore(tmp_path / "db.sqlite")

    raw = b"User strongly prefers window seats on long flights."
    h, uri = sub.put(raw)
    rec = MemoryRecord(
        content_hash=h, raw_uri=uri, text=raw.decode(),
        fsrs=fsrs.init_state(importance=0.9, surprise=0.9),
    )
    store.upsert_record(rec)

    t0 = rec.fsrs.last_review
    priority_now = fsrs.current_retrievability(rec.fsrs, t0)

    # Fast-forward 50 years: heavy forgetting.
    far = t0 + 50 * 365 * 86400
    fsrs.decay(rec.fsrs, at=far)
    store.upsert_record(rec)
    priority_old = fsrs.current_retrievability(rec.fsrs, far)

    # Forgetting happened at the index...
    assert priority_old < priority_now
    assert priority_old < 0.2

    # ...but the record still exists and the RAW bytes are untouched.
    persisted = store.get_record(rec.memory_id)
    assert persisted is not None
    assert sub.get(h) == raw
    assert sub.verify(h)

    # The substrate has no delete: forgetting could never have removed it.
    try:
        sub.delete(h)
        deleted = True
    except ImmutableViolation:
        deleted = False
    assert deleted is False
    assert sub.get(h) == raw


def test_reawakening_restores_priority(tmp_path):
    store = RecordStore(tmp_path / "db.sqlite")
    rec = MemoryRecord(text="reawaken me", fsrs=fsrs.init_state(0.5, 0.5))
    store.upsert_record(rec)

    far = rec.fsrs.last_review + 10 * 365 * 86400
    fsrs.decay(rec.fsrs, at=far)
    decayed = fsrs.current_retrievability(rec.fsrs, far)
    assert decayed < 0.5

    fsrs.reinforce(rec.fsrs, importance=0.9, at=far)
    assert abs(rec.fsrs.retrievability - 1.0) < 1e-6  # O(1) re-promotion
