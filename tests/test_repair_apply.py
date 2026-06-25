"""Offline tests for memory autopsy + guarded repair apply (Connected Brain Loop, Phase 5).

The apply path is additive-immutable by construction: dry-run never touches the store, and with
the flag off even apply=True is a dry-run, so no test here ever needs a model call or mutates raw.
"""
from __future__ import annotations

from dataclasses import replace

from eidetic.models import BrainEventType, MemoryRecord, Scope, now


# ---- memory autopsy (read-only failure classifier) -----------------------------------------
def test_autopsy_missing_write(engine):
    out = engine.memory_autopsy("what is the zebra protocol", scope=Scope(namespace="empty"))
    assert out["diagnosis"] == "missing_write"
    assert out["matching_memories"] == 0


def test_autopsy_pending_consolidation(engine):
    scope = Scope(namespace="ns")
    engine.store.upsert_record(MemoryRecord(
        memory_id="m1", content_hash="h1", text="the zebra protocol is documented here",
        scope=scope, valid_at=now(), metadata={"pending_consolidation": True}))
    out = engine.memory_autopsy("what is the zebra protocol", scope=scope)
    assert out["diagnosis"] == "pending_consolidation_not_run"


def test_autopsy_entity_extraction_failure(engine):
    scope = Scope(namespace="ns2")
    engine.store.upsert_record(MemoryRecord(
        memory_id="m1", content_hash="h1", text="the zebra protocol matters", scope=scope,
        valid_at=now(), entities=[], metadata={}))
    out = engine.memory_autopsy("what is the zebra protocol", scope=scope)
    assert out["diagnosis"] == "entity_extraction_failure"


# ---- guarded repair apply ------------------------------------------------------------------
def test_apply_proposals_dry_run_does_not_mutate(engine):
    scope = Scope(namespace="rep")
    before = engine.store.count(scope)
    props = [{"target_id": "m1", "action": "insert", "answer": "corrected fact",
              "diagnosis": "missing"}]
    rep = engine.apply_repair_proposals(props, scope=scope, apply=False)
    assert rep["mode"] == "dry-run" and rep["applied_count"] == 0
    assert len(rep["planned"]) == 1 and rep["planned"][0]["action"] == "insert"
    assert engine.store.count(scope) == before                 # immutable: nothing ingested


def test_apply_refused_when_flag_off_even_with_apply_true(engine):
    # apply=True but DREAM_REPAIR_APPLY off -> still a dry-run (no ingest, which would need a key).
    scope = Scope(namespace="rep")
    before = engine.store.count(scope)
    props = [{"target_id": "m1", "action": "merge", "answer": "x", "diagnosis": "contradicted"}]
    rep = engine.apply_repair_proposals(props, scope=scope, apply=True)
    assert rep["mode"] == "dry-run" and rep["applied_count"] == 0
    assert engine.store.count(scope) == before


def test_skip_proposals_are_dropped(engine):
    rep = engine.apply_repair_proposals(
        [{"action": "skip", "target_id": "m1", "answer": ""}], apply=False)
    assert rep["planned"] == []


def test_apply_executes_additive_ingest_and_never_mutates_target(fresh_settings, monkeypatch):
    # Positive coverage of the APPLY branch (flag on + apply=True). ingest_text is the only
    # model-bound call; stub it (a test double for the embed, like the MCP suite's FakeClient) so
    # the additive-immutable logic runs offline. Proves: a NEW record is created, the OLD target
    # is untouched, and REPAIR_APPLIED is emitted. The raw store is never mutated or deleted.
    from eidetic.engine import Engine
    e = Engine(replace(fresh_settings, dream_repair_apply_enabled=True, brain_events_enabled=True))
    scope = Scope(namespace="rep")
    e.store.upsert_record(MemoryRecord(memory_id="m1", content_hash="h1", text="old fact",
                                       scope=scope, valid_at=now()))
    before_count = e.store.count(scope)
    before_text = e.get_record("m1").text

    def fake_ingest_text(text, *, source="user", scope=None, consolidate_now=True, **kw):
        rec = MemoryRecord(memory_id="new1", content_hash="hn", text=text,
                           scope=scope or Scope(), valid_at=now())
        e.store.upsert_record(rec)
        return rec

    monkeypatch.setattr(e, "ingest_text", fake_ingest_text)
    props = [{"target_id": "m1", "action": "insert", "answer": "corrected fact",
              "diagnosis": "missing"}]
    rep = e.apply_repair_proposals(props, scope=scope, apply=True)

    assert rep["mode"] == "applied" and rep["applied_count"] == 1
    assert rep["applied"][0]["new_memory_id"] == "new1"
    assert e.get_record("m1").text == before_text          # target raw record untouched
    assert e.store.count(scope) == before_count + 1        # exactly one NEW additive record
    assert e.brain_log.by_type(BrainEventType.REPAIR_APPLIED)
