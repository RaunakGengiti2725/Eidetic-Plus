"""Offline tests for the LifecycleController + unified sleep (Connected Brain Loop, Phase 1)."""
from __future__ import annotations

from dataclasses import replace
import threading

import numpy as np

from eidetic.engine import Engine
from eidetic.models import (Answer, Citation, NLILabel, RecallTrace, Scope)


def test_sleep_composite_is_offline_on_quiet_scope(engine):
    # No pending records + token-free dream -> the composite runs with no model call.
    out = engine.sleep(scope=Scope(namespace="quiet"))
    assert out["consolidate_pending"]["pending_processed"] == 0
    assert "dream" in out and "replay" in out["dream"]
    assert "consolidate" not in out                      # LLM summaries off by default


def test_idle_tick_returns_report_with_effectiveness(engine):
    rep = engine.idle_tick()
    assert "fusion_weights" in rep                        # learn over an empty buffer -> {}
    assert "connection_effectiveness" in rep


def test_after_recall_is_identity_passthrough(engine):
    ans = Answer(question="q", answer="a")
    assert engine.lifecycle.after_recall(ans) is ans     # never alters the answer
    assert engine.lifecycle.after_ingest is not None


def test_repair_tick_noop_when_disabled(engine):
    assert engine.lifecycle.repair_tick(Scope()) == {"skipped": "disabled"}


def test_record_channel_wins_from_matching_trace(engine):
    engine.retriever.last_trace = RecallTrace(
        query="q", channel_results={"dense": ["m1"], "event": ["m1"], "bm25": ["m9"]})
    ans = Answer(question="q", answer="a", citations=[
        Citation(memory_id="m1", content_hash="h", raw_uri="", source="u", valid_at=1.0,
                 nli_label=NLILabel.ENTAILMENT)])
    wins = engine.record_channel_wins(ans)
    assert wins["dense"] == 1 and wins["event"] == 1     # both surfaced the confirmed source
    assert "bm25" not in wins                            # bm25 did not surface m1
    assert engine.connection_effectiveness()["channel_wins"]["event"] == 1


def test_record_channel_wins_ignores_mismatched_trace(engine):
    engine.retriever.last_trace = RecallTrace(query="other", channel_results={"dense": ["m1"]})
    ans = Answer(question="q", answer="a", citations=[
        Citation(memory_id="m1", content_hash="h", raw_uri="", source="u", valid_at=1.0,
                 nli_label=NLILabel.ENTAILMENT)])
    assert engine.record_channel_wins(ans) == {}         # stale trace -> no wins credited


def test_auto_sleep_off_in_neutral_baseline(engine):
    scope = Scope(namespace="auto-off")
    report = engine.lifecycle.maybe_auto_sleep(scope)
    assert report["enabled"] is False and report["scheduled"] is False
    status = engine.auto_sleep_status(scope)
    assert status["enabled"] is False and status["pending_consolidation"] == 0


def test_after_ingest_schedules_background_auto_sleep_when_enabled(engine, monkeypatch):
    from eidetic.models import MemoryRecord, now

    scope = Scope(namespace="auto-on")
    engine.settings = replace(
        engine.settings,
        host_auto_sleep_enabled=True,
        host_auto_sleep_min_interval_sec=0.0,
        host_auto_sleep_score_importance=False,
    )
    done = threading.Event()

    def fake_consolidate_pending(*, scope=None, score_importance=True, max_workers=8):
        assert scope.namespace == "auto-on"
        assert score_importance is False
        return {"pending_processed": 1, "facts_extracted": 1, "events_indexed": 1}

    def fake_dream(*, scope=None):
        done.set()
        return {"replay": {"selected": 0}}

    monkeypatch.setattr(engine, "consolidate_pending", fake_consolidate_pending)
    monkeypatch.setattr(engine, "dream", fake_dream)
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text="Alice works at Acme",
                       scope=scope, valid_at=now(),
                       metadata={"pending_consolidation": True})

    assert engine.lifecycle.after_ingest(rec, scope) is rec
    assert done.wait(1.0), "auto-sleep background drain did not run"
    status = engine.auto_sleep_status(scope)
    assert status["enabled"] is True and status["running"] is False
    assert status["last_report"]["consolidate_pending"]["pending_processed"] == 1


class _EmbedOnly:
    def __init__(self, dim: int):
        self.dim = dim

    def embed_text(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        v[0] = 1.0
        return v


def test_engine_ingest_calls_lifecycle_after_ingest_for_fast_writes(fresh_settings, monkeypatch):
    eng = Engine(fresh_settings, client=_EmbedOnly(fresh_settings.embed_dim))
    scope = Scope(namespace="hook")
    called = []

    def fake_after_ingest(record, scope=None):
        called.append((record.memory_id, scope.namespace, record.metadata["pending_consolidation"]))
        return record

    monkeypatch.setattr(eng.lifecycle, "after_ingest", fake_after_ingest)
    rec = eng.ingest_text("remember this fast host fact", scope=scope, consolidate_now=False)

    assert called == [(rec.memory_id, "hook", True)]
