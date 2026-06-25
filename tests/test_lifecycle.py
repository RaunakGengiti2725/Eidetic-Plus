"""Offline tests for the LifecycleController + unified sleep (Connected Brain Loop, Phase 1)."""
from __future__ import annotations

from dataclasses import replace

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
