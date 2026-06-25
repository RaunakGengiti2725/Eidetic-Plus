"""Offline tests for Phase 0 correctness patches: scope-isolated brain telemetry."""
from __future__ import annotations

from dataclasses import replace

from eidetic.engine import Engine
from eidetic.models import (Answer, BrainEventType, Citation, NLILabel, RecallTrace, Scope)


def test_channel_wins_are_isolated_by_namespace(engine):
    def record(ns, ch):
        engine.retriever.last_trace = RecallTrace(query="q", scope=Scope(namespace=ns),
                                                  channel_results={ch: ["m1"]})
        ans = Answer(question="q", answer="a", citations=[
            Citation(memory_id="m1", content_hash="h", raw_uri="", source="u", valid_at=1.0,
                     nli_label=NLILabel.ENTAILMENT)])
        engine.record_channel_wins(ans)

    record("A", "dense"); record("A", "dense"); record("B", "event")
    assert engine.channel_win_stats(Scope(namespace="A")) == {"dense": 2}
    assert engine.channel_win_stats(Scope(namespace="B")) == {"event": 1}
    assert engine.channel_win_stats() == {"dense": 2, "event": 1}     # merged across namespaces


def test_brain_event_counts_and_health_are_scope_isolated(fresh_settings):
    e = Engine(replace(fresh_settings, brain_events_enabled=True))
    e._brain(BrainEventType.ANSWER_VERIFIED, namespace="A")
    e._brain(BrainEventType.ANSWER_VERIFIED, namespace="A")
    e._brain(BrainEventType.RETRIEVAL_MISSED, namespace="B")

    assert e.connection_effectiveness(Scope(namespace="A"))["events"] == {"answer_verified": 2}
    assert e.connection_effectiveness(Scope(namespace="B"))["events"] == {"retrieval_missed": 1}
    assert e.connection_effectiveness()["total_events"] == 3          # merged

    # brain_health_score(scope=A) must not see B's miss (which would inflate unsupported rate).
    hb = e.brain_health_score(Scope(namespace="A"))
    assert hb["events"] == {"answer_verified": 2}
    assert hb["components"]["unsupported_answer_rate"] == 0.0         # A had no misses
