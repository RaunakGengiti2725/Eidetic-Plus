"""Offline tests for the competitive moat: C2 deterministic supersession + time-travel, C1
operation-level integrity report."""
from __future__ import annotations

from dataclasses import replace

from eidetic.engine import Engine
from eidetic.models import BrainEventType, Scope


# ---- C2: deterministic value-as-of + current-vs-historical chain ---------------------------
def test_value_as_of_time_travel(fresh_settings):
    e = Engine(fresh_settings)
    scope = Scope(namespace="kg")
    e.graph.add_fact("Alice", "works_at", "Acme", valid_at=100.0, scope=scope)
    e.graph.add_fact("Alice", "works_at", "Beta", valid_at=200.0, scope=scope)
    e.graph.add_fact("Alice", "works_at", "Gamma", valid_at=300.0, scope=scope)

    assert e.value_as_of("Alice", "works_at", as_of=250.0, scope=scope)["value"] == "Beta"
    assert e.value_as_of("Alice", "works_at", scope=scope)["value"] == "Gamma"     # current
    assert e.value_as_of("Alice", "works_at", as_of=50.0, scope=scope) is None      # before any fact


def test_fact_history_shows_superseded_chain_retained(fresh_settings):
    e = Engine(fresh_settings)
    scope = Scope(namespace="kg")
    e.graph.add_fact("Alice", "works_at", "Acme", valid_at=100.0, scope=scope)
    e.graph.add_fact("Alice", "works_at", "Beta", valid_at=200.0, scope=scope)
    e.graph.add_fact("Alice", "works_at", "Gamma", valid_at=300.0, scope=scope)

    hist = e.fact_history("Alice", "works_at", scope=scope)
    assert [h["value"] for h in hist] == ["Acme", "Beta", "Gamma"]   # oldest first, all retained
    assert [h["current"] for h in hist] == [False, False, True]      # only the latest is current
    assert hist[0]["invalid_at"] == 200.0                            # Acme closed when Beta arrived


# ---- C1: operation-level integrity report --------------------------------------------------
def test_integrity_report_rates(fresh_settings):
    e = Engine(replace(fresh_settings, brain_events_enabled=True))
    scope = Scope(namespace="ir")
    e._brain(BrainEventType.ANSWER_VERIFIED, namespace="ir")
    e._brain(BrainEventType.ANSWER_VERIFIED, namespace="ir")
    e._brain(BrainEventType.ANSWER_ABSTAINED, namespace="ir")
    e._brain(BrainEventType.RETRIEVAL_MISSED, namespace="ir")

    rep = e.integrity_report(scope)
    assert rep["answered"] == 4
    assert rep["verified_rate"] == 0.5
    assert rep["abstention_rate"] == 0.25
    assert rep["fabrication_rate"] == 0.25         # answered-but-ungrounded surface
    assert e.brain_log.by_type(BrainEventType.INTEGRITY_CHECKED)    # emits its own event
