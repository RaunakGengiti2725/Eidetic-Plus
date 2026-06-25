"""Offline tests for verified-helpful reinforcement (Phase 4)."""
from __future__ import annotations

from dataclasses import replace

from eidetic.engine import Engine
from eidetic.models import MemoryRecord, Scope, now
from eidetic.salience import verified_helpful_signal


def _rec(valid_at=None, **kw):
    return MemoryRecord(memory_id=kw.pop("mid", "m1"), content_hash="h", text="x",
                        scope=Scope(), valid_at=valid_at if valid_at is not None else now(), **kw)


def test_signal_saturates_at_cap():
    assert verified_helpful_signal(0, 5) == 0.0
    assert verified_helpful_signal(5, 5) == 1.0
    assert verified_helpful_signal(100, 5) == 1.0          # bounded -> age-leakage guard


def test_count_increments_and_salience_untouched_when_affect_off(engine):
    rec = _rec(salience=0.42)
    engine._reinforce_verified_helpful(rec)
    assert rec.verified_helpful_count == 1
    assert rec.salience == 0.42                            # affect off -> ranking-side untouched


def test_verified_helpful_raises_salience_when_affect_on(fresh_settings):
    e = Engine(replace(fresh_settings, affect_salience_enabled=True, affect_w_helpful=1.0,
                       verified_helpful_cap=5))
    many = _rec(importance=0.5, surprise=0.5, metadata={"arousal": 0.5, "emphasis": 0.5})
    for _ in range(3):
        e._reinforce_verified_helpful(many)
    once = _rec(mid="m2", importance=0.5, surprise=0.5, metadata={"arousal": 0.5, "emphasis": 0.5})
    e._reinforce_verified_helpful(once)
    assert many.verified_helpful_count == 3 and once.verified_helpful_count == 1
    assert many.salience > once.salience                   # more verified-helpful -> more salient


def test_verified_helpful_boost_is_age_invariant(fresh_settings):
    e = Engine(replace(fresh_settings, affect_salience_enabled=True, affect_w_helpful=1.0))

    def salience_after(valid_at):
        rec = _rec(valid_at=valid_at, importance=0.5, surprise=0.5,
                   metadata={"arousal": 0.5, "emphasis": 0.5})
        for _ in range(3):
            e._reinforce_verified_helpful(rec)
        return rec.salience

    assert salience_after(1.0) == salience_after(1_000_000_000.0)   # same count -> same salience


def test_multi_citation_attribution_credits_each_independently(engine):
    recs = [_rec(mid=f"m{i}") for i in range(3)]
    for r in recs:
        engine._reinforce_verified_helpful(r)
    assert all(r.verified_helpful_count == 1 for r in recs)
