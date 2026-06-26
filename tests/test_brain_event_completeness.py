"""Track 2.3: close the BrainEvent emission gaps -- RATE_LIMITED (governor 429/backoff) and
CACHE_HIT (a versioned, truth-fresh answer-cache hit)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.dashscope_client import ModelCallError, RateGovernor
from eidetic.engine import Engine
from eidetic.models import BrainEventType


class _FakeReader:
    def __init__(self, dim):
        self.dim = dim

    def _e(self, t):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, t):
        return self._e(t)

    def embed_texts(self, ts):
        return np.stack([self._e(t) for t in ts]) if ts else np.zeros((0, self.dim), np.float32)

    def generate_answer(self, q, blocks, model=None):
        return "Helios revenue was 4.2 million dollars"

    def nli(self, premise, hypothesis):
        return ("entailment", 0.9) if "helios" in (premise or "").lower() else ("neutral", 0.2)


# ---- RATE_LIMITED --------------------------------------------------------------------------
def test_governor_fires_on_rate_limit_callback_per_429():
    calls = []
    g = RateGovernor(rpm=600, max_concurrency=2, max_retries=3, backoff_base=0.0, backoff_max=0.0)
    g.on_rate_limit = lambda info: calls.append(info)
    n = {"i": 0}

    def fn():
        n["i"] += 1
        if n["i"] < 3:
            raise ModelCallError("DashScope call failed (HTTP 429): rate limit")
        return "ok"

    assert g.run(fn) == "ok"
    assert len(calls) == 2                      # two 429s retried -> two callbacks


def test_governor_retries_transient_transport_error_then_succeeds():
    import ssl
    g = RateGovernor(rpm=600, max_concurrency=2, max_retries=3, backoff_base=0.0, backoff_max=0.0)
    n = {"i": 0}

    def flaky():
        n["i"] += 1
        if n["i"] < 3:
            raise ssl.SSLEOFError("[SSL: UNEXPECTED_EOF_WHILE_READING]")
        return "ok"

    assert g.run(flaky) == "ok"          # a TLS blip is retried, not fatal
    assert n["i"] == 3


def test_governor_fails_loud_on_non_transient_and_exhausted():
    g = RateGovernor(rpm=600, max_concurrency=2, max_retries=3, backoff_base=0.0, backoff_max=0.0)
    import pytest as _pytest
    with _pytest.raises(ValueError):     # a real code error is NOT retried
        g.run(lambda: (_ for _ in ()).throw(ValueError("real bug")))
    with _pytest.raises(ModelCallError):  # quota exhaustion is NOT retried (never succeeds)
        g.run(lambda: (_ for _ in ()).throw(ModelCallError("HTTP 403: free tier exhausted")))


def test_governor_does_not_fire_callback_on_non_rate_error():
    calls = []
    g = RateGovernor(rpm=600, max_concurrency=2, max_retries=2, backoff_base=0.0, backoff_max=0.0)
    g.on_rate_limit = lambda info: calls.append(info)

    def fn():
        raise ModelCallError("DashScope call failed (HTTP 500): internal error")

    try:
        g.run(fn)
    except ModelCallError:
        pass
    assert calls == []                          # non-retryable error -> no rate-limit signal


def test_engine_on_rate_limit_emits_brain_event(fresh_settings):
    e = Engine(replace(fresh_settings, brain_events_enabled=True), client=_FakeReader(fresh_settings.embed_dim))
    e._on_rate_limit({"attempt": 1, "sleep_s": 0.5})
    assert e.brain_log.by_type(BrainEventType.RATE_LIMITED)


def test_engine_wires_rate_limit_callback_into_governor(fresh_settings):
    from eidetic.dashscope_client import DashScopeClient
    client = DashScopeClient(replace(fresh_settings, dashscope_govern_enabled=True))
    e = Engine(replace(fresh_settings), client=client)
    assert e.client._governor is not None
    assert e.client._governor.on_rate_limit is not None


# ---- CACHE_HIT -----------------------------------------------------------------------------
def test_repeated_query_emits_cache_hit_when_versioned(fresh_settings):
    e = Engine(replace(fresh_settings, brain_events_enabled=True, semantic_cache_enabled=True,
                       cache_versioning_enabled=True, rerank_enabled=False,
                       recall_trace_enabled=False),
               client=_FakeReader(fresh_settings.embed_dim))
    e.ingest_text("Helios revenue was 4.2 million dollars", consolidate_now=False)
    e.ask("what was Helios revenue")            # miss -> compute + cache (truth-fresh)
    e.ask("what was Helios revenue")            # identical, no write between -> versioned cache hit
    assert e.brain_log.by_type(BrainEventType.CACHE_HIT)


def test_write_between_repeats_invalidates_so_no_cache_hit(fresh_settings):
    e = Engine(replace(fresh_settings, brain_events_enabled=True, semantic_cache_enabled=True,
                       cache_versioning_enabled=True, rerank_enabled=False,
                       recall_trace_enabled=False),
               client=_FakeReader(fresh_settings.embed_dim))
    e.ingest_text("Helios revenue was 4.2 million dollars", consolidate_now=False)
    e.ask("what was Helios revenue")
    e.ingest_text("Bob enjoys hiking on weekends", consolidate_now=False)   # bumps version
    e.ask("what was Helios revenue")            # version moved -> stale -> recompute, NOT a hit
    assert not e.brain_log.by_type(BrainEventType.CACHE_HIT)
