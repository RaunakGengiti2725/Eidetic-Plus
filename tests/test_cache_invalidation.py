"""Track 2: versioned answer-cache invalidation kills stale-truth cache hits.

The cache entry IDENTITY stays (scope_key, query) so agent A's answer never serves to agent B
(that would be a scope leak, not a hit-rate cost). The VERSION is per-namespace, so any content
write in the namespace makes every prior cached answer in that namespace unreachable -- including
a namespace-wide query's entry when the write was under a finer sub-scope."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.models import Scope
from eidetic.semantic_cache import SemanticCache


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


# ---- SemanticCache version semantics (unit) ------------------------------------------------
def test_cache_exact_hit_requires_matching_version():
    c = SemanticCache()
    c.put("sk", "q", None, "ANS", version=1)
    assert c.get("sk", "q", None, version=1) == "ANS"
    assert c.get("sk", "q", None, version=2) is None     # bumped version -> stale -> miss


def test_cache_cosine_hit_requires_matching_version():
    c = SemanticCache(cosine_threshold=0.5)
    qv = np.ones(8, np.float32)
    c.put("sk", "q", qv, "ANS", version=1)
    assert c.get("sk", "different but similar", qv, version=1) == "ANS"
    assert c.get("sk", "different but similar", qv, version=2) is None


def test_cache_version_zero_is_legacy_behavior():
    c = SemanticCache()
    c.put("sk", "q", None, "ANS")            # default version=0
    assert c.get("sk", "q", None) == "ANS"   # default version=0 -> hit (back-compat)


# ---- Engine namespace versioning -----------------------------------------------------------
def _engine(fresh_settings, **kw):
    s = replace(fresh_settings, rerank_enabled=False, **kw)
    return Engine(s, client=_FakeReader(s.embed_dim))


def test_ingest_bumps_namespace_version(fresh_settings):
    e = _engine(fresh_settings)
    v0 = e._ns_version("default")
    e.ingest_text("alpha fact", consolidate_now=False)
    assert e._ns_version("default") == v0 + 1


def test_subscope_write_bumps_the_namespace_version(fresh_settings):
    # the bug this track kills: a write under (ns, agent A) must invalidate a (ns) query's cache.
    e = _engine(fresh_settings)
    v0 = e._ns_version("proj")
    e.ingest_text("alpha fact", scope=Scope(namespace="proj", agent_id="A"), consolidate_now=False)
    assert e._ns_version("proj") == v0 + 1


def test_write_invalidates_a_cached_answer(fresh_settings):
    e = _engine(fresh_settings, cache_versioning_enabled=True, semantic_cache_enabled=True)
    e.ingest_text("Helios revenue was 4.2 million dollars", consolidate_now=False)
    sk = Scope().key()
    e.ask("what was Helios revenue")
    v1 = e._ns_version("default")
    assert e.cache.get(sk, "what was Helios revenue", None, version=v1) is not None  # cached
    e.ingest_text("Bob enjoys hiking on weekends", consolidate_now=False)            # bumps version
    v2 = e._ns_version("default")
    assert v2 == v1 + 1
    assert e.cache.get(sk, "what was Helios revenue", None, version=v2) is None       # invalidated


def test_versioning_off_keeps_legacy_cache(fresh_settings):
    e = _engine(fresh_settings, cache_versioning_enabled=False, semantic_cache_enabled=True)
    e.ingest_text("Helios revenue was 4.2 million dollars", consolidate_now=False)
    sk = Scope().key()
    e.ask("what was Helios revenue")
    e.ingest_text("Bob enjoys hiking on weekends", consolidate_now=False)
    # legacy: version always 0 -> the prior answer is still cached (the pre-fix behavior, unchanged).
    assert e.cache.get(sk, "what was Helios revenue", None, version=0) is not None


def test_reflex_flag_off_rebuilds_only_when_store_changed(fresh_settings, monkeypatch):
    e = _engine(fresh_settings, reflex_recall_enabled=False)
    from eidetic.models import MemoryRecord
    e.store.upsert_record(MemoryRecord(memory_id="m1", content_hash="h1", text="alpha keyword",
                                       scope=Scope(), valid_at=1.0))
    calls = {"n": 0}
    orig = e.reflex_index.rebuild_from_store

    def _counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(e.reflex_index, "rebuild_from_store", _counted)
    e.reflex_recall("alpha keyword")     # never built -> rebuild
    e.reflex_recall("alpha keyword")     # store unchanged -> NO rebuild
    assert calls["n"] == 1
    e.store.upsert_record(MemoryRecord(memory_id="m2", content_hash="h2", text="beta keyword",
                                       scope=Scope(), valid_at=1.0))
    assert "m2" in e.reflex_recall("beta keyword").candidate_ids()  # store changed -> rebuild
    assert calls["n"] == 2
