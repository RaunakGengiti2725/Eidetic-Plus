"""Offline tests for S1 read-path latency: batched NLI, short-circuit verify, deferred re-embed."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.config import get_settings
from eidetic.dashscope_client import DashScopeClient
from eidetic.graph import KnowledgeGraph
from eidetic.models import (BrainEventType, MemoryRecord, NLILabel, RetrievalCandidate, Scope)
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore


class _FakeEmbed:
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


class _FakeSub:
    def get(self, h):
        raise KeyError(h)            # -> _ground_truth falls back to rec.text


def _cands(n):
    return [RetrievalCandidate(record=MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}",
            text=f"fact {i}", scope=Scope(), valid_at=1.0)) for i in range(n)]


# ---- batched NLI ---------------------------------------------------------------------------
def test_nli_batch_maps_by_index_and_fills_missing_as_neutral(monkeypatch):
    c = DashScopeClient(get_settings())
    monkeypatch.setattr(c, "chat_json", lambda *a, **k: {"results": [
        {"index": 0, "label": "entailment", "confidence": 0.9},
        {"index": 2, "label": "contradiction", "confidence": 0.8},   # index 1 omitted -> neutral
    ]})
    out = c.nli_batch([("p0", "h"), ("p1", "h"), ("p2", "h")])
    assert out == [("entailment", 0.9), ("neutral", 0.0), ("contradiction", 0.8)]


def test_batch_nli_verify_path_matches_labels(fresh_settings):
    s = replace(fresh_settings, batch_nli_enabled=True)
    store = RecordStore(s.sqlite_path)

    class C:
        def nli_batch(self, pairs):
            return [("entailment", 0.9)] * len(pairs)

    r = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), C(), s)
    cits, entailed = r._verify_candidates(_cands(3), "ans", verify=True)
    assert entailed == 3 and all(c.nli_label == NLILabel.ENTAILMENT for c in cits)


# ---- short-circuit verify ------------------------------------------------------------------
def test_fast_verify_short_circuits_after_cap(fresh_settings, monkeypatch):
    s = replace(fresh_settings, fast_verify_enabled=True, verify_citation_cap=2)
    store = RecordStore(s.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), object(), s)
    calls = {"n": 0}

    def fake_verify(rec, text):
        calls["n"] += 1
        return NLILabel.ENTAILMENT, 0.9

    monkeypatch.setattr(r, "verify_citation", fake_verify)
    cits, entailed = r._verify_candidates(_cands(5), "ans", verify=True)
    assert calls["n"] == 2 and entailed == 2          # stopped after the cap
    assert len(cits) == 5                             # the rest still get a (neutral) citation


def test_baseline_verify_unchanged_when_flags_off(fresh_settings, monkeypatch):
    store = RecordStore(fresh_settings.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), object(), fresh_settings)
    monkeypatch.setattr(r, "verify_citation", lambda rec, text: (NLILabel.ENTAILMENT, 0.7))
    cits, entailed = r._verify_candidates(_cands(4), "ans", verify=True)
    assert entailed == 4 and len(cits) == 4          # every candidate verified serially (baseline)


# ---- deferred re-embed ---------------------------------------------------------------------
def test_deferred_reembed_enqueue_and_drain(fresh_settings):
    from eidetic.engine import Engine
    e = Engine(replace(fresh_settings, defer_reembed_enabled=True, brain_events_enabled=True),
               client=_FakeEmbed(fresh_settings.embed_dim))
    scope = Scope(namespace="d")
    rec = e.ingest_text("alice likes green tea", scope=scope, consolidate_now=False)

    e._enqueue_reembed([rec.memory_id])
    assert rec.memory_id in e._reembed_queue
    out = e.drain_reembed_queue()
    assert out["reembedded"] == 1
    assert rec.memory_id not in e._reembed_queue          # queue drained
    assert e.brain_log.by_type(BrainEventType.REEMBED_DEFERRED)
    # a second drain is a clean no-op.
    assert e.drain_reembed_queue()["reembedded"] == 0
