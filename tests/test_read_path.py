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


# ---- reader ordering guidance ----------------------------------------------------------------
def test_reader_prompt_anchors_event_order_questions(monkeypatch):
    """Ordering questions get an explicit date-anchoring instruction: an answer that merely
    echoes the question's event phrases is unprovable; dates make the ordering checkable."""
    c = DashScopeClient(replace(get_settings(), reader_cot_enabled=True))
    seen = {}

    def fake_chat_json(model, system, user, **kw):
        seen["system"] = system
        return {"answer": "ok"}

    monkeypatch.setattr(c, "chat_json", fake_chat_json)
    c.generate_answer(
        "Which three events happened in the order from first to last: the fair, the recital, "
        "and the workshop?",
        ["[2024-02-05] the fair", "[2024-03-01] the recital"],
    )
    assert "chronological" in seen["system"].lower()
    assert "date" in seen["system"].lower()

    c.generate_answer("What color is my bike?", ["the bike is red"])
    assert "chronological" not in seen["system"].lower()


# ---- answer-path index save gating -----------------------------------------------------------
def test_ask_saves_index_only_when_reconsolidation_mutated_it(fresh_settings, monkeypatch):
    """The per-answer index.save() is an O(corpus) disk write under the write lock. It must run
    only when reconsolidation actually updated a vector this ask: with DEFER_REEMBED the update
    is queued for the idle drain, so saving an unchanged index buys nothing and serializes
    concurrent asks behind disk IO that grows with corpus size."""
    from dataclasses import replace as _replace

    from eidetic.engine import Engine
    from eidetic.models import Scope

    class _Client(_FakeEmbed):
        def extract_edges(self, text):
            return []

        def generate_answer(self, q, blocks, model=None):
            return blocks[0][:200] if blocks else "I do not have that in memory."

        def nli(self, premise, hypothesis):
            return ("entailment", 0.9)

    # deferred mode: the ask must NOT save (no index mutation happened on the answer path)
    s = _replace(fresh_settings, defer_reembed_enabled=True, rerank_enabled=False)
    e = Engine(s, client=_Client(s.embed_dim))
    scope = Scope(namespace="savegate")
    e.ingest_text("The greenhouse fan runs on circuit twelve.", scope=scope, consolidate_now=False)
    saves = {"n": 0}
    real_save = e.index.save
    monkeypatch.setattr(e.index, "save", lambda: (saves.__setitem__("n", saves["n"] + 1),
                                                  real_save())[1])
    ans = e.ask("which circuit runs the greenhouse fan", scope=scope)
    assert ans.citations
    assert saves["n"] == 0

    # inline mode: the re-embed mutates the index -> the save must still happen
    s2 = _replace(fresh_settings, defer_reembed_enabled=False, rerank_enabled=False)
    e2 = Engine(s2, client=_Client(s2.embed_dim))
    scope2 = Scope(namespace="savegate2")
    e2.ingest_text("The orchard pump runs on circuit nine.", scope=scope2, consolidate_now=False)
    saves2 = {"n": 0}
    real_save2 = e2.index.save
    monkeypatch.setattr(e2.index, "save", lambda: (saves2.__setitem__("n", saves2["n"] + 1),
                                                   real_save2())[1])
    ans2 = e2.ask("which circuit runs the orchard pump", scope=scope2)
    assert ans2.citations
    assert saves2["n"] >= 1
