"""Track 9 Task 6: activation is ACCESS-recency, never memory age. Equal-content records of very
different valid_at score identically with no activation (no age bias); activating the OLDER one
lifts it (proving access, not age) on BOTH the reflex and hybrid paths; dense_score is unchanged."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.brain import QualityGate
from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.reflex_activation import build_memory_packet
from eidetic.reflex_index import ReflexIndex
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore
from eidetic.vector_index import make_vector_index

_YOUNG = 1_700_000_000.0
_OLD = 1_000_000.0
_TEXT = "project alpha milestone details"


class _Embed:
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
        raise KeyError(h)


def _rec(mid, valid_at):
    return MemoryRecord(memory_id=mid, content_hash=f"h_{mid}", raw_uri=f"cas://h_{mid}",
                        text=_TEXT, scope=Scope(), valid_at=valid_at)


def test_reflex_no_age_bias_without_activation(fresh_settings):
    store = RecordStore(fresh_settings.sqlite_path)
    store.upsert_record(_rec("young", _YOUNG))
    store.upsert_record(_rec("old", _OLD))
    idx = ReflexIndex()
    idx.rebuild_from_store(store)
    p = build_memory_packet(_TEXT, Scope(), store=store, graph=KnowledgeGraph(store), index=idx,
                            settings=fresh_settings)
    gap = abs(p.scores["young"].aggregate - p.scores["old"].aggregate)
    assert QualityGate.no_age_bias(gap, 0.0)                  # identical -> no age bias


def test_reflex_activation_lifts_the_older_memory(fresh_settings):
    s = replace(fresh_settings, reflex_w_activation=0.4)
    store = RecordStore(s.sqlite_path)
    store.upsert_record(_rec("young", _YOUNG))
    store.upsert_record(_rec("old", _OLD))
    idx = ReflexIndex()
    idx.rebuild_from_store(store)
    p = build_memory_packet(_TEXT, Scope(), store=store, graph=KnowledgeGraph(store), index=idx,
                            settings=s, activation={"old": 1.0})
    assert p.candidate_ids()[0] == "old"                     # access (activation), not age, decides


def test_hybrid_activation_lifts_older_fused_dense_unchanged(fresh_settings):
    s = replace(fresh_settings, rerank_enabled=False, flow_hybrid_channel_enabled=True,
                flow_hybrid_weight=0.5)
    store = RecordStore(s.sqlite_path)
    index = make_vector_index(s)
    client = _Embed(s.embed_dim)
    # distinct trailing token so the near-duplicate dedup keeps BOTH; same query-matching content.
    texts = {"young": "project alpha milestone fresh", "old": "project alpha milestone classic"}
    for mid, va in (("young", _YOUNG), ("old", _OLD)):
        rec = MemoryRecord(memory_id=mid, content_hash=f"h_{mid}", raw_uri=f"cas://h_{mid}",
                           text=texts[mid], scope=Scope(), valid_at=va)
        store.upsert_record(rec)
        index.add(mid, client.embed_text(texts[mid]), None)
    r = Retriever(store, index, KnowledgeGraph(store), _FakeSub(), client, s)
    base = {c.record.memory_id: c for c in r.retrieve("project alpha milestone", scope=Scope())}
    act = {c.record.memory_id: c for c in r.retrieve("project alpha milestone", scope=Scope(),
                                                     activation={"old": 1.0})}
    assert act["old"].fused_score > base["old"].fused_score   # activation lifted the OLDER one
    assert act["old"].dense_score == base["old"].dense_score  # content score untouched (access, not age)
