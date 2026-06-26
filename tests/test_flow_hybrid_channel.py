"""Track 9 Task 5: the hybrid retrieve activation channel. ~half of recall volume can hit hybrid
(reflex miss / REFLEX_RECALL=0 / bench). The activation channel + field-seeded coactivation give
those paths instinct too. activation=None is byte-identical to today; an activation-seeded id is
gated to the in-scope active corpus and carries dense_score 0 (coverage-safe)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore
from eidetic.vector_index import make_vector_index


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


def _retriever(settings):
    store = RecordStore(settings.sqlite_path)
    index = make_vector_index(settings)
    client = _Embed(settings.embed_dim)
    r = Retriever(store, index, KnowledgeGraph(store), _FakeSub(), client, settings)
    return store, index, r, client


def _add(store, index, client, mid, text, *, ns="default", valid_at=1.0):
    rec = MemoryRecord(memory_id=mid, content_hash=f"h_{mid}", raw_uri=f"cas://h_{mid}",
                       text=text, scope=Scope(namespace=ns), valid_at=valid_at)
    store.upsert_record(rec)
    index.add(mid, client.embed_text(text), None)
    return rec


def test_activation_channel_lifts_fused_score_coverage_safe(fresh_settings):
    s = replace(fresh_settings, rerank_enabled=False, flow_hybrid_channel_enabled=True,
                flow_hybrid_weight=0.35)
    store, index, r, client = _retriever(s)
    _add(store, index, client, "m_named", "alpha keyword target")
    _add(store, index, client, "m_quiet", "completely different beta wording")
    base = {c.record.memory_id: c.fused_score for c in r.retrieve("alpha keyword target", scope=Scope())}
    cands = r.retrieve("alpha keyword target", scope=Scope(), activation={"m_quiet": 1.0})
    act = {c.record.memory_id: c.fused_score for c in cands}
    assert "m_quiet" in act
    assert act["m_quiet"] > base.get("m_quiet", 0.0)         # the channel lifted its fused score
    quiet = next(c for c in cands if c.record.memory_id == "m_quiet")
    assert quiet.dense_score == 0.0                           # content-only score untouched (coverage-safe)


def test_activation_none_is_byte_identical(fresh_settings):
    s = replace(fresh_settings, rerank_enabled=False, flow_hybrid_channel_enabled=True)
    store, index, r, client = _retriever(s)
    _add(store, index, client, "a", "alpha keyword")
    _add(store, index, client, "b", "beta wording")
    base = [c.record.memory_id for c in r.retrieve("alpha keyword", scope=Scope())]
    none = [c.record.memory_id for c in r.retrieve("alpha keyword", scope=Scope(), activation=None)]
    assert base == none                                       # no activation -> unchanged ranking


def test_activation_channel_off_is_inert(fresh_settings):
    # channel off AND coactivation off -> activation has no effect on ranking at all.
    s = replace(fresh_settings, rerank_enabled=False, flow_hybrid_channel_enabled=False,
                coactivation_channel_enabled=False)
    store, index, r, client = _retriever(s)
    _add(store, index, client, "m_named", "alpha keyword")
    _add(store, index, client, "m_quiet", "beta wording")
    base = [(c.record.memory_id, c.fused_score) for c in r.retrieve("alpha keyword", scope=Scope())]
    withact = [(c.record.memory_id, c.fused_score)
               for c in r.retrieve("alpha keyword", scope=Scope(), activation={"m_quiet": 1.0})]
    assert base == withact                                    # channel off -> activation inert


def test_activation_channel_respects_scope(fresh_settings):
    s = replace(fresh_settings, rerank_enabled=False, flow_hybrid_channel_enabled=True)
    store, index, r, client = _retriever(s)
    _add(store, index, client, "here", "alpha keyword", ns="A")
    _add(store, index, client, "there", "beta wording", ns="B")
    ids = [c.record.memory_id for c in r.retrieve("alpha keyword", scope=Scope(namespace="A"),
                                                  activation={"there": 1.0})]
    assert "there" not in ids                                 # cross-namespace activated id dropped
