"""Track 9 Task 8: activation surfaces facts in CONTEXT BLOCKS, not just ranking. The scratchpad
ranks by salience + flow_context_weight*activation, so an activated high-salience fact appears even
when salience alone would not top-k it. activation=None/weight=0 is byte-identical to today."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.retrieval import Retriever
from eidetic.scratchpad import select_scratchpad
from eidetic.store import RecordStore
from eidetic.vector_index import make_vector_index


class _Embed:
    def __init__(self, dim):
        self.dim = dim

    def embed_text(self, t):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_texts(self, ts):
        return np.stack([self.embed_text(t) for t in ts]) if ts else np.zeros((0, self.dim), np.float32)


class _FakeSub:
    def get(self, h):
        raise KeyError(h)


def _rec(mid, text, salience):
    return MemoryRecord(memory_id=mid, content_hash=f"h_{mid}", raw_uri=f"cas://h_{mid}", text=text,
                        scope=Scope(), valid_at=1.0, salience=salience)


def test_select_scratchpad_activation_reranks_within_eligible():
    recs = [_rec("a", "alpha", 0.70), _rec("b", "beta", 0.65)]
    assert select_scratchpad(recs, top_k=1, min_salience=0.6)[0]["memory_id"] == "a"
    act = select_scratchpad(recs, top_k=1, min_salience=0.6, activation={"b": 1.0}, weight=0.25)
    assert act[0]["memory_id"] == "b"                     # activation promoted b into top-1


def test_select_scratchpad_activation_none_is_unchanged():
    recs = [_rec("a", "alpha", 0.70), _rec("b", "beta", 0.65)]
    assert (select_scratchpad(recs, top_k=2, min_salience=0.6)
            == select_scratchpad(recs, top_k=2, min_salience=0.6, activation=None))


def test_build_scratchpad_surfaces_activated_fact(fresh_settings):
    s = replace(fresh_settings, flow_activation_enabled=True, flow_context_weight=0.25,
                scratchpad_enabled=True, scratchpad_min_salience=0.6, scratchpad_topk=1)
    e = Engine(s, client=_Embed(s.embed_dim))
    e.store.upsert_record(_rec("a", "alpha fact", 0.70))
    e.store.upsert_record(_rec("b", "beta fact", 0.65))
    assert e.build_scratchpad(Scope(), top_k=1)[0]["memory_id"] == "a"   # salience alone
    e.activation.inject("default", ["b"], 1.0)
    assert e.build_scratchpad(Scope(), top_k=1)[0]["memory_id"] == "b"   # instinct promoted b


def test_assemble_context_scratchpad_reads_activation(fresh_settings):
    s = replace(fresh_settings, scratchpad_enabled=True, scratchpad_min_salience=0.6,
                scratchpad_topk=1, flow_context_weight=0.25, rerank_enabled=False)
    store = RecordStore(s.sqlite_path)
    for r in (_rec("a", "alpha fact", 0.70), _rec("b", "beta fact", 0.65)):
        store.upsert_record(r)
    r = Retriever(store, make_vector_index(s), KnowledgeGraph(store), _FakeSub(), _Embed(s.embed_dim), s)
    blocks_base = r.assemble_context("q", [], scope=Scope(), include_conflict_resolution=False)
    blocks_act = r.assemble_context("q", [], scope=Scope(), include_conflict_resolution=False,
                                    activation={"b": 1.0})
    assert any("alpha fact" in b for b in blocks_base)    # salience-only -> a
    assert any("beta fact" in b for b in blocks_act)      # activation -> b surfaces in context
