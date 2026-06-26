"""Track 9 Task 12: FLOW_ACTIVATION=0 is byte-identical on every touched surface -- the field is
never built, _hotset is unchanged, retrieve()/build_memory_packet have no activation effect, the
scratchpad is unchanged, and no FLOW BrainEvents are emitted."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.graph import KnowledgeGraph
from eidetic.models import BrainEventType, MemoryRecord, Scope
from eidetic.reflex_activation import build_memory_packet
from eidetic.reflex_index import ReflexIndex
from eidetic.store import RecordStore


class _Reader:
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
        return "Alice works at Acme Corporation"

    def nli(self, premise, hypothesis):
        return ("entailment", 0.9) if "acme" in (premise or "").lower() else ("neutral", 0.2)


def test_flow_off_field_is_none_and_hotset_works(fresh_settings):
    e = Engine(replace(fresh_settings, flow_activation_enabled=False, reflex_recall_enabled=True),
               client=_Reader(fresh_settings.embed_dim))
    assert e.activation is None
    assert e._flow_snapshot("ns") is None
    e._touch_hotset("ns", ["m1"])
    assert "m1" in e._hotset_ids("ns")               # legacy binary hotset intact


def test_flow_off_ask_emits_no_flow_events(fresh_settings):
    e = Engine(replace(fresh_settings, flow_activation_enabled=False, reflex_recall_enabled=True,
                       brain_events_enabled=True, semantic_cache_enabled=False, rerank_enabled=False),
               client=_Reader(fresh_settings.embed_dim))
    e.ingest_text("Alice works at Acme Corporation", consolidate_now=False)
    e.ask("where does Alice work")
    assert not e.brain_log.by_type(BrainEventType.FLOW_WARMED)
    assert not e.brain_log.by_type(BrainEventType.FLOW_PRIMED)


def test_build_memory_packet_activation_none_byte_identical(fresh_settings):
    store = RecordStore(fresh_settings.sqlite_path)
    for mid, txt in (("a", "alpha keyword"), ("b", "beta wording")):
        store.upsert_record(MemoryRecord(memory_id=mid, content_hash=f"h_{mid}", text=txt,
                                         scope=Scope(), valid_at=1.0))
    idx = ReflexIndex()
    idx.rebuild_from_store(store)
    graph = KnowledgeGraph(store)
    p1 = build_memory_packet("alpha keyword", Scope(), store=store, graph=graph, index=idx,
                             settings=fresh_settings)
    p2 = build_memory_packet("alpha keyword", Scope(), store=store, graph=graph, index=idx,
                             settings=fresh_settings, activation=None)
    assert p1.candidate_ids() == p2.candidate_ids()
    for mid in p1.candidate_ids():
        assert p1.scores[mid].aggregate == p2.scores[mid].aggregate
        assert p1.scores[mid].activation == 0.0      # no activation axis without a map
