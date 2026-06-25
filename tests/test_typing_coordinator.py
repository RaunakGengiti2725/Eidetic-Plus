"""Offline tests for the memory-typing coordinator (Connected Brain Loop, Phase 4)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from eidetic.graph import KnowledgeGraph
from eidetic.memory_types import MemoryType, classify_memory_type
from eidetic.models import MemoryRecord, Scope
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore


class _FakeIndex:
    def __init__(self, dense):
        self.dense = dense

    def __len__(self):
        return max(len(self.dense), 1)

    def search(self, q, k, allowed_ids=None, ef=None):
        items = [(m, s) for m, s in self.dense if allowed_ids is None or m in allowed_ids]
        return sorted(items, key=lambda x: -x[1])[:k]

    def get_vectors(self, ids):
        return {}


def _build(tmp_path, settings):
    store = RecordStore(tmp_path / "db.sqlite")
    scope = Scope(namespace="t")
    # m_epi has the STRONGER dense score; m_core is a standing preference.
    store.upsert_record(MemoryRecord(memory_id="m_epi", content_hash="he", text="went jogging",
                                     scope=scope, valid_at=1.0, metadata={"type": "episodic"}))
    store.upsert_record(MemoryRecord(memory_id="m_core", content_hash="hc",
                                     text="green tea ceremony", scope=scope, valid_at=1.0,
                                     metadata={"type": "core"}))
    idx = _FakeIndex(dense=[("m_epi", 0.85), ("m_core", 0.80)])
    s = replace(settings, rerank_enabled=False, persistent_bm25_enabled=False)
    return Retriever(store, idx, KnowledgeGraph(store), object(), object(), s), scope, s


def test_classifier_maps_signals_to_types():
    assert classify_memory_type("my favorite drink is tea") == MemoryType.CORE
    assert classify_memory_type("how to deploy the service") == MemoryType.PROCEDURAL
    assert classify_memory_type("the api key is sk-123") == MemoryType.KNOWLEDGE_VAULT
    assert classify_memory_type("a chart", modality="image") == MemoryType.RESOURCE
    assert classify_memory_type("we met on tuesday") == MemoryType.EPISODIC


def test_type_prior_reorders_for_a_preference_query(fresh_settings, tmp_path):
    qvec = np.array([1.0, 0.0], np.float32)
    q = "what is my favorite drink"

    r_off, scope, _ = _build(tmp_path / "off", fresh_settings)         # MEMORY_TYPING off
    off = [c.record.memory_id for c in r_off.retrieve(q, scope=scope, qvec=qvec, use_recency=False)]

    r_on, scope2, _ = _build(tmp_path / "on", replace(fresh_settings, memory_typing_enabled=True))
    on = [c.record.memory_id for c in r_on.retrieve(q, scope=scope2, qvec=qvec, use_recency=False)]

    assert off[0] == "m_epi"        # baseline: stronger dense score wins
    assert on[0] == "m_core"        # typing prior surfaces the preference (core) for a pref query


def test_type_prior_is_noop_for_a_factual_query(fresh_settings, tmp_path):
    # A factual query should NOT promote the core/preference memory over the stronger dense hit.
    qvec = np.array([1.0, 0.0], np.float32)
    r_on, scope, _ = _build(tmp_path, replace(fresh_settings, memory_typing_enabled=True))
    on = [c.record.memory_id for c in r_on.retrieve("where did the jog happen", scope=scope,
                                                    qvec=qvec, use_recency=False)]
    assert on[0] == "m_epi"         # episodic preferred for a factual/temporal-ish query; no flip
