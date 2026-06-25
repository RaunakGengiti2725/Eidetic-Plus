"""Offline tests for the Phase-1 multi-view retrieval channels (structure / event / gist),
all flag-gated. No key. Proves each dormant channel surfaces memories the dense channel alone
would miss, and that the neutral path is unchanged when the flags are off."""
from __future__ import annotations

import types
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from eidetic.events import EventRecord
from eidetic.graph import KnowledgeGraph
from eidetic.models import DerivedRecord, MemoryRecord, Scope
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore


def _may(day=15):
    return datetime(2023, 5, day).timestamp()


class _FakeIndex:
    """Dense returns only m0,m1,m2; struct returns what the test programs; both honor allowed."""
    def __init__(self, dense, struct=None, vecs=None):
        self.dense, self.struct, self.vecs = dense, struct or [], vecs or {}

    def __len__(self):
        return max(len(self.dense), len(self.vecs), 1)

    def search(self, q, k, allowed_ids=None, ef=None):
        items = [(m, s) for m, s in self.dense if allowed_ids is None or m in allowed_ids]
        return sorted(items, key=lambda x: -x[1])[:k]

    def search_struct(self, qs, k):
        return self.struct[:k]

    def get_vectors(self, ids):
        return {m: self.vecs[m] for m in ids if m in self.vecs}


def _store_with(tmp_path, ns="proj"):
    store = RecordStore(tmp_path / "db.sqlite")
    scope = Scope(namespace=ns)
    for i in range(5):
        store.upsert_record(MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}",
                                         text=f"memory {i}", scope=scope, valid_at=1.0))
    return store, scope


def _retriever(tmp_path, settings, index):
    store, scope = _store_with(tmp_path)
    return Retriever(store, index, KnowledgeGraph(store), object(), object(), settings), store, scope


# ---- helper-level unit tests ---------------------------------------------------------------
def test_run_struct_filters_to_allowed(fresh_settings, tmp_path):
    idx = _FakeIndex(dense=[], struct=[("m0", 0.9), ("m1", 0.8), ("not_allowed", 0.7)])
    r, _, _ = _retriever(tmp_path, fresh_settings, idx)
    order, m = r._run_struct({"entities": ["alice"]}, allowed={"m0", "m1"})
    assert order == ["m0", "m1"] and "not_allowed" not in m


def test_run_event_boosts_temporally_matching_memory(fresh_settings, tmp_path):
    r, store, scope = _retriever(tmp_path, fresh_settings, _FakeIndex(dense=[]))
    store.add_event(EventRecord(subject="exercise", verb="did", object="run",
                                start=_may(), end=_may(), source_memory_id="m3",
                                namespace=scope.namespace, valid_at=_may()))
    parsed = {"entities": ["exercise"], "operation": "count",
              "ranges": [{"start": "2023-05-01T00:00:00", "end": "2023-05-31T23:59:59"}]}
    order, m = r._run_event(parsed, records={f"m{i}": None for i in range(5)}, at=None, scope=scope)
    assert order == ["m3"] and m["m3"] > 0


def test_run_gist_boosts_members_with_provenance(fresh_settings, tmp_path):
    r, store, scope = _retriever(tmp_path, fresh_settings, _FakeIndex(dense=[]))
    store.add_derived(DerivedRecord(cid="gist1", kind="gist", namespace=scope.namespace,
                                    member_ids=["m4"], vector=[1.0, 0.0]))
    order, m, prov = r._run_gist(np.array([1.0, 0.0], np.float32), scope, allowed={"m4"})
    assert order == ["m4"] and prov["m4"] == "gist1"


# ---- integration through retrieve() --------------------------------------------------------
def test_gist_channel_surfaces_a_dense_miss(fresh_settings, tmp_path):
    # dense returns only m0,m1,m2; m4 is reachable ONLY through a matching gist.
    qvec = np.array([1.0, 0.0], np.float32)
    idx = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85), ("m2", 0.8)],
                     vecs={f"m{i}": qvec for i in range(5)})
    base = replace(fresh_settings, rerank_enabled=False, gist_channel_enabled=False)
    r0, store0, scope = _retriever(tmp_path / "a", base, idx)
    store0.add_derived(DerivedRecord(cid="g", kind="gist", namespace=scope.namespace,
                                     member_ids=["m4"], vector=[1.0, 0.0]))
    off = {c.record.memory_id for c in r0.retrieve("q", scope=scope, qvec=qvec, use_recency=False)}

    on_s = replace(fresh_settings, rerank_enabled=False, gist_channel_enabled=True, rrf_w_gist=2.0)
    r1, store1, _ = _retriever(tmp_path / "b", on_s, idx)
    store1.add_derived(DerivedRecord(cid="g", kind="gist", namespace=scope.namespace,
                                     member_ids=["m4"], vector=[1.0, 0.0]))
    on = {c.record.memory_id for c in r1.retrieve("q", scope=scope, qvec=qvec, use_recency=False)}
    assert "m4" not in off and "m4" in on               # gist channel surfaced the dense miss


def test_struct_channel_surfaces_a_dense_miss(fresh_settings, tmp_path):
    qvec = np.array([1.0, 0.0], np.float32)
    idx = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85)], struct=[("m3", 0.95)])
    s = replace(fresh_settings, rerank_enabled=False, struct_channel_enabled=True, rrf_w_struct=2.0)
    r, _, scope = _retriever(tmp_path, s, idx)
    ids = {c.record.memory_id for c in r.retrieve("q", scope=scope, qvec=qvec, use_recency=False)}
    assert "m3" in ids                                  # structure channel surfaced m3


def test_neutral_path_unchanged_when_channels_off(fresh_settings, tmp_path):
    qvec = np.array([1.0, 0.0], np.float32)
    idx = _FakeIndex(dense=[("m0", 0.9), ("m1", 0.85), ("m2", 0.8)], struct=[("m3", 0.99)])
    s = replace(fresh_settings, rerank_enabled=False)    # all Phase-1 channels default OFF
    r, store, scope = _retriever(tmp_path, s, idx)
    store.add_derived(DerivedRecord(cid="g", kind="gist", namespace=scope.namespace,
                                    member_ids=["m4"], vector=[1.0, 0.0]))
    ids = {c.record.memory_id for c in r.retrieve("q", scope=scope, qvec=qvec, use_recency=False)}
    assert ids <= {"m0", "m1", "m2"}                     # no struct/gist leakage when off


# ---- age-independence of the new channels (the discriminating check) -----------------------
def test_structure_channel_is_age_invariant():
    # The structure code encodes only CYCLIC time. Two memories one year apart but at the SAME
    # weekday + time-of-day (364 = 52 weeks) have identical structure codes, so the structure
    # channel ranks them identically: it never decays with absolute age (flat recall-vs-age).
    from eidetic.structure_code import build_query_structure_code, build_structure_code
    dim = 128
    q = build_query_structure_code(["alice", "globex"], dim)
    t_new = 1_700_000_000.0
    t_old = t_new - 364 * 86400
    common = dict(content_hash="h", entities=["alice", "globex"], scope=Scope(namespace="t"))
    s_new = build_structure_code(MemoryRecord(valid_at=t_new, **common), dim)
    s_old = build_structure_code(MemoryRecord(valid_at=t_old, **common), dim)
    assert abs(float(q @ s_new) - float(q @ s_old)) < 1e-6


# ---- graph-vocab seeding -------------------------------------------------------------------
def test_vocab_seed_matches_store_vocabulary():
    from eidetic.retrieval import _vocab_seed_entities
    recs = [types.SimpleNamespace(entities=["green tea ceremony"]),
            types.SimpleNamespace(entities=["Acme Corp"]),
            types.SimpleNamespace(entities=["bob"])]
    seeds = _vocab_seed_entities("what did i say about tea and bob", recs)
    assert "green tea ceremony" in seeds and "bob" in seeds   # lowercase/multi-word matched
    assert "Acme Corp" not in seeds                            # unrelated entity not seeded


# ---- HNSW scoped exact-subset fallback (direct hnswlib backend; numpy suite cannot see it) --
def test_hnsw_scoped_search_returns_full_allowed_subset(tmp_path):
    pytest.importorskip("hnswlib")
    from eidetic.vector_index import HnswVectorIndex
    idx = HnswVectorIndex(tmp_path, dim=8, struct_dim=4, ef=64, M=16)
    rng = np.random.default_rng(0)
    vecs = {}
    for i in range(200):
        v = rng.standard_normal(8).astype(np.float32)
        vecs[f"m{i}"] = v
        idx.add(f"m{i}", v)
    allowed = {"m3", "m77", "m120", "m155", "m198"}      # scattered small in-scope subset
    res = idx.search(vecs["m120"], k=5, allowed_ids=allowed)
    assert {mid for mid, _ in res} == allowed            # exact-subset fallback fills the scope
    # a larger allowed set with k beyond the default ef still fills correctly
    allowed2 = {f"m{i}" for i in range(0, 200, 2)}
    assert len(idx.search(vecs["m120"], k=50, allowed_ids=allowed2)) == 50
