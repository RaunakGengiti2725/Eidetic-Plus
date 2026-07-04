"""Offline tests for S2 write-path throughput: batched ingest_many, index rebuild, windowed scan."""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from eidetic.dashscope_client import ModelCallError
from eidetic.engine import Engine
from eidetic.ingestion import from_text
from eidetic.models import MemoryRecord, Scope
from eidetic.store import RecordStore
from eidetic.vector_index import make_vector_index


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


class _FailingEmbed(_FakeEmbed):
    def embed_text(self, t):
        raise ModelCallError("embed failed")


def _substrate_files(root):
    return [p for p in root.rglob("*") if p.is_file()]


def test_failed_embed_does_not_leave_orphan_raw_blob(fresh_settings):
    e = Engine(fresh_settings, client=_FailingEmbed(fresh_settings.embed_dim))
    with pytest.raises(ModelCallError, match="embed failed"):
        e.ingest_text("this write should not commit raw bytes", consolidate_now=False)
    assert e.store.count() == 0
    assert len(e.index) == 0
    assert _substrate_files(fresh_settings.substrate_dir) == []


def test_ingest_many_batches_and_dedups(fresh_settings):
    e = Engine(fresh_settings, client=_FakeEmbed(fresh_settings.embed_dim))
    scope = Scope(namespace="bulk")
    items = [from_text(f"memory {i} about alice and bob", "user") for i in range(25)]
    recs = e.ingest_many(items, scope=scope)
    assert len(recs) == 25 and len(e.index) == 25 and e.store.count(scope) == 25
    # re-ingesting an identical item dedups (same memory_id, no new index entry).
    again = e.ingest_many([items[0]], scope=scope)
    assert again[0].memory_id == recs[0].memory_id and len(e.index) == 25


def test_identical_text_at_different_times_remains_distinct_memory_event(fresh_settings):
    e = Engine(fresh_settings, client=_FakeEmbed(fresh_settings.embed_dim))
    scope = Scope(namespace="repeat")

    first = e.ingest_text("I spent 4 hours on the field guide.", scope=scope,
                          valid_at=100.0, consolidate_now=False)
    same_time = e.ingest_text("I spent 4 hours on the field guide.", scope=scope,
                              valid_at=100.0, consolidate_now=False)
    later = e.ingest_text("I spent 4 hours on the field guide.", scope=scope,
                          valid_at=200.0, consolidate_now=False)

    assert same_time.memory_id == first.memory_id
    assert later.memory_id != first.memory_id
    assert later.content_hash == first.content_hash
    assert e.store.count(scope) == 2


def test_rebuild_index_from_store_recovers_a_lost_index(fresh_settings):
    e = Engine(fresh_settings, client=_FakeEmbed(fresh_settings.embed_dim))
    scope = Scope(namespace="r")
    for i in range(10):
        e.ingest_text(f"memory {i} zulu", scope=scope, consolidate_now=False)
    assert len(e.index) == 10

    # simulate a lost/corrupt index: drop the files + swap in a fresh empty index.
    for f in fresh_settings.index_dir.glob("numpy_index*"):
        f.unlink()
    e.index = make_vector_index(fresh_settings)
    e.retriever.index = e.index
    assert len(e.index) == 0

    assert e.rebuild_index_from_store()["rebuilt"] == 10        # rebuilt from substrate + SQLite
    assert len(e.index) == 10
    res = e.index.search(e.client.embed_text("memory 3 zulu"), 3)
    assert res and res[0][0]                                    # search works after rebuild


def test_records_in_time_range_is_windowed(fresh_settings):
    store = RecordStore(fresh_settings.sqlite_path)
    scope = Scope(namespace="t")
    for t in (100, 200, 5000, 10000):
        store.upsert_record(MemoryRecord(memory_id=f"m{t}", content_hash=f"h{t}", text="x",
                                         scope=scope, valid_at=float(t)))
    got = {r.memory_id for r in store.records_in_time_range(150, 6000, scope)}
    assert got == {"m200", "m5000"}                            # only the in-window records


def test_ingest_many_dedups_within_one_batch(fresh_settings):
    """Two identical items in ONE call used to become two records: the store check runs
    before any write, so neither saw the other. First occurrence wins; later duplicates
    resolve to its record. (Deferred #17's blocking bug.)"""
    e = Engine(fresh_settings, client=_FakeEmbed(fresh_settings.embed_dim))
    scope = Scope(namespace="batch-dup")
    items = [from_text("alice waters the fern", "user"),
             from_text("bob repots the cactus", "user"),
             from_text("alice waters the fern", "user")]     # exact duplicate of item 0
    recs = e.ingest_many(items, scope=scope)
    assert len(recs) == 3
    assert recs[0].memory_id == recs[2].memory_id            # duplicate resolved, not re-written
    assert e.store.count(scope) == 2 and len(e.index) == 2
