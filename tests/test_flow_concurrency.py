"""Track 9 concurrency gate: the per-turn decay+inject+spread sequence is atomic under the
per-namespace turn lock, so concurrent asks cannot double-decay or lose an injection. Stress the
hub from many threads and assert no exception and a consistent field."""
from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.models import MemoryRecord, Scope


class _Embed:
    def __init__(self, dim):
        self.dim = dim

    def embed_text(self, t):
        return np.zeros(self.dim, np.float32)

    def embed_texts(self, ts):
        return np.zeros((len(ts), self.dim), np.float32)


def test_concurrent_commit_and_begin_turn_no_crash(fresh_settings):
    e = Engine(replace(fresh_settings, flow_activation_enabled=True, flow_decay=0.9),
               client=_Embed(fresh_settings.embed_dim))
    ns = Scope(namespace="ns")
    for i in range(20):
        e.store.upsert_record(MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}", text="x",
                                           scope=ns, valid_at=1.0))
    errors: list = []

    def worker(k):
        try:
            for _ in range(40):
                e._flow_begin_turn("ns", "q", scope=ns, as_of=None)
                e._flow_commit_recall("ns", [f"m{k}", f"m{(k + 1) % 20}"], ns, 1.0)
                _ = e._flow_snapshot("ns")
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    snap = e._flow_snapshot("ns")
    assert all(0.0 <= v <= 1.0 for v in snap.values())   # every activation stays in range


def test_concurrent_namespaces_isolated(fresh_settings):
    e = Engine(replace(fresh_settings, flow_activation_enabled=True),
               client=_Embed(fresh_settings.embed_dim))
    errors: list = []

    def worker(ns_name):
        try:
            for _ in range(50):
                e._flow_commit_recall(ns_name, [f"{ns_name}_m"], Scope(namespace=ns_name), 1.0)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("A", "B", "C")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert "A_m" in e._flow_snapshot("A") and "A_m" not in e._flow_snapshot("B")
