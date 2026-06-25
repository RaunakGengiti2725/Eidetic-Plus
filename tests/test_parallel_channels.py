"""S3: PARALLEL_CHANNELS must be safe (channels read-only after the backfill move) and equivalent
to the serial fan-out. Offline."""
from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.models import Scope
from eidetic.retrieval import Retriever


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


def test_parallel_channels_equivalent_and_thread_safe(fresh_settings):
    base = replace(fresh_settings, rerank_enabled=False)        # fake client only embeds
    e = Engine(base, client=_FakeEmbed(fresh_settings.embed_dim))
    scope = Scope(namespace="pc")
    for i in range(15):
        e.ingest_text(f"memory {i} alice bob carol topic{i % 3}", scope=scope, consolidate_now=False)
    q = e.client.embed_text("alice topic1")

    serial = [c.record.memory_id for c in
              e.retriever.retrieve("alice topic1", scope=scope, qvec=q, use_recency=True)]

    # a parallel-channels retriever over the SAME store/index/bm25 file.
    par = Retriever(e.store, e.index, e.graph, e.substrate, e.client,
                    replace(base, parallel_channels_enabled=True))
    parallel = [c.record.memory_id for c in
                par.retrieve("alice topic1", scope=scope, qvec=q, use_recency=True)]
    assert set(serial) == set(parallel)               # fan-out does not change the candidate set

    # many concurrent parallel-channel retrieves -> no race/exception (channels are read-only now).
    errors: list = []

    def worker():
        try:
            for _ in range(10):
                par.retrieve("alice topic1", scope=scope, qvec=q, use_recency=True)
        except Exception as ex:
            errors.append(repr(ex))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
