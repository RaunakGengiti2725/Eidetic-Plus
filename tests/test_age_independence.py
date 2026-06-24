"""Offline proof of the core recency-independence property at the index level.

The vector index has NO concept of age: ranking is pure content similarity. So a
memory inserted FIRST (oldest) is retrieved exactly as well as one inserted LAST
(newest). Uses synthetic test vectors (fixtures, not model outputs) -- the live
signature_demo.py proves the same with real embeddings."""
from __future__ import annotations

import numpy as np

from eidetic.vector_index import NumpyVectorIndex


def test_oldest_and_newest_equally_retrievable(tmp_path):
    dim, struct = 64, 8
    idx = NumpyVectorIndex(tmp_path, dim, struct)
    rng = np.random.default_rng(0)

    # Two distinctive targets + many distractors.
    old_vec = rng.normal(size=dim).astype(np.float32)
    new_vec = rng.normal(size=dim).astype(np.float32)

    idx.add("OLDEST", old_vec)                       # inserted first == "oldest"
    for i in range(500):                              # 500 distractors in between
        idx.add(f"d{i}", rng.normal(size=dim).astype(np.float32))
    idx.add("NEWEST", new_vec)                        # inserted last == "newest"

    k = 5
    old_hits = idx.search(old_vec + 0.01 * rng.normal(size=dim).astype(np.float32), k)
    new_hits = idx.search(new_vec + 0.01 * rng.normal(size=dim).astype(np.float32), k)

    assert old_hits[0][0] == "OLDEST"
    assert new_hits[0][0] == "NEWEST"
    # Top scores are comparable: age (insertion order) plays no role.
    assert abs(old_hits[0][1] - new_hits[0][1]) < 0.05


def test_recall_flat_across_insertion_order(tmp_path):
    dim, struct = 64, 8
    idx = NumpyVectorIndex(tmp_path, dim, struct)
    rng = np.random.default_rng(1)
    n = 300
    targets = rng.normal(size=(n, dim)).astype(np.float32)
    for i in range(n):
        idx.add(f"m{i}", targets[i])

    k = 5
    hits = []
    for i in range(n):
        q = targets[i] + 0.02 * rng.normal(size=dim).astype(np.float32)
        res = idx.search(q, k)
        hits.append(1 if any(mid == f"m{i}" for mid, _ in res) else 0)

    hits = np.array(hits)
    first_third = hits[: n // 3].mean()   # "oldest" inserted
    last_third = hits[-n // 3:].mean()    # "newest" inserted
    assert first_third > 0.95
    assert last_third > 0.95
    assert abs(first_third - last_third) < 0.05  # flat recall vs insertion age
