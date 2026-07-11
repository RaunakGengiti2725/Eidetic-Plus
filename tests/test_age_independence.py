"""Offline proof of the core recency-independence property, at BOTH levels.

Index level: the vector index has NO concept of age -- ranking is pure content
similarity, so a memory inserted FIRST (oldest) is retrieved exactly as well as one
inserted LAST (newest). Uses synthetic test vectors (fixtures, not model outputs);
the live signature_demo.py proves the same with real embeddings.

Full-path level (WP5): the SHIPPED hybrid ranker (`Retriever.retrieve` -- dense +
BM25 + graph fusion) must be a pure function of CONTENT: permuting the records'
valid_at timestamps cannot change the ranking. The recency channel exists only when
a run explicitly opts in with a positive RRF_W_RECENCY -- at the 0.0 default the
channel is dead, including its underfill fallback. `Engine.prove_age_independence`
probes this same full path live."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore
from eidetic.vector_index import NumpyVectorIndex


class _NoSubstrate:
    def get(self, content_hash):
        raise KeyError(content_hash)


def _full_path_ranking(tmp_path, settings, valid_ats: list[float]) -> list[str]:
    """Build a real store+index corpus (deterministic synthetic vectors keyed by text)
    and return retrieve()'s ranked memory ids. Age enters ONLY through valid_ats."""
    dim = 32
    rng = np.random.default_rng(11)
    texts = [f"note {i}: the {w} ledger entry for project {w}" for i, w in
             enumerate(["amber", "basalt", "cedar", "delta", "ember", "flint",
                        "garnet", "harbor", "iris", "juniper"])]
    store = RecordStore(tmp_path / "age.sqlite")
    index = NumpyVectorIndex(tmp_path / "idx", dim, 4)
    scope = Scope(namespace="age-perm")
    vecs = {}
    for i, (text, valid_at) in enumerate(zip(texts, valid_ats)):
        rec = MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}", text=text,
                           scope=scope, valid_at=valid_at)
        store.upsert_record(rec)
        vecs[f"m{i}"] = rng.normal(size=dim).astype(np.float32)
        index.add(f"m{i}", vecs[f"m{i}"])
    r = Retriever(store, index, KnowledgeGraph(store), _NoSubstrate(), object(), settings)
    qvec = vecs["m3"] + 0.01 * rng.normal(size=dim).astype(np.float32)
    cands = r.retrieve("the delta ledger entry", at=max(valid_ats) + 10.0, scope=scope,
                       qvec=qvec, skip_rerank=True)
    return [c.record.memory_id for c in cands]


def test_full_path_ranking_is_invariant_under_age_permutation(fresh_settings, tmp_path):
    s = replace(fresh_settings, rerank_enabled=False)
    assert s.rrf_w_recency == 0.0  # the age-neutral default IS the product config
    young_to_old = [1_700_000_000.0 - i * 5_000_000 for i in range(10)]
    old_to_young = list(reversed(young_to_old))

    rank_a = _full_path_ranking(tmp_path / "a", s, young_to_old)
    rank_b = _full_path_ranking(tmp_path / "b", s, old_to_young)

    assert rank_a, "retrieve() returned no candidates"
    assert rank_a == rank_b
    assert rank_a[0] == "m3"


def test_recency_channel_requires_explicit_opt_in(fresh_settings, tmp_path):
    """With a positive weight the channel exists again (the documented ablation switch),
    so the two age orderings may rank differently -- proving the 0.0 default is what
    guarantees neutrality, not luck."""
    s = replace(fresh_settings, rerank_enabled=False, rrf_w_recency=0.9)
    young_to_old = [1_700_000_000.0 - i * 5_000_000 for i in range(10)]
    old_to_young = list(reversed(young_to_old))

    rank_a = _full_path_ranking(tmp_path / "a", s, young_to_old)
    rank_b = _full_path_ranking(tmp_path / "b", s, old_to_young)

    assert rank_a and rank_b
    assert rank_a != rank_b


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
