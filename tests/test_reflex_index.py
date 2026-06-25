"""Offline tests for the ReflexIndex: a derived, namespace-partitioned inverted lookup
(entity/term -> memory_ids). It is a rebuildable cache -- the store stays authoritative for
bi-temporal validity and finer (agent/project) scope filtering, so the only thing the index
can get wrong is COVERAGE (a missing seed), never correctness."""
from __future__ import annotations

from eidetic.models import MemoryRecord, Scope
from eidetic.reflex_index import ReflexIndex, tokenize
from eidetic.store import RecordStore


def _rec(mid: str, text: str, *, namespace: str = "default", entities=None) -> MemoryRecord:
    return MemoryRecord(memory_id=mid, content_hash=f"h_{mid}", text=text,
                        scope=Scope(namespace=namespace), valid_at=1.0,
                        entities=list(entities or []))


def test_tokenize_drops_stopwords_and_shorts_and_lowercases():
    toks = set(tokenize("What was the Helios PROJECT revenue?"))
    assert "helios" in toks and "project" in toks and "revenue" in toks
    # question words / articles / sub-3-char tokens are dropped.
    assert "the" not in toks and "was" not in toks and "what" not in toks


def test_seeds_match_by_text_term():
    idx = ReflexIndex()
    idx.add_record(_rec("m1", "The Helios project quarterly revenue was 4.2 million"))
    idx.add_record(_rec("m2", "Bob enjoys hiking in the mountains"))
    assert idx.seeds("default", entities=[], terms=["helios", "revenue"]) == {"m1"}
    assert idx.seeds("default", entities=[], terms=["hiking"]) == {"m2"}


def test_seeds_match_by_explicit_entity():
    idx = ReflexIndex()
    idx.add_record(_rec("m1", "she works there", entities=["Acme Corporation"]))
    # entity lookup is normalized (case-insensitive)
    assert idx.seeds("default", entities=["acme corporation"], terms=[]) == {"m1"}
    assert idx.seeds("default", entities=["ACME CORPORATION"], terms=[]) == {"m1"}


def test_seeds_union_of_entity_and_term():
    idx = ReflexIndex()
    idx.add_record(_rec("m1", "alpha beta", entities=["Zeta"]))
    idx.add_record(_rec("m2", "gamma delta"))
    seeds = idx.seeds("default", entities=["Zeta"], terms=["gamma"])
    assert seeds == {"m1", "m2"}


def test_namespace_partition_is_a_hard_boundary():
    idx = ReflexIndex()
    idx.add_record(_rec("a1", "shared secret token", namespace="alpha"))
    idx.add_record(_rec("b1", "shared secret token", namespace="beta"))
    assert idx.seeds("alpha", entities=[], terms=["secret"]) == {"a1"}
    assert idx.seeds("beta", entities=[], terms=["secret"]) == {"b1"}
    assert idx.seeds("gamma", entities=[], terms=["secret"]) == set()


def test_rebuild_from_store_is_authoritative(fresh_settings):
    store = RecordStore(fresh_settings.sqlite_path)
    store.upsert_record(_rec("m1", "Helios revenue report"))
    store.upsert_record(_rec("m2", "Bob hiking trip", namespace="other"))
    idx = ReflexIndex()
    n = idx.rebuild_from_store(store)
    assert n == 2
    assert idx.seeds("default", entities=[], terms=["helios"]) == {"m1"}
    assert idx.seeds("other", entities=[], terms=["hiking"]) == {"m2"}


def test_ensure_built_is_idempotent(fresh_settings):
    store = RecordStore(fresh_settings.sqlite_path)
    store.upsert_record(_rec("m1", "alpha"))
    idx = ReflexIndex()
    idx.ensure_built(store)
    idx.ensure_built(store)  # second call must not duplicate or re-scan
    assert idx.seeds("default", entities=[], terms=["alpha"]) == {"m1"}
    assert idx.built is True


def test_seeds_are_deterministic():
    idx = ReflexIndex()
    for i in range(20):
        idx.add_record(_rec(f"m{i}", f"common term variant {i}"))
    a = idx.seeds("default", entities=[], terms=["common", "term"])
    b = idx.seeds("default", entities=[], terms=["term", "common"])
    assert a == b and len(a) == 20
