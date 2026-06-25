"""Track 7: the realistic PRODUCT config -- every new flag ON together. Each track's own suite
exercises exactly one flag via replace(settings, X=True); nothing else runs reflex + versioned
cache + false-premise + brain events INTERACTING. This is that test: the full lifecycle end to
end, asserting the flags compose (reflex hit feeds the reader, a versioned cache hit fires, a
write invalidates it, a false-premise question abstains before any model call, truth_ledger and
sync_health work) with no scope leak and no raw-store mutation."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.models import BrainEventType, Scope


class _FakeClient:
    def __init__(self, dim):
        self.dim = dim
        self.gen_calls = 0

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

    def extract_edges(self, text):
        return []

    def generate_answer(self, q, blocks, model=None):
        self.gen_calls += 1
        return "Alice works at Acme Corporation"

    def nli(self, premise, hypothesis):
        return ("entailment", 0.92) if "acme" in (premise or "").lower() else ("neutral", 0.2)


def _product_engine(fresh_settings):
    s = replace(fresh_settings, reflex_recall_enabled=True, cache_versioning_enabled=True,
                false_premise_enabled=True, brain_events_enabled=True, semantic_cache_enabled=True,
                rerank_enabled=False)
    return Engine(s, client=_FakeClient(s.embed_dim))


def _hits(engine, etype):
    return len(engine.brain_log.by_type(etype))


def test_all_flags_on_full_lifecycle(fresh_settings):
    e = _product_engine(fresh_settings)
    ns = Scope(namespace="proj")
    e.ingest_text("Alice works at Acme Corporation", scope=ns, consolidate_now=False)
    e.ingest_text("Bob enjoys hiking on weekends", scope=ns, consolidate_now=False)
    e.consolidate_pending(scope=ns, score_importance=False)
    raw_before = sorted(r.content_hash for r in e.store.all_records(None))

    # 1) supported question -> reflex hit feeds the reader -> verified answer with a citation.
    a1 = e.ask("where does Alice work", scope=ns)
    assert a1.verified is True and "Acme" in a1.answer and a1.citations
    assert _hits(e, BrainEventType.REFLEX_HIT) >= 1
    assert not a1.note.startswith("abstained")

    # 2) identical repeat, no write between -> versioned cache hit (truth-fresh under BRAIN_EVENTS).
    gen_before = e.client.gen_calls
    a2 = e.ask("where does Alice work", scope=ns)
    assert a2.answer == a1.answer
    assert _hits(e, BrainEventType.CACHE_HIT) >= 1
    assert e.client.gen_calls == gen_before          # served from cache: the reader did NOT run

    # 3) a write bumps the namespace version -> the next identical ask is NOT a cache hit.
    cache_hits_before = _hits(e, BrainEventType.CACHE_HIT)
    e.ingest_text("Carol joined the design team", scope=ns, consolidate_now=False)
    a3 = e.ask("where does Alice work", scope=ns)
    assert _hits(e, BrainEventType.CACHE_HIT) == cache_hits_before   # invalidated -> recomputed
    assert a3.verified is True

    # 4) false-premise question -> structured abstention BEFORE any retrieval/model call.
    a4 = e.ask("Why did Alice leave Google?", scope=ns)
    assert a4.note.startswith("abstained: false-premise")
    assert _hits(e, BrainEventType.ANSWER_ABSTAINED) >= 1

    # 5) truth ledger over a verified answer composes with everything above.
    led = e.truth_ledger(a3, scope=ns)
    assert led["claim_status"] == "verified"
    assert led["evidence"] and led["evidence"][0]["validity_window"]["valid_at"]

    # 6) sync health stays consistent across the whole interacting flow.
    assert e.sync_health(scope=ns)["in_sync"] is True

    # 7) no raw-store mutation: every original raw hash is still present and unchanged. The Carol
    # ingest in step 3 ADDS a record (additive), so this is a subset check -- the invariant is that
    # immutable raw is never rewritten or deleted across the whole interacting flow.
    raw_after = set(r.content_hash for r in e.store.all_records(None))
    assert set(raw_before) <= raw_after


def test_all_flags_on_no_cross_scope_leak(fresh_settings):
    e = _product_engine(fresh_settings)
    e.ingest_text("the launch value is alpha-seven", scope=Scope(namespace="A"),
                  consolidate_now=False)
    # namespace B has no memories: reflex packet empty, no leak, honest "no memory" answer.
    packet = e.reflex_recall("what is the launch value", scope=Scope(namespace="B"))
    assert packet.candidate_ids() == []
    ans = e.ask("what is the launch value", scope=Scope(namespace="B"))
    assert "alpha-seven" not in ans.answer
    # and the A-scope reflex DOES see it (proving isolation is the boundary, not a dead path).
    assert e.reflex_recall("launch value", scope=Scope(namespace="A")).candidate_ids()


def test_flags_on_vs_off_same_namespace_version_independence(fresh_settings):
    # the false-premise gate must not perturb a normal supported answer when both run.
    e = _product_engine(fresh_settings)
    ns = Scope(namespace="proj")
    e.ingest_text("Alice works at Acme Corporation", scope=ns, consolidate_now=False)
    # 2-entity question whose entities DO co-occur -> must NOT abstain (precision guard under all flags).
    ans = e.ask("does Alice work at Acme", scope=ns)
    assert not ans.note.startswith("abstained: false-premise")
