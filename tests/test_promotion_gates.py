"""Track 7: the promotion gates as EXECUTABLE checks. A flag may default-on only if it preserves
the invariants. This runs QualityGate over the product config's actual signals (age-independence,
scope isolation, raw immutability) and asserts a passing QualityGateResult -- the gate the plan
describes, encoded as a test rather than a checklist."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.brain import QualityGate
from eidetic.engine import Engine
from eidetic.models import MemoryRecord, Scope


class _Embed:
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


def _product_engine(fresh_settings):
    s = replace(fresh_settings, reflex_recall_enabled=True, cache_versioning_enabled=True,
                false_premise_enabled=True, brain_events_enabled=True, rerank_enabled=False)
    return Engine(s, client=_Embed(s.embed_dim))


def test_reflex_age_independence_gate(fresh_settings):
    """Promotion gate: enabling reflex must not make recall depend on memory age. Same content at
    very different ages, with no query time constraint, must score identically."""
    e = _product_engine(fresh_settings)
    ns = Scope(namespace="proj")
    e.store.upsert_record(MemoryRecord(memory_id="young", content_hash="hy",
                                       text="project alpha milestone details", scope=ns,
                                       valid_at=1_700_000_000.0))
    e.store.upsert_record(MemoryRecord(memory_id="old", content_hash="ho",
                                       text="project alpha milestone details", scope=ns,
                                       valid_at=1_000_000.0))
    e.reflex_index.rebuild_from_store(e.store)
    p = e.reflex_recall("tell me about project alpha milestone", scope=ns)
    recall_gap = abs(p.scores["young"].aggregate - p.scores["old"].aggregate)
    assert QualityGate.no_age_bias(recall_gap, 0.0)          # gap ~ 0 -> age-independent


def test_product_config_passes_quality_gate(fresh_settings):
    """The combined product config evaluated against the promotion gates -> a passing result."""
    e = _product_engine(fresh_settings)
    ns = Scope(namespace="proj")
    e.store.upsert_record(MemoryRecord(memory_id="m1", content_hash="h1",
                                       text="Alice works at Acme", scope=ns, valid_at=100.0))
    e.store.upsert_record(MemoryRecord(memory_id="m2", content_hash="h2",
                                       text="Alice works at Acme", scope=ns, valid_at=1_000_000.0))
    # a foreign-namespace record with the SAME content -> the no_scope_leak gate now has teeth: a
    # reflex/store regression that crossed the namespace boundary would surface it as a candidate.
    e.store.upsert_record(MemoryRecord(memory_id="leak", content_hash="h2",
                                       text="Alice works at Acme", scope=Scope(namespace="other"),
                                       valid_at=100.0))
    e.reflex_index.rebuild_from_store(e.store)

    raw_before = sorted(r.content_hash for r in e.store.all_records(None))
    p = e.reflex_recall("where does Alice work", scope=ns)              # a read; must not mutate raw
    raw_after = sorted(r.content_hash for r in e.store.all_records(None))

    # age-independence over the two same-content, different-age records.
    gap = abs(p.scores["m1"].aggregate - p.scores["m2"].aggregate)
    # scope isolation: every candidate the reflex path surfaced is visible to the query scope.
    result_scopes = [e.store.get_record(mid).scope for mid in p.candidate_ids()]

    checks = {
        "no_age_bias": QualityGate.no_age_bias(gap, 0.0),
        "no_raw_mutation": QualityGate.no_raw_mutation(raw_before, raw_after),
        "no_scope_leak": QualityGate.no_scope_leak(result_scopes, ns),
    }
    res = QualityGate.evaluate("reflex+sync+false_premise+truth_ledger", checks=checks)
    assert res.passed
    assert all(res.checks.values())


def test_off_by_default_flags_do_not_run_when_off(fresh_settings):
    """Gate #1 (flag-off behavior unchanged), concretely: with the off-by-default flags off, none
    of their machinery activates -- no reflex index, no false-premise gate, no new state."""
    e = Engine(replace(fresh_settings, reflex_recall_enabled=False, false_premise_enabled=False),
               client=_Embed(fresh_settings.embed_dim))
    e.store.upsert_record(MemoryRecord(memory_id="m1", content_hash="h1",
                                       text="Alice works at Acme", scope=Scope(), valid_at=1.0))
    assert e.reflex_index.built is False                     # reflex index never built when off
    assert e.check_false_premise("Why did Alice leave Google?") is not None  # method works...
    # ...but the ask() gate is inert when the flag is off (covered end-to-end in test_false_premise).
