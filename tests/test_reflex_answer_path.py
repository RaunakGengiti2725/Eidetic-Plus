"""Engine integration for Track 1.4: reflex recall wired into ask() as a flag-gated fast path.

Invariants pinned here:
  * REFLEX_RECALL=0 is byte-identical -- the reflex index is never built or touched.
  * A confident reflex hit feeds answer() candidates whose dense/rerank scores are populated, so
    it never spuriously abstains and confidence is not capped at 0.5 (the precomputed-contract bug).
  * A low-coverage reflex miss falls back to full retrieval; the final answer still verifies.
  * The fast path never skips NLI/abstention/proof.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.models import BrainEventType, Scope


class _FakeReader:
    """Deterministic offline embed + reader + NLI. Entailment fires when the source text (the
    NLI premise) mentions 'helios' -- enough to exercise verify/abstain end-to-end with no key."""
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

    def generate_answer(self, q, blocks, model=None):
        self.gen_calls += 1
        return "The Helios project quarterly revenue was 4.2 million dollars"

    def nli(self, premise, hypothesis):
        return ("entailment", 0.92) if "helios" in (premise or "").lower() else ("neutral", 0.2)


def _engine(fresh_settings, **overrides):
    s = replace(fresh_settings, **overrides)
    return Engine(s, client=_FakeReader(s.embed_dim))


def test_flag_off_never_builds_reflex_index(fresh_settings):
    e = _engine(fresh_settings, reflex_recall_enabled=False, brain_events_enabled=True,
                semantic_cache_enabled=False, rerank_enabled=False)
    e.ingest_text("The Helios project quarterly revenue was 4.2 million dollars",
                  source="memo", extract_graph=False, consolidate_now=False)
    assert e.reflex_index.built is False
    ans = e.ask("What was the Helios project revenue?")
    assert ans.answer
    assert e.reflex_index.built is False
    assert not e.brain_log.by_type(BrainEventType.REFLEX_HIT)
    assert not e.brain_log.by_type(BrainEventType.REFLEX_MISS)


def test_reflex_recall_public_api_returns_packet(fresh_settings):
    e = _engine(fresh_settings, reflex_recall_enabled=True)
    e.ingest_text("The Helios project quarterly revenue was 4.2 million dollars",
                  source="memo", extract_graph=False, consolidate_now=False)
    packet = e.reflex_recall("What was the Helios project revenue?")
    ids = packet.candidate_ids()
    assert ids
    assert packet.coverage > 0.0


def test_index_maintained_incrementally_when_enabled(fresh_settings):
    e = _engine(fresh_settings, reflex_recall_enabled=True)
    assert e.reflex_index.built is True  # built from (empty) store at construction
    rec = e.ingest_text("Helios alpha keyword target memory", source="memo", extract_graph=False, consolidate_now=False)
    mid = rec.memory_id
    assert e.reflex_index.seeds("default", entities=[], terms=["helios", "alpha"]) == {mid}


def test_confident_reflex_hit_does_not_spuriously_abstain(fresh_settings):
    e = _engine(fresh_settings, reflex_recall_enabled=True, brain_events_enabled=True,
                semantic_cache_enabled=False)
    e.ingest_text("The Helios project quarterly revenue was 4.2 million dollars",
                  source="memo", extract_graph=False, consolidate_now=False)
    ans = e.ask("What was the Helios project revenue?")
    assert not e.brain_log.by_type(BrainEventType.REFLEX_FALLBACK)
    assert not ans.note.startswith("abstained")
    assert ans.verified is True
    assert ans.generated_by == "smqe"
    assert ans.note.startswith("smqe:")
    assert ans.confidence > 0.5
    assert ans.citations
    assert all(c.nli_label.value == "entailment" for c in ans.citations)
    assert sum(len(c.snippet or "") for c in ans.citations) < 160
    assert e.client.gen_calls == 0


def test_low_coverage_reflex_falls_back_to_full_retrieval(fresh_settings):
    # SMQE runs before reflex; when it can answer, an impossible reflex bar is irrelevant.
    e = _engine(fresh_settings, reflex_recall_enabled=True, reflex_min_coverage=1.5,
                brain_events_enabled=True, semantic_cache_enabled=False, rerank_enabled=False)
    e.ingest_text("The Helios project quarterly revenue was 4.2 million dollars",
                  source="memo", extract_graph=False, consolidate_now=False)
    ans = e.ask("What was the Helios project revenue?")
    assert not e.brain_log.by_type(BrainEventType.REFLEX_MISS)
    assert not e.brain_log.by_type(BrainEventType.REFLEX_FALLBACK)
    assert ans.answer
    assert ans.verified is True


class _FakeExtractor(_FakeReader):
    """Fast-path embed + a deterministic extract_edges that yields an entity token absent from the
    record text -- so the entity only reaches the index via consolidation, not the text-term path."""
    def __init__(self, dim, triples):
        super().__init__(dim)
        self.triples = triples

    def extract_edges(self, text):
        return list(self.triples)


def test_consolidate_pending_indexes_extracted_entities(fresh_settings):
    triples = [{"src": "Zephyr", "relation": "owns", "dst": "the asset",
                "fact": "Zephyr owns the asset"}]
    s = replace(fresh_settings, reflex_recall_enabled=True)
    e = Engine(s, client=_FakeExtractor(s.embed_dim, triples))
    rec = e.ingest_text("the quarterly arrangement was finalized recently",
                        source="memo", extract_graph=False, consolidate_now=False)
    mid = rec.memory_id
    # fast path indexes only text terms -> the extracted entity is not seedable yet.
    assert e.reflex_index.seeds("default", entities=["Zephyr"], terms=[]) == set()
    e.consolidate_pending(score_importance=False)
    # consolidation populated rec.entities AND must re-index reflex (no rebuild required).
    assert mid in e.reflex_index.seeds("default", entities=["Zephyr"], terms=[])


def test_reflex_hit_sets_a_fresh_recall_trace(fresh_settings):
    """A structured fast-path hit bypasses retrieve(), so proof recall-paths and channel telemetry
    still need an honest trace for the current query."""
    e = _engine(fresh_settings, reflex_recall_enabled=True, recall_trace_enabled=True,
                brain_events_enabled=True, semantic_cache_enabled=False)
    e.ingest_text("The Helios project quarterly revenue was 4.2 million dollars",
                  source="memo", extract_graph=False, consolidate_now=False)
    e.ask("What was the Helios project revenue?")
    tr = e.retriever.last_trace
    assert tr is not None
    assert tr.query == "What was the Helios project revenue?"
    assert "smqe" in tr.enabled_channels
    assert tr.selected_candidates


def test_reflex_recall_event_types_exist():
    for name in ("REFLEX_HIT", "REFLEX_MISS", "REFLEX_FALLBACK"):
        assert hasattr(BrainEventType, name)
