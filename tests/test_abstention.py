"""Offline tests for calibrated abstention (Phase 2): pure signals + the answer/abstain decision."""
from __future__ import annotations

import types
from dataclasses import replace

import pytest

from eidetic.abstention import (channel_agreement, combine_confidence, pick_tau,
                                proof_completeness)


def test_channel_agreement_counts_independent_channels():
    top = types.SimpleNamespace(dense_score=0.8, bm25_score=0.5, graph_score=0.0)
    assert channel_agreement(top) == pytest.approx(2 / 3)
    none = types.SimpleNamespace(dense_score=0.0, bm25_score=0.0, graph_score=0.0)
    assert channel_agreement(none) == 0.0


def test_proof_completeness_fraction_with_hashes():
    cits = [types.SimpleNamespace(content_hash="h"), types.SimpleNamespace(content_hash="")]
    assert proof_completeness(cits) == 0.5
    assert proof_completeness([]) == 0.0


def test_combine_confidence_is_weighted_sum():
    c = combine_confidence(1.0, 1.0, 1.0, 1.0, w_entail=0.4, w_coverage=0.2,
                           w_agreement=0.2, w_proof=0.2)
    assert c == pytest.approx(1.0)


def test_pick_tau_meets_precision_target():
    samples = [(0.9, True), (0.8, True), (0.7, True), (0.5, False), (0.4, False), (0.3, False)]
    tau = pick_tau(samples, precision_target=0.95)
    answered = [ok for c, ok in samples if c >= tau]
    assert answered and all(answered)          # everything answered at tau is correct
    assert 0.5 < tau <= 0.7                     # sits above the wrong item, keeps the correct ones


def test_pick_tau_abstains_everything_when_target_unreachable():
    tau = pick_tau([(0.9, False), (0.8, True)], precision_target=0.95)
    assert tau > 0.9                            # no tau reaches 0.95 precision -> answer nothing


def test_strong_evidence_answers_thin_evidence_abstains(fresh_settings):
    from eidetic.graph import KnowledgeGraph
    from eidetic.models import Citation, MemoryRecord, NLILabel, RetrievalCandidate, Scope
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    s = replace(fresh_settings, abstention_v2_enabled=True, abstention_v2_tau=0.5)
    store = RecordStore(s.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), object(), object(), s)
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text="Alice works at Acme",
                       scope=Scope(), valid_at=1.0)

    strong_c = [RetrievalCandidate(record=rec, dense_score=0.9, bm25_score=0.6, graph_score=0.4,
                                   fused_score=1.0)]
    strong_cit = [Citation(memory_id="m1", content_hash="h1", raw_uri="", source="u", valid_at=1.0,
                           nli_label=NLILabel.ENTAILMENT, nli_score=0.95)]
    conf_strong, _ = r._abstention_confidence(strong_c, strong_cit)
    assert conf_strong >= s.abstention_v2_tau          # strong, agreed, proven -> answer

    thin_c = [RetrievalCandidate(record=rec, dense_score=0.1, bm25_score=0.0, graph_score=0.0,
                                 fused_score=0.1)]
    thin_cit = [Citation(memory_id="m1", content_hash="", raw_uri="", source="u", valid_at=1.0,
                         nli_label=NLILabel.NEUTRAL, nli_score=0.2)]
    conf_thin, _ = r._abstention_confidence(thin_c, thin_cit)
    assert conf_thin < s.abstention_v2_tau             # thin, unentailed, unproven -> abstain
