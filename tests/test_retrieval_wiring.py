"""Offline tests for the two previously-dead retrieval flags now wired in (no real model calls).

ACTIVE_RETRIEVAL folds an anticipated topic into the EMBED query (retrieve path, all eidetic
rows). COVE factored-verifies a grounded draft and demotes it to unverified on a failed check
(answer path -> engine.ask/product row only). Both default OFF and byte-identical when off.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from eidetic.graph import KnowledgeGraph
from eidetic.models import Citation, MemoryRecord, NLILabel, RetrievalCandidate, Scope
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore


class _FakeIndex:
    """Just enough VectorIndex surface for retrieve()'s early-exit path."""

    def __len__(self):
        return 1

    def search(self, *a, **k):
        return []


class _FakeStore:
    def active_records_at(self, at, scope):
        return []          # empty corpus -> retrieve() returns [] right after the embed step


class _RecordingClient:
    def __init__(self, topic="TOPIC"):
        self.embedded = []
        self.topic = topic
        self.topic_calls = 0

    def embed_text(self, text):
        self.embedded.append(text)
        import numpy as np
        return np.zeros(8, dtype="float32")

    def generate_topic(self, query):
        self.topic_calls += 1
        return self.topic


def _retriever(settings, client, store=None, index=None):
    return Retriever(store or _FakeStore(), index or _FakeIndex(),
                     KnowledgeGraph(RecordStore(settings.sqlite_path)), object(), client, settings)


# ---- ACTIVE_RETRIEVAL -------------------------------------------------------------------

def test_active_retrieval_off_embeds_raw_query_no_topic(fresh_settings):
    s = replace(fresh_settings, active_retrieval_enabled=False)
    client = _RecordingClient()
    r = _retriever(s, client)
    r.retrieve("where did Mel move from", scope=Scope())
    assert client.topic_calls == 0
    assert client.embedded == ["where did Mel move from"]   # byte-identical to the legacy path


def test_active_retrieval_on_folds_topic_into_embed(fresh_settings):
    s = replace(fresh_settings, active_retrieval_enabled=True)
    client = _RecordingClient(topic="relocation origin city")
    r = _retriever(s, client)
    r.retrieve("where did Mel move from", scope=Scope())
    assert client.topic_calls == 1
    assert client.embedded == ["where did Mel move from relocation origin city"]


# ---- COVE -------------------------------------------------------------------------------

class _CoveClient:
    """Generates a draft, then plans + answers verification questions."""

    def __init__(self):
        self.calls = []

    def generate_answer(self, query, blocks, model=None):
        self.calls.append(("gen", query))
        return "Mel moved from Sweden."

    def plan_verification_questions(self, draft, n=3):
        return ["Did Mel move from Sweden?"]


def _cove_retriever(settings, sub_entailed):
    """A retriever whose answer() runs with COVE on; _verify_candidates returns entailed=1 for the
    main draft and `sub_entailed` for the factored CoVe check."""
    s = settings
    client = _CoveClient()
    r = _retriever(s, client)
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text="Mel moved from Sweden",
                       scope=Scope(), valid_at=1.0)
    cand = RetrievalCandidate(record=rec, dense_score=0.9, bm25_score=0.6, graph_score=0.4,
                              fused_score=1.0, rerank_score=0.9)
    cit = [Citation(memory_id="m1", content_hash="h1", raw_uri="", source="u", valid_at=1.0,
                    nli_label=NLILabel.ENTAILMENT, nli_score=0.95)]

    state = {"n": 0}

    def fake_verify(cands, text, verify):
        state["n"] += 1
        return (cit, 1) if state["n"] == 1 else (cit, sub_entailed)

    r._verify_candidates = fake_verify
    r._try_conflict_resolver = lambda *a, **k: None
    r.assemble_context = lambda *a, **k: ["[S0] Mel moved from Sweden"]
    return r, [cand]


def test_cove_failed_check_demotes_to_unverified(fresh_settings):
    s = replace(fresh_settings, cove_enabled=True, cove_questions=1, cascade_enabled=False,
                abstention_v2_enabled=False, abstention_threshold=0.0)
    r, cands = _cove_retriever(s, sub_entailed=0)
    ans = r.answer("where did Mel move from", verify=True, precomputed=cands)
    assert ans.verified is False
    assert "CoVe" in ans.note


def test_cove_passed_check_keeps_verified(fresh_settings):
    s = replace(fresh_settings, cove_enabled=True, cove_questions=1, cascade_enabled=False,
                abstention_v2_enabled=False, abstention_threshold=0.0)
    r, cands = _cove_retriever(s, sub_entailed=1)
    ans = r.answer("where did Mel move from", verify=True, precomputed=cands)
    assert ans.verified is True
    assert "CoVe" not in ans.note


def test_cove_off_skips_verification_questions(fresh_settings):
    s = replace(fresh_settings, cove_enabled=False, cascade_enabled=False,
                abstention_v2_enabled=False, abstention_threshold=0.0)
    r, cands = _cove_retriever(s, sub_entailed=0)   # sub_entailed irrelevant: CoVe must not run
    ans = r.answer("where did Mel move from", verify=True, precomputed=cands)
    assert ans.verified is True                      # main draft entailed, no CoVe demotion


def test_retrieval_wiring_flags_default_off(fresh_settings):
    assert fresh_settings.active_retrieval_enabled is False
    assert fresh_settings.cove_enabled is False
