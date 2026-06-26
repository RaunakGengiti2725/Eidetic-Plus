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
    # The stale ENTAILMENT citations MUST be demoted to NEUTRAL: otherwise engine.ask()
    # reconsolidation reinforces the very memories the factored check rejected, and the proof
    # surface ships a verified=False answer carrying ENTAILMENT citations.
    assert ans.citations
    assert all(c.nli_label == NLILabel.NEUTRAL and c.nli_score == 0.0 for c in ans.citations)


def test_cove_passed_check_keeps_verified(fresh_settings):
    s = replace(fresh_settings, cove_enabled=True, cove_questions=1, cascade_enabled=False,
                abstention_v2_enabled=False, abstention_threshold=0.0)
    r, cands = _cove_retriever(s, sub_entailed=1)
    ans = r.answer("where did Mel move from", verify=True, precomputed=cands)
    assert ans.verified is True
    assert "CoVe" not in ans.note
    # Assert CoVe ACTUALLY ran (the positive test must fail if the feature became a no-op): the
    # planned verification question was answered independently.
    assert ("gen", "Did Mel move from Sweden?") in r.client.calls
    # A passing CoVe leaves the entailment citations intact (reconsolidation still reinforces them).
    assert any(c.nli_label == NLILabel.ENTAILMENT for c in ans.citations)


def test_cove_off_skips_verification_questions(fresh_settings):
    s = replace(fresh_settings, cove_enabled=False, cascade_enabled=False,
                abstention_v2_enabled=False, abstention_threshold=0.0)
    r, cands = _cove_retriever(s, sub_entailed=0)   # sub_entailed irrelevant: CoVe must not run
    ans = r.answer("where did Mel move from", verify=True, precomputed=cands)
    assert ans.verified is True                      # main draft entailed, no CoVe demotion


def _span_retriever(settings, claim_entailed_for):
    """A retriever for SPAN_NLI: the whole-answer verify entails (1); per-claim verify returns
    `claim_entailed_for(claim_text)` for each sentence checked."""
    s = settings
    client = _CoveClient()
    r = _retriever(s, client)
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text="Mel reads and runs",
                       scope=Scope(), valid_at=1.0)
    cand = RetrievalCandidate(record=rec, dense_score=0.9, bm25_score=0.6, graph_score=0.4,
                              fused_score=1.0, rerank_score=0.9)
    cit = [Citation(memory_id="m1", content_hash="h1", raw_uri="", source="u", valid_at=1.0,
                    nli_label=NLILabel.ENTAILMENT, nli_score=0.95)]

    state = {"first": True, "calls": 0}

    def fake_verify(cands, text, verify):
        state["calls"] += 1
        if state["first"]:
            state["first"] = False
            return (cit, 1)                     # whole-answer entails
        return (cit, 1 if claim_entailed_for(text) else 0)   # per-claim

    r._verify_candidates = fake_verify
    r._try_conflict_resolver = lambda *a, **k: None
    r.assemble_context = lambda *a, **k: ["[S0] Mel reads and runs"]
    # The reader returns a TWO-sentence answer, one of which is ungrounded.
    client.generate_answer = lambda q, b, model=None: "Mel reads books. Mel pilots jets."
    r._verify_state = state
    return r, [cand]


def test_span_nli_demotes_when_one_claim_ungrounded(fresh_settings):
    s = replace(fresh_settings, span_nli_enabled=True, cove_enabled=False, cascade_enabled=False,
                abstention_v2_enabled=False, abstention_threshold=0.0)
    # "pilots jets" is not grounded -> the whole answer demotes.
    r, cands = _span_retriever(s, claim_entailed_for=lambda t: "reads" in t)
    ans = r.answer("what does Mel do", verify=True, precomputed=cands)
    assert ans.verified is False
    assert "sentence-level claim" in ans.note


def test_span_nli_keeps_verified_when_all_claims_grounded(fresh_settings):
    s = replace(fresh_settings, span_nli_enabled=True, cove_enabled=False, cascade_enabled=False,
                abstention_v2_enabled=False, abstention_threshold=0.0)
    r, cands = _span_retriever(s, claim_entailed_for=lambda t: True)
    ans = r.answer("what does Mel do", verify=True, precomputed=cands)
    assert ans.verified is True
    # The per-claim loop MUST have run (positive test fails if SPAN became a no-op): 1 whole-answer
    # verify + 2 per-claim verifies ("Mel reads books." / "Mel pilots jets.").
    assert r._verify_state["calls"] == 3


def test_span_nli_off_keeps_whole_answer_verdict(fresh_settings):
    s = replace(fresh_settings, span_nli_enabled=False, cove_enabled=False, cascade_enabled=False,
                abstention_v2_enabled=False, abstention_threshold=0.0)
    r, cands = _span_retriever(s, claim_entailed_for=lambda t: False)  # would fail if it ran
    ans = r.answer("what does Mel do", verify=True, precomputed=cands)
    assert ans.verified is True                  # span check never ran -> whole-answer verdict


def test_cove_demotion_into_abstention_overwrites_note(fresh_settings):
    # When a CoVe demotion lands in the low-coverage abstention branch (coverage < threshold), the
    # abstention note takes precedence over the CoVe-specific unverified reason, and the answer
    # abstains. Citations are still demoted to NEUTRAL by the demotion fix.
    s = replace(fresh_settings, cove_enabled=True, cove_questions=1, cascade_enabled=False,
                abstention_v2_enabled=False, abstention_threshold=0.99)  # coverage 0.9 < 0.99
    r, cands = _cove_retriever(s, sub_entailed=0)
    ans = r.answer("where did Mel move from", verify=True, precomputed=cands)
    assert ans.verified is False
    assert ans.note.startswith("abstained")
    assert all(c.nli_label == NLILabel.NEUTRAL for c in ans.citations)


class _RaisingTopicClient(_RecordingClient):
    def generate_topic(self, query):
        self.topic_calls += 1
        raise RuntimeError("topic backend down")


def test_active_retrieval_topic_failure_falls_back_to_raw_query(fresh_settings):
    # The best-effort guard: a failed generate_topic must NOT propagate; retrieve() falls back to
    # embedding the raw query (core recall is never aborted by the optional scaffolding call).
    s = replace(fresh_settings, active_retrieval_enabled=True)
    client = _RaisingTopicClient()
    r = _retriever(s, client)
    r.retrieve("where did Mel move from", scope=Scope())   # must not raise
    assert client.topic_calls == 1
    assert client.embedded == ["where did Mel move from"]   # raw query embedded after fallback


def test_retrieval_wiring_flags_default_off(fresh_settings):
    assert fresh_settings.active_retrieval_enabled is False
    assert fresh_settings.cove_enabled is False
    assert fresh_settings.span_nli_enabled is False
