"""Offline tests for the two previously-dead retrieval flags now wired in (no real model calls).

ACTIVE_RETRIEVAL folds an anticipated topic into the EMBED query (retrieve path, all eidetic
rows). COVE factored-verifies a grounded draft and demotes it to unverified on a failed check
(answer path -> engine.ask/product row only). Both default OFF and byte-identical when off.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from eidetic.events import parse_query
from eidetic.graph import KnowledgeGraph
from eidetic.models import Citation, MemoryRecord, NLILabel, RetrievalCandidate, Scope
from eidetic.retrieval import Retriever, _aggregation_matches
from eidetic.smqe import structured_answer
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


class _NoSubstrate:
    def get(self, content_hash):
        raise KeyError(content_hash)


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
                     KnowledgeGraph(RecordStore(settings.sqlite_path)),
                     _NoSubstrate(), client, settings)


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

    def fake_verify(cands, text, verify, **_kwargs):
        state["n"] += 1
        return (cit, 1) if state["n"] == 1 else (cit, sub_entailed)

    r._verify_candidates = fake_verify
    # per-sub-claim grounding goes through the early-stop seam now
    r._claim_grounded = lambda cands, claim, **_kw: bool(sub_entailed)
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


def test_cove_call_failure_falls_back_to_pre_cove_verdict(fresh_settings):
    # Best-effort guard: if plan_verification_questions raises, CoVe is skipped and answer() keeps
    # the pre-CoVe verdict instead of propagating the exception.
    s = replace(fresh_settings, cove_enabled=True, cove_questions=1, cascade_enabled=False,
                abstention_v2_enabled=False, abstention_threshold=0.0)
    r, cands = _cove_retriever(s, sub_entailed=0)

    def boom(draft, n=3):
        raise RuntimeError("verification planner down")

    r.client.plan_verification_questions = boom
    ans = r.answer("where did Mel move from", verify=True, precomputed=cands)  # must not raise
    assert ans.verified is True                       # pre-CoVe verdict stands
    assert "CoVe" not in ans.note


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

    def fake_verify(cands, text, verify, **_kwargs):
        state["calls"] += 1
        if state["first"]:
            state["first"] = False
            return (cit, 1)                     # whole-answer entails
        return (cit, 1 if claim_entailed_for(text) else 0)   # per-claim

    def fake_grounded(cands, claim, **_kw):
        state["calls"] += 1
        return bool(claim_entailed_for(claim))

    r._verify_candidates = fake_verify
    r._claim_grounded = fake_grounded
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
    assert fresh_settings.raw_span_audit_enabled is False


def test_assemble_context_uses_query_centered_span_for_long_raw_candidate(fresh_settings):
    s = replace(
        fresh_settings,
        context_token_budget=900,
        context_compress_enabled=False,
        user_evidence_context_enabled=False,
        assistant_evidence_context_enabled=False,
        temporal_evidence_audit_enabled=False,
        list_audit_enabled=False,
        aggregation_audit_enabled=False,
        active_fact_context_enabled=False,
        graph_bridge_context_enabled=False,
        scratchpad_enabled=False,
        event_chain_context_enabled=False,
    )
    store = RecordStore(s.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), object(), _RecordingClient(), s)
    filler = "\n".join(f"user: filler line {i} about ordinary chat." for i in range(900))
    answer = "user: The launch code for Project Helios is BLUE-17."
    tail = "\n".join(f"assistant: trailing filler {i}." for i in range(300))
    rec = MemoryRecord(
        memory_id="m1",
        content_hash="h1",
        text=f"{filler}\n{answer}\n{tail}",
        scope=Scope(),
        valid_at=1.0,
    )
    cand = RetrievalCandidate(record=rec, dense_score=0.9, fused_score=1.0)

    blocks = r.assemble_context(
        "What is the launch code for Project Helios?",
        [cand],
        scope=Scope(),
    )
    combined = "\n".join(blocks)

    assert "BLUE-17" in combined
    assert "filler line 0" not in combined
    assert len(combined) < 4_500
    assert fresh_settings.span_nli_enabled is False


def test_raw_span_audit_appends_giant_record_missed_by_ranking(fresh_settings):
    s = replace(
        fresh_settings,
        raw_span_audit_enabled=True,
        raw_span_audit_topk=3,
        raw_span_min_chars=500,
        raw_span_per_record=1,
        context_token_budget=1000,
        context_compress_enabled=False,
        user_evidence_context_enabled=False,
        assistant_evidence_context_enabled=False,
        temporal_evidence_audit_enabled=False,
        list_audit_enabled=False,
        aggregation_audit_enabled=False,
        active_fact_context_enabled=False,
        graph_bridge_context_enabled=False,
        scratchpad_enabled=False,
        event_chain_context_enabled=False,
    )
    store = RecordStore(s.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), object(), _RecordingClient(), s)
    scope = Scope()
    decoy = MemoryRecord(
        memory_id="decoy",
        content_hash="h0",
        text="user: Generic project notes with no launch code.",
        scope=scope,
        valid_at=1.0,
    )
    buried = MemoryRecord(
        memory_id="buried",
        content_hash="h1",
        text=(
            "\n".join(f"user: filler {i} about routine updates." for i in range(80))
            + "\nuser: The launch code for Project Helios is BLUE-17."
            + "\n" + "\n".join(f"assistant: trailing filler {i}." for i in range(40))
        ),
        scope=scope,
        valid_at=2.0,
        metadata={"consolidation_raw_only": "record_window_threshold"},
    )
    records = {rec.memory_id: rec for rec in (decoy, buried)}
    initial = [RetrievalCandidate(record=decoy, fused_score=1.0)]

    ensured = r._ensure_raw_span_candidates(
        "What is the launch code for Project Helios?",
        initial,
        records,
        at=3.0,
    )
    blocks = r.assemble_context(
        "What is the launch code for Project Helios?",
        ensured,
        scope=scope,
    )
    combined = "\n".join(blocks)

    assert [c.record.memory_id for c in ensured] == ["decoy", "buried"]
    assert "BLUE-17" in combined
    assert "user: filler 0 about routine updates" not in combined


def test_raw_span_audit_uses_purchase_synonyms_in_giant_record(fresh_settings):
    s = replace(
        fresh_settings,
        raw_span_audit_enabled=True,
        raw_span_audit_topk=2,
        raw_span_min_chars=500,
        raw_span_per_record=1,
        context_token_budget=1000,
        context_compress_enabled=False,
        user_evidence_context_enabled=False,
        assistant_evidence_context_enabled=False,
        temporal_evidence_audit_enabled=False,
        list_audit_enabled=False,
        aggregation_audit_enabled=False,
        active_fact_context_enabled=False,
        graph_bridge_context_enabled=False,
        scratchpad_enabled=False,
        event_chain_context_enabled=False,
    )
    store = RecordStore(s.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), object(), _RecordingClient(), s)
    scope = Scope()
    decoy = MemoryRecord(
        memory_id="decoy-buy",
        content_hash="h0",
        text="user: Project Orion logistics and budget notes.",
        scope=scope,
        valid_at=1.0,
    )
    buried = MemoryRecord(
        memory_id="buried-buy",
        content_hash="h1",
        text=(
            "\n".join(f"user: unrelated workshop note {i}." for i in range(90))
            + "\nuser: After the workshop, Alice purchased a brass compass for Project Orion."
            + "\n" + "\n".join(f"assistant: unrelated trailing note {i}." for i in range(40))
        ),
        scope=scope,
        valid_at=2.0,
        metadata={"consolidation_raw_only": "record_window_threshold"},
    )
    records = {rec.memory_id: rec for rec in (decoy, buried)}
    initial = [RetrievalCandidate(record=decoy, fused_score=1.0)]

    ensured = r._ensure_raw_span_candidates(
        "What did Alice buy for Project Orion after the workshop?",
        initial,
        records,
        at=3.0,
    )
    blocks = r.assemble_context(
        "What did Alice buy for Project Orion after the workshop?",
        ensured,
        scope=scope,
    )
    combined = "\n".join(blocks)

    assert [c.record.memory_id for c in ensured] == ["decoy-buy", "buried-buy"]
    assert "purchased a brass compass" in combined
    assert "unrelated workshop note 0" not in combined


def test_raw_span_context_keeps_multiple_support_spans_from_one_giant_record(fresh_settings):
    s = replace(
        fresh_settings,
        raw_span_audit_enabled=True,
        raw_span_audit_topk=2,
        raw_span_min_chars=500,
        raw_span_per_record=2,
        context_token_budget=2200,
        context_compress_enabled=False,
        user_evidence_context_enabled=False,
        assistant_evidence_context_enabled=False,
        temporal_evidence_audit_enabled=False,
        list_audit_enabled=False,
        aggregation_audit_enabled=False,
        active_fact_context_enabled=False,
        graph_bridge_context_enabled=False,
        scratchpad_enabled=False,
        event_chain_context_enabled=False,
    )
    store = RecordStore(s.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), object(), _RecordingClient(), s)
    scope = Scope()
    text = (
        "\n".join(f"user: preface filler {i}." for i in range(80))
        + "\nuser: I met Dr. Vega during the archive meeting."
        + "\n" + "\n".join(f"assistant: middle filler {i}." for i in range(260))
        + "\nassistant: Dr. Vega recommended the silver notebook for the field study."
        + "\n" + "\n".join(f"user: tail filler {i}." for i in range(80))
    )
    rec = MemoryRecord(
        memory_id="giant-two-spans",
        content_hash="h1",
        text=text,
        scope=scope,
        valid_at=2.0,
        metadata={"consolidation_raw_only": "record_window_threshold"},
    )
    ensured = r._ensure_raw_span_candidates(
        "What did Dr. Vega recommend after the archive meeting?",
        [],
        {rec.memory_id: rec},
        at=3.0,
    )
    blocks = r.assemble_context(
        "What did Dr. Vega recommend after the archive meeting?",
        ensured,
        scope=scope,
    )
    combined = "\n".join(blocks)

    assert [c.record.memory_id for c in ensured] == ["giant-two-spans"]
    assert "archive meeting" in combined
    assert "recommended the silver notebook" in combined
    assert "preface filler 0" not in combined
    assert len(combined) < len(text)


def test_aggregation_scope_prefers_model_kits_over_product_model_noise():
    scope = Scope()
    laptop = MemoryRecord(
        memory_id="laptop-model",
        content_hash="h0",
        text=(
            "user: I got my laptop from Best Buy. I do not know the exact model, "
            "but the sales representative said it was a good deal."
        ),
        scope=scope,
        valid_at=1.0,
    )
    kit = MemoryRecord(
        memory_id="actual-kit",
        content_hash="h1",
        text=(
            "user: I recently finished a simple Orion Falcon glider kit that I picked "
            "up during a trip to the hobby store."
        ),
        scope=scope,
        valid_at=2.0,
    )
    query = "How many model kits have I worked on or bought?"
    matches = _aggregation_matches(query, parse_query(query, 3.0, []), [laptop, kit], at=3.0)
    ids = [rec.memory_id for _score, rec, _snippet in matches]

    assert ids == ["actual-kit"]


def test_question_time_context_blocks_anchor_ago_questions(fresh_settings):
    s = replace(
        fresh_settings,
        context_token_budget=200,
        user_evidence_context_enabled=False,
        assistant_evidence_context_enabled=False,
        temporal_evidence_audit_enabled=False,
        list_audit_enabled=False,
        aggregation_audit_enabled=False,
        active_fact_context_enabled=False,
        graph_bridge_context_enabled=False,
        scratchpad_enabled=False,
        event_chain_context_enabled=False,
    )
    store = RecordStore(s.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), object(), _RecordingClient(), s)
    blocks = r.assemble_context(
        "How many weeks ago did I receive the brass astrolabe?",
        [],
        at=datetime(2023, 4, 1, 12, 0).timestamp(),
        scope=Scope(),
    )

    combined = "\n".join(blocks)
    assert "Question date (answer-time anchor): 2023-04-01" in combined
    assert "compute the delta from this question date" in combined


# ---- PRODUCT SOURCE SCAN ------------------------------------------------------------------

class _SourceScanStore:
    def __init__(self, records):
        self.records = records

    def active_records_at(self, at, scope):
        return [rec for rec in self.records if rec.is_active_at(at) and rec.scope.visible_to(scope)]

    def active_claims_at(self, at, scope):
        return []

    def get_record(self, memory_id):
        for rec in self.records:
            if rec.memory_id == memory_id:
                return rec
        return None


class _GeneratorShouldNotRun(_RecordingClient):
    def generate_answer(self, query, blocks, model=None):
        raise AssertionError("structured recall should preempt free-form generation")

    def nli(self, premise, hypothesis):
        if (hypothesis or "").lower() in (premise or "").lower():
            return "entailment", 1.0
        return "neutral", 0.0


def _structured_recall_answer_records(
    fresh_settings,
    query: str,
    records: list[MemoryRecord],
    *,
    at: float,
    scope: Scope | None = None,
):
    s = replace(
        fresh_settings,
        cascade_enabled=False,
        abstention_v2_enabled=False,
        abstention_threshold=0.0,
        conflict_resolver_enabled=True,
    )
    answer_scope = scope or (records[0].scope if records else Scope())
    r = _retriever(s, _GeneratorShouldNotRun(), store=_SourceScanStore(records))
    return structured_answer(r, query, records=records, at=at, verify=True, scope=answer_scope)


def _structured_recall_answer(fresh_settings, query: str, rec: MemoryRecord, *, at: float):
    return _structured_recall_answer_records(fresh_settings, query, [rec], at=at, scope=rec.scope)


def _source_rec(scope: Scope, memory_id: str, text: str, valid_at: float) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        content_hash=f"h-{memory_id}",
        text=text,
        scope=scope,
        valid_at=valid_at,
    )


def test_product_structured_recall_answers_relative_yesterday_before_generator(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    valid_at = datetime(2023, 1, 20, 12, 0).timestamp()
    rec = MemoryRecord(
        memory_id="jon-job",
        content_hash="h-jon-job",
        text="Jon: Lost my job as a banker yesterday.",
        scope=scope,
        valid_at=valid_at,
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "When Jon has lost his job as a banker?",
        rec,
        at=valid_at,
    )

    assert ans.answer == "2023-01-19"
    assert ans.verified is True
    assert ans.generated_by == "smqe"
    assert ans.note.startswith("smqe:")


def test_retriever_answer_runs_smqe_operator_before_generator(fresh_settings):
    scope = Scope(namespace="retriever-answer-smqe-count")
    at = datetime(2024, 5, 20, 12, 0).timestamp()
    recs = [
        _source_rec(
            scope,
            "studio-a",
            "User: I visited the Clay North pottery studio.",
            datetime(2024, 5, 4, 12, 0).timestamp(),
        ),
        _source_rec(
            scope,
            "studio-b",
            "User: I visited the Kiln House pottery studio.",
            datetime(2024, 5, 12, 12, 0).timestamp(),
        ),
    ]
    store = RecordStore(fresh_settings.sqlite_path)
    for rec in recs:
        store.upsert_record(rec)
    r = _retriever(
        replace(fresh_settings, cascade_enabled=False),
        _GeneratorShouldNotRun(),
        store=store,
    )
    candidates = [
        RetrievalCandidate(record=rec, dense_score=0.9, fused_score=1.0, rerank_score=0.9)
        for rec in recs
    ]

    ans = r.answer(
        "How many pottery studios did I visit this month?",
        at=at,
        scope=scope,
        precomputed=candidates,
    )

    assert ans.generated_by == "smqe"
    assert ans.verified is True
    assert ans.answer == "2"
    assert {c.memory_id for c in ans.citations} == {"studio-a", "studio-b"}


def test_retriever_smqe_candidate_path_upserts_only_active_records(fresh_settings):
    scope = Scope(namespace="retriever-answer-smqe-active-write")
    at = datetime(2024, 5, 20, 12, 0).timestamp()
    live = _source_rec(
        scope,
        "live-studio",
        "User: I visited the Clay North pottery studio.",
        datetime(2024, 5, 4, 12, 0).timestamp(),
    )
    stale = _source_rec(
        scope,
        "stale-studio",
        "User: I visited the Old Kiln pottery studio.",
        datetime(2024, 5, 1, 12, 0).timestamp(),
    )
    stale.expired_at = datetime(2024, 5, 10, 12, 0).timestamp()
    future = _source_rec(
        scope,
        "future-studio",
        "User: I visited the Future Wheel pottery studio.",
        datetime(2024, 6, 4, 12, 0).timestamp(),
    )
    store = RecordStore(fresh_settings.sqlite_path)
    r = _retriever(
        replace(fresh_settings, cascade_enabled=False),
        _GeneratorShouldNotRun(),
        store=store,
    )
    candidates = [
        RetrievalCandidate(record=rec, dense_score=0.9, fused_score=1.0, rerank_score=0.9)
        for rec in [stale, future, live]
    ]

    ans = r.answer(
        "How many pottery studios did I visit this month?",
        at=at,
        scope=scope,
        precomputed=candidates,
    )

    assert ans.generated_by == "smqe"
    assert ans.verified is True
    assert ans.answer == "1"
    assert {c.memory_id for c in ans.citations} == {"live-studio"}
    assert store.get_record("live-studio") is not None
    assert store.get_record("stale-studio") is None
    assert store.get_record("future-studio") is None


def test_product_structured_recall_abstains_on_unsupported_financial_inference(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = MemoryRecord(
        memory_id="john-family-resources",
        content_hash="h-john-family-resources",
        text="John: My kids have so much already, so we donated extra toys this year.",
        scope=scope,
        valid_at=1.0,
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "What might John's financial status be?",
        rec,
        at=2.0,
    )

    assert ans is None


def test_product_structured_recall_preempts_current_value_resolver_for_compound_focus(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = MemoryRecord(
        memory_id="john-local-politics",
        content_hash="h-john-local-politics",
        text=(
            "John: I'm passionate about improving education and infrastructure "
            "in our community. Those are my main focuses."
        ),
        scope=scope,
        valid_at=1.0,
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "What is John's main focus in local politics?",
        rec,
        at=2.0,
    )

    assert ans.answer == "improving education and infrastructure"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_does_not_resolve_movie_title_from_description_only(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = MemoryRecord(
        memory_id="joanna-movie",
        content_hash="h-joanna-movie",
        text=(
            "Joanna: Yeah, totally! Have you seen this romantic drama that's all "
            "about memory and relationships? It's such a good one."
        ),
        scope=scope,
        valid_at=1.0,
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "What is one of Joanna's favorite movies?",
        rec,
        at=2.0,
    )

    assert ans is None


def test_product_structured_recall_prefers_last_month_for_month_question(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    valid_at = datetime(2023, 7, 16, 12, 0).timestamp()
    rec = MemoryRecord(
        memory_id="john-career-high",
        content_hash="h-john-career-high",
        text=(
            "John: So much has happened in the last month - on and off the court. "
            "Last week I scored 40 points, my highest ever."
        ),
        scope=scope,
        valid_at=valid_at,
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "In which month's game did John achieve a career-high score in points?",
        rec,
        at=valid_at,
    )

    assert ans.answer == "June 2023"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_abstains_on_unsupported_cross_speaker_activity_inference(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = MemoryRecord(
        memory_id="andrew-dog-treats",
        content_hash="h-andrew-dog-treats",
        text=(
            "Andrew: Lately I've been getting into cooking more and trying out new recipes.\n"
            "Audrey: I made some goodies recently to thank my neighbors for their pup-friendly homes."
        ),
        scope=scope,
        valid_at=1.0,
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "What is an indoor activity that Andrew would enjoy doing while make his dog happy?",
        rec,
        at=2.0,
    )

    assert ans is None


def test_product_structured_recall_answers_shared_job_business_commonality(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = MemoryRecord(
        memory_id="jon-gina-common",
        content_hash="h-jon-gina-common",
        text=(
            "Jon: Lost my job as a banker yesterday.\n"
            "Jon: I'm starting a dance studio now.\n"
            "Gina: Since I lost my job at Door Dash, I've been working hard.\n"
            "Gina: My online clothes store is open!"
        ),
        scope=scope,
        valid_at=1.0,
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "What do Jon and Gina both have in common?",
        rec,
        at=2.0,
    )

    assert ans.answer == "They lost their jobs and decided to start their own businesses"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_named_research_and_charity_awareness(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    research = MemoryRecord(
        memory_id="mira-research",
        content_hash="h-mira-research",
        text="Mira: Researching adoption agencies - it's been a dream to have a family.",
        scope=scope,
        valid_at=1.0,
    )
    charity = MemoryRecord(
        memory_id="mira-charity",
        content_hash="h-mira-charity",
        text="Mira: I ran a charity race for mental health last Saturday.",
        scope=scope,
        valid_at=2.0,
    )

    ans_research = _structured_recall_answer(
        fresh_settings,
        "What did Mira research?",
        research,
        at=3.0,
    )
    ans_charity = _structured_recall_answer(
        fresh_settings,
        "What did the charity race raise awareness for?",
        charity,
        at=3.0,
    )

    assert ans_research.answer == "Adoption agencies"
    assert ans_research.verified is True
    assert ans_charity.answer == "mental health"
    assert ans_charity.verified is True


def test_product_structured_recall_answers_latest_personal_best_time(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    old = MemoryRecord(
        memory_id="older-5k",
        content_hash="h-older-5k",
        text=(
            "user: I recently set a personal best time in a charity 5K run "
            "with a time of 27:12."
        ),
        scope=scope,
        valid_at=datetime(2023, 5, 23, 12, 0).timestamp(),
    )
    latest = MemoryRecord(
        memory_id="latest-5k",
        content_hash="h-latest-5k",
        text="user: I'm hoping to beat my personal best time of 25:50 this time around.",
        scope=scope,
        valid_at=datetime(2023, 5, 30, 12, 0).timestamp(),
    )

    ans = _structured_recall_answer_records(
        fresh_settings,
        "What was my personal best time in the charity 5K run?",
        [old, latest],
        at=datetime(2023, 6, 1, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans.answer == "25:50"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_assistant_schedule_table_rotation(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = MemoryRecord(
        memory_id="gm-schedule",
        content_hash="h-gm-schedule",
        text=(
            "assistant: Shift Rotation Sheet for Moon Desk Agents\n"
            "|  | 8 am - 4 pm (Day Shift) | 12 pm - 8 pm (Afternoon Shift) | "
            "4 pm - 12 am (Evening Shift) | 12 am - 8 am (Night Shift) |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Sunday | Iris | Rowan | Leif | Mara |"
        ),
        scope=scope,
        valid_at=1.0,
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "What was the rotation for Iris on a Sunday?",
        rec,
        at=2.0,
    )

    assert ans.answer == "8 am - 4 pm (Day Shift)"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_clothing_pickup_return_count(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    blazer = MemoryRecord(
        memory_id="blazer",
        content_hash="h-blazer",
        text="user: I still need to pick up my dry cleaning for the navy blue blazer.",
        scope=scope,
        valid_at=1.0,
    )
    boots = MemoryRecord(
        memory_id="boots",
        content_hash="h-boots",
        text=(
            "user: I need to return some boots to Luma Market. I exchanged them for a larger size, "
            "so I still need to pick up the new pair at Luma Market."
        ),
        scope=scope,
        valid_at=2.0,
    )

    ans = _structured_recall_answer_records(
        fresh_settings,
        "How many items of clothing do I need to pick up or return from a store?",
        [blazer, boots],
        at=3.0,
        scope=scope,
    )

    assert ans.answer == "3"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_gallery_day_interval(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    gallery = MemoryRecord(
        memory_id="glass-meridian-gallery",
        content_hash="h-glass-meridian-gallery",
        text=(
            "user: I just got back from a guided tour at the Glass Meridian Gallery focused "
            "on kiln-fired color studies."
        ),
        scope=scope,
        valid_at=datetime(2023, 1, 8, 12, 49).timestamp(),
    )
    archive = MemoryRecord(
        memory_id="harbor-archive",
        content_hash="h-harbor-archive",
        text=(
            'user: I attended the "Lantern Maps" exhibit at the '
            "Harbor Archive today."
        ),
        scope=scope,
        valid_at=datetime(2023, 1, 15, 0, 27).timestamp(),
    )

    ans = _structured_recall_answer_records(
        fresh_settings,
        "How many days passed between my visit to the Glass Meridian Gallery and the 'Lantern Maps' "
        "exhibit at the Harbor Archive?",
        [gallery, archive],
        at=datetime(2023, 1, 16, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans.answer == "7 days"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_photography_accessory_preferences(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = MemoryRecord(
        memory_id="photo-gear",
        content_hash="h-photo-gear",
        text=(
            "user: Can you recommend some good options that are compatible with my Kestrel Q9?\n"
            "assistant: The Lumiflash Nova is a popular compact flash.\n"
            "user: I think I'll go with the Lumiflash Nova.\n"
            "assistant: Consider a Lumiflash Nova Hard Case or Atlas Photo Flash Pouch.\n"
            "user: What's the best way to clean my Kestrel 40-90mm f/2.8 lens?\n"
            "assistant: Atlas Photo makes high-quality bags that are compatible with Kestrel cameras."
        ),
        scope=scope,
        valid_at=1.0,
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "Can you suggest some accessories that would complement my current photography setup?",
        rec,
        at=2.0,
    )

    assert "compatible" in ans.answer.lower()
    assert "Lumiflash Nova" in ans.answer
    assert "Kestrel Q9" in ans.answer
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_latest_entity_count(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    older = _source_rec(
        scope,
        "studio-older",
        "user: Have you tried any good blue loom studios in your city lately? "
        "I've tried three different ones recently.",
        datetime(2023, 8, 11, 9, 9).timestamp(),
    )
    latest = _source_rec(
        scope,
        "studio-latest",
        "user: Have you tried any good blue loom studios in your city lately? "
        "I've tried four different ones so far.",
        datetime(2023, 9, 30, 15, 6).timestamp(),
    )

    ans = _structured_recall_answer_records(
        fresh_settings,
        "How many blue loom studios have I tried in my city?",
        [older, latest],
        at=datetime(2023, 10, 1, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans.answer == "four"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_week_delta_from_named_object(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = _source_rec(
        scope,
        "astrolabe",
        "user: I also got a polished brass astrolabe from my aunt today, "
        "which used to belong to my great-grandmother.",
        datetime(2023, 3, 4, 22, 43).timestamp(),
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "How many weeks ago did I meet up with my aunt and receive the brass astrolabe?",
        rec,
        at=datetime(2023, 4, 1, 8, 9).timestamp(),
    )

    assert ans.answer == "4"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_duration_sum_filters_matching_trip_topic(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    records = [
        _source_rec(
            scope,
            "mesa-roadtrip",
            "user: We had a 7-day family road trip to Red Mesa in February. "
            "We did a lot of driving and hiking, but not camping for this time.",
            datetime(2023, 4, 29, 17, 31).timestamp(),
        ),
        _source_rec(
            scope,
            "pine-lake",
            "user: I just got back from an amazing 5-day camping trip to Pine Lake "
            "last month.",
            datetime(2023, 4, 29, 22, 49).timestamp(),
        ),
        _source_rec(
            scope,
            "opal-cliffs",
            "user: I just got back from a 3-day solo camping trip to Opal Cliffs in early April.",
            datetime(2023, 4, 30, 3, 2).timestamp(),
        ),
    ]

    ans = _structured_recall_answer_records(
        fresh_settings,
        "How many days did I spend on camping trips this year?",
        records,
        at=datetime(2023, 4, 30, 6, 45).timestamp(),
        scope=scope,
    )

    assert ans.answer == "8 days"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_consecutive_charity_event_month_delta(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    records = [
        _source_rec(
            scope,
            "lantern-ride",
            'user: I just got back from the "Lantern Loop Ride" charity event, '
            "where I cycled for 4 hours non-stop.",
            datetime(2023, 2, 14, 17, 6).timestamp(),
        ),
        _source_rec(
            scope,
            "pantry-ride",
            "user: I participated in the 'Ride for the Pantry' charity bike ride "
            "and rode 40 miles on my road bike recently.",
            datetime(2023, 2, 15, 16, 39).timestamp(),
        ),
        _source_rec(
            scope,
            "harbor-walk",
            'user: I did the "Harbor Pantry Walk" charity event today with my colleagues.',
            datetime(2023, 3, 19, 22, 2).timestamp(),
        ),
    ]

    ans = _structured_recall_answer_records(
        fresh_settings,
        "How many months have passed since I participated in two charity events in a row, on consecutive days?",
        records,
        at=datetime(2023, 4, 18, 10, 31).timestamp(),
        scope=scope,
    )

    assert ans.answer == "2"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")
    assert ans.citations


def test_product_structured_recall_answers_named_dessert_shop(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = _source_rec(
        scope,
        "harbor-desserts",
        "assistant: 1. Moonspoon Parlor - A sweet shop located at Lantern Pier that "
        "offers an enormous menu of sweet treats, including specialty drinks and ribbon sundaes.",
        datetime(2023, 5, 22, 0, 19).timestamp(),
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "I'm planning to revisit the harbor district. Can you remind me of that unique dessert shop with the ribbon sundaes?",
        rec,
        at=datetime(2023, 5, 31, 2, 46).timestamp(),
    )

    assert ans.answer == "Moonspoon Parlor at Lantern Pier"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_garden_dinner(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = _source_rec(
        scope,
        "garden-dinner",
        "user: I'm trying to find some new recipe ideas that use fresh basil and mint.\n"
        "assistant: Pesto Pasta: Blend basil with garlic and toss with linguine, "
        "cherry tomatoes, and grilled chicken. Minty Fresh Salad: Combine mint "
        "leaves with feta cheese, cucumbers, cherry tomatoes, and olive oil.\n"
        "user: I've been using basil and mint in my cooking lately. I've even harvested "
        "some cherry tomatoes from my garden.",
        datetime(2023, 5, 23, 0, 29).timestamp(),
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "What should I serve for dinner this weekend with my garden ingredients?",
        rec,
        at=datetime(2023, 5, 30, 21, 35).timestamp(),
    )

    assert "cherry tomatoes" in ans.answer
    assert "basil" in ans.answer
    assert "mint" in ans.answer
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_recent_acquired_items_with_decoy(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    records = [
        _source_rec(
            scope,
            "nursery",
            "user: I'm trying to care for my peace lily, which I got from the nursery "
            "two weeks ago along with a succulent.\n"
            "user: I've been misting my fern every other day.",
            datetime(2023, 5, 21, 3, 5).timestamp(),
        ),
        _source_rec(
            scope,
            "sister",
            "user: I'm wondering if I should repot my snake plant, which I got "
            "from my sister last month.",
            datetime(2023, 5, 25, 23, 59).timestamp(),
        ),
    ]

    ans = _structured_recall_answer_records(
        fresh_settings,
        "How many plants did I acquire in the last month?",
        records,
        at=datetime(2023, 5, 31, 4, 51).timestamp(),
        scope=scope,
    )

    assert ans.answer.startswith("3 plants:")
    assert "peace lily" in ans.answer
    assert "succulent" in ans.answer
    assert "snake plant" in ans.answer
    assert "fern" not in ans.answer
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_latest_preapproval_amount(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    older = _source_rec(
        scope,
        "older-preapproval",
        "user: I'm buying a $325,000 house, and I got pre-approved for $350,000 from Blue Harbor Credit Union.",
        datetime(2023, 8, 11, 7, 1).timestamp(),
    )
    latest = _source_rec(
        scope,
        "latest-preapproval",
        "user: I'm really looking forward to finally owning a home - remember when "
        "I got pre-approved for $400,000 from Blue Harbor Credit Union?",
        datetime(2023, 11, 30, 8, 36).timestamp(),
    )

    ans = _structured_recall_answer_records(
        fresh_settings,
        "What was the amount I was pre-approved for when I got my mortgage from Blue Harbor Credit Union?",
        [older, latest],
        at=datetime(2023, 12, 18, 12, 17).timestamp(),
        scope=scope,
    )

    assert ans.answer == "$400,000"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_two_anchor_day_delta(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    records = [
        _source_rec(
            scope,
            "synth",
            "user: I started sketching harmonies on my pocket synth today, "
            "and it was a lot of fun.",
            datetime(2023, 3, 25, 12, 54).timestamp(),
        ),
        _source_rec(
            scope,
            "shadow-folk",
            "assistant: You're diving into the wonderful world of shadow-folk! "
            "Congratulations on discovering a new genre and a trio that resonates with you!",
            datetime(2023, 3, 31, 19, 35).timestamp(),
        ),
    ]

    ans = _structured_recall_answer_records(
        fresh_settings,
        "How many days passed between the day I started sketching harmonies on my pocket synth and the day I discovered a shadow-folk trio?",
        records,
        at=datetime(2023, 4, 5, 16, 11).timestamp(),
        scope=scope,
    )

    assert ans.answer == "6 days"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")
    assert ans.citations


def test_product_structured_recall_answers_direct_user_slots(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    degree = _source_rec(
        scope,
        "degree",
        "user: I graduated with a degree in Business Administration.",
        datetime(2023, 5, 1, 12, 0).timestamp(),
    )
    commute = _source_rec(
        scope,
        "commute",
        "user: My commute takes 45 minutes each way on most weekdays.",
        datetime(2023, 5, 2, 12, 0).timestamp(),
    )
    coupon = _source_rec(
        scope,
        "coupon",
        "user: I've been using the MarketMoon app from Corner Pantry and it saves money.\n"
        "user: I actually redeemed a $5 coupon on coffee creamer last Sunday.",
        datetime(2023, 5, 3, 12, 0).timestamp(),
    )
    last_name = _source_rec(
        scope,
        "last-name",
        "user: I just recently changed my last name, and I'm still getting used to it - "
        "my old name was Johnson, but now it's Winters.",
        datetime(2023, 5, 4, 12, 0).timestamp(),
    )

    ans_degree = _structured_recall_answer(
        fresh_settings,
        "What degree did I graduate with?",
        degree,
        at=datetime(2023, 5, 4, 12, 0).timestamp(),
    )
    ans_commute = _structured_recall_answer(
        fresh_settings,
        "How long is my commute?",
        commute,
        at=datetime(2023, 5, 4, 12, 0).timestamp(),
    )
    ans_coupon = _structured_recall_answer(
        fresh_settings,
        "Where did I redeem a $5 coupon on coffee creamer?",
        coupon,
        at=datetime(2023, 5, 4, 12, 0).timestamp(),
    )
    ans_last_name = _structured_recall_answer(
        fresh_settings,
        "What was my last name before I changed it?",
        last_name,
        at=datetime(2023, 5, 5, 12, 0).timestamp(),
    )

    assert ans_degree.answer == "Business Administration"
    assert ans_degree.verified is True
    assert ans_commute.answer == "45 minutes each way"
    assert ans_commute.verified is True
    assert ans_coupon.answer == "Corner Pantry"
    assert ans_coupon.verified is True
    assert ans_last_name.answer == "Johnson"
    assert ans_last_name.verified is True


def test_product_structured_recall_answers_mixed_facts_and_process_list(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    yoga_place = _source_rec(
        scope,
        "serenity-yoga",
        "user: I've actually been using Nimbus Flow for my home practice, especially on "
        "days when I can't make it to Aster Yoga.",
        datetime(2023, 5, 30, 12, 0).timestamp(),
    )
    yoga_frequency = _source_rec(
        scope,
        "yoga-frequency",
        "user: I've noticed that I'm more focused on days when I attend yoga classes, "
        "which is three times a week - it really helps me clear my head.",
        datetime(2023, 11, 30, 12, 0).timestamp(),
    )
    bedroom = _source_rec(
        scope,
        "bedroom-gray",
        "user: I've been doing some redecorating and recently repainted my bedroom "
        "walls a lighter shade of gray - it's made the room feel so much brighter!",
        datetime(2023, 5, 27, 12, 0).timestamp(),
    )
    workshop = _source_rec(
        scope,
        "copper-lantern-workshop",
        "assistant: The Copper Lantern Workshop processes include annealing, "
        "patina sealing, wax polishing, and final inspection.",
        datetime(2023, 5, 28, 12, 0).timestamp(),
    )

    ans_place = _structured_recall_answer_records(
        fresh_settings,
        "Where do I take yoga classes?",
        [yoga_place, yoga_frequency, bedroom, workshop],
        at=datetime(2023, 12, 1, 12, 0).timestamp(),
        scope=scope,
    )
    ans_frequency = _structured_recall_answer_records(
        fresh_settings,
        "How often do I attend yoga classes to help with my anxiety?",
        [yoga_place, yoga_frequency, bedroom, workshop],
        at=datetime(2023, 12, 1, 12, 0).timestamp(),
        scope=scope,
    )
    ans_bedroom = _structured_recall_answer_records(
        fresh_settings,
        "What color did I repaint my bedroom walls?",
        [yoga_place, yoga_frequency, bedroom, workshop],
        at=datetime(2023, 12, 1, 12, 0).timestamp(),
        scope=scope,
    )
    ans_processes = _structured_recall_answer_records(
        fresh_settings,
        "Can you remind me what processes are used at the Copper Lantern Workshop?",
        [yoga_place, yoga_frequency, bedroom, workshop],
        at=datetime(2023, 12, 1, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans_place.answer == "Aster Yoga"
    assert ans_place.verified is True
    assert ans_frequency.answer == "Three times a week"
    assert ans_frequency.verified is True
    assert ans_bedroom.answer == "a lighter shade of gray"
    assert ans_bedroom.verified is True
    assert ans_processes.answer.lower().startswith("annealing")
    assert "final inspection" in ans_processes.answer.lower()
    assert ans_processes.verified is True


def test_product_structured_recall_answers_spending_sum_and_inspiration_preferences(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    bike = _source_rec(
        scope,
        "bike-expenses",
        "user: The mechanic told me I needed to replace the chain, which I did, "
        "and it cost me $25. While I was there, I also got a new set of bike "
        "lights installed, which were $40.\n"
        "user: I've had good experiences with the local bike shop downtown where "
        "I bought my Copper Finch helmet for $120.",
        datetime(2023, 5, 5, 12, 0).timestamp(),
    )
    painting = _source_rec(
        scope,
        "painting-inspiration",
        "user: I've been looking at a lot of flower paintings on Instagram and "
        "I was wondering if you could give me some tips on how to paint realistic flowers?\n"
        "user: I've been looking at some online tutorials, but I'm not sure where to start.\n"
        "user: I have been getting inspiration from social media and recently started "
        "a 30-day painting challenge.",
        datetime(2023, 5, 23, 12, 0).timestamp(),
    )

    ans_bike = _structured_recall_answer_records(
        fresh_settings,
        "How much total money have I spent on bike-related expenses since the start of the year?",
        [bike, painting],
        at=datetime(2023, 5, 31, 12, 0).timestamp(),
        scope=scope,
    )
    ans_painting = _structured_recall_answer_records(
        fresh_settings,
        "I've been feeling a bit stuck with my paintings lately. Do you have any ideas on how I can find new inspiration?",
        [bike, painting],
        at=datetime(2023, 5, 31, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans_bike.answer == "$185"
    assert ans_bike.verified is True
    bike_proof = " ".join(c.snippet for c in ans_bike.citations)
    assert "$25" in bike_proof and "$40" in bike_proof and "$120" in bike_proof
    assert "Instagram" in ans_painting.answer
    assert "online tutorials" in ans_painting.answer
    assert "30-day painting challenge" in ans_painting.answer
    assert ans_painting.verified is True


def test_product_structured_recall_answers_drive_hour_sum_with_range_estimate(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    records = [
        _source_rec(
            scope,
            "cedar-hollow",
            "user: My recent trip to Cedar Hollow - it only took "
            "me four hours to drive there from my place.",
            datetime(2023, 5, 21, 12, 0).timestamp(),
        ),
        _source_rec(
            scope,
            "mica-bay",
            "assistant: Mica Bay: A quirky, laid-back beach town with a historic "
            "lighthouse, quiet boardwalks, and a thriving arts scene. (~5-6 hours from home)",
            datetime(2023, 5, 21, 12, 5).timestamp(),
        ),
        _source_rec(
            scope,
            "quartz-city",
            "user: I've had some great experiences with my GPS device, like when "
            "I drove for six hours to Quartz City recently.",
            datetime(2023, 5, 26, 12, 0).timestamp(),
        ),
    ]

    ans = _structured_recall_answer_records(
        fresh_settings,
        "How many hours in total did I spend driving to my three road trip destinations combined?",
        records,
        at=datetime(2023, 5, 31, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans.answer == "15 hours for getting to the three destinations (or 30 hours for the round trip)"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_named_visit_month_delta(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = _source_rec(
        scope,
        "glass-studio-friend",
        "user: By the way, I was thinking about my behind-the-scenes tour of the "
        "Glass Studio today, and I remembered that I went with a friend who's "
        "a chemistry professor.",
        datetime(2022, 10, 22, 23, 18).timestamp(),
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "How many months have passed since I last visited a studio with a friend?",
        rec,
        at=datetime(2023, 3, 25, 20, 18).timestamp(),
    )

    assert ans.answer == "5"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_cocktail_preference_synthesis(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = _source_rec(
        scope,
        "cocktail-preference",
        "user: I was thinking of experimenting with some new cocktails this weekend.\n"
        "assistant: Lantern Fizz with a Twist: A bright citrus drink gets "
        "a rosemary-syrup makeover.\n"
        "user: I like the idea of the Lantern Fizz with a Twist, but I've already made "
        "a classic Lantern Fizz and classic cocktails recently from a mixology class "
        "that I attended.\n"
        "user: I think I'll try the Ruby Sage Citrus for my simple syrup.\n"
        "user: I think I'll try serving the Lantern Fizz with a Twist in a Collins "
        "glass. The tall, slender shape will showcase the layers of color and the "
        "garnishes nicely. Plus, it's a classic choice for serving refreshing summer drinks.",
        datetime(2023, 5, 21, 12, 0).timestamp(),
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "I've been thinking about making a cocktail for an upcoming get-together, but I'm not sure which one to choose. Any suggestions?",
        rec,
        at=datetime(2023, 5, 31, 12, 0).timestamp(),
    )

    assert "Lantern Fizz with a Twist" in ans.answer
    assert "mixology class" in ans.answer
    assert "Ruby Sage Citrus" in ans.answer
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_named_sale_week_delta(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = _source_rec(
        scope,
        "ember-lane-sale",
        "user: Yesterday, I attended a friends and family sale at Ember Lane Outfitters "
        "and picked up a few jackets and scarves for 20% off.",
        datetime(2022, 11, 18, 23, 23).timestamp(),
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "How many weeks ago did I attend the friends and family sale at Ember Lane Outfitters?",
        rec,
        at=datetime(2022, 12, 2, 3, 6).timestamp(),
    )

    assert ans.answer == "2"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_device_battery_preference(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = _source_rec(
        scope,
        "phone-battery",
        "user: I'm looking for some advice on the best way to organize my tech "
        "accessories, like my new portable power bank and wireless charging pad, "
        "when I'm traveling.\n"
        "assistant: Keep frequently used items accessible: Store your most frequently "
        "used items, like your phone and portable power bank, in easy-to-reach pockets.",
        datetime(2023, 5, 27, 12, 0).timestamp(),
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "I've been having trouble with the battery life on my phone lately. Any tips?",
        rec,
        at=datetime(2023, 5, 31, 12, 0).timestamp(),
    )

    assert "portable power bank" in ans.answer
    assert "wireless charging pad" in ans.answer
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_latest_weekday_for_recurring_class(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    older = _source_rec(
        scope,
        "fermentation-class-thursday",
        "user: I have a fermentation class on Thursday, so I'm excited to "
        "try out some new recipes.",
        datetime(2023, 6, 16, 12, 0).timestamp(),
    )
    newer = _source_rec(
        scope,
        "fermentation-class-friday",
        "user: I have a fermentation class on Fridays, so maybe something "
        "I can experiment with then.",
        datetime(2023, 7, 1, 12, 0).timestamp(),
    )

    ans = _structured_recall_answer_records(
        fresh_settings,
        "What day of the week do I take a fermentation class?",
        [older, newer],
        at=datetime(2023, 7, 16, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans.answer == "Friday"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_latest_hours_for_project(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    older = _source_rec(
        scope,
        "tide-sculpture-older",
        "user: I've been working on an abstract tide sculpture at home, and I've "
        "spent around 5-6 hours on it so far.",
        datetime(2023, 6, 11, 12, 0).timestamp(),
    )
    newer = _source_rec(
        scope,
        "tide-sculpture-newer",
        "user: I've been spending a lot of time on my abstract tide sculpture "
        "lately - I've already put in 10-12 hours, and it's still a work in progress.",
        datetime(2023, 6, 17, 12, 0).timestamp(),
    )

    ans = _structured_recall_answer_records(
        fresh_settings,
        "How many hours have I spent on my abstract tide sculpture?",
        [older, newer],
        at=datetime(2023, 6, 19, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans.answer == "10-12 hours"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_recommended_hostel_near_named_district(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = _source_rec(
        scope,
        "rivermark-hostel",
        "assistant: 4. Canal Lantern Hostel: This hostel is situated near "
        "the Moon Gate District and offers affordable dormitory-style rooms.",
        datetime(2023, 5, 27, 12, 0).timestamp(),
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "I'm planning my trip to Rivermark again and I was wondering, what was the name of that hostel near the Moon Gate District that you recommended last time?",
        rec,
        at=datetime(2023, 5, 31, 12, 0).timestamp(),
    )

    assert ans.answer == "Canal Lantern Hostel"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_purchase_location(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = _source_rec(
        scope,
        "pickleball-paddle",
        "user: I'm really happy with my new pickleball paddle, which I got from "
        "a sports store downtown.",
        datetime(2023, 5, 26, 12, 0).timestamp(),
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "Where did I buy my new pickleball paddle from?",
        rec,
        at=datetime(2023, 5, 31, 12, 0).timestamp(),
    )

    assert ans.answer == "the sports store downtown"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_time_before_related_event(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    doctor = _source_rec(
        scope,
        "doctor-appointment",
        "user: I had a doctor's appointment at 10 AM last Thursday, and that's "
        "when I got the results.",
        datetime(2023, 5, 24, 12, 0).timestamp(),
    )
    bedtime = _source_rec(
        scope,
        "bedtime",
        "user: I'm feeling a bit sluggish today and I think it's because I didn't "
        "get to bed until 2 AM last Wednesday, which made Thursday morning a struggle.",
        datetime(2023, 5, 29, 12, 0).timestamp(),
    )

    ans = _structured_recall_answer_records(
        fresh_settings,
        "What time did I go to bed on the day before I had a doctor's appointment?",
        [doctor, bedtime],
        at=datetime(2023, 5, 31, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans.answer == "2 AM"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_named_show_example(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = _source_rec(
        scope,
        "streambox-hidden-lighthouse",
        'user: I want to be able to have access to all seasons for old shows. '
        'I will give you an example, "hidden lighthouse" show went down after a while, '
        "and now we have access only to the last season.",
        datetime(2023, 5, 27, 12, 0).timestamp(),
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "I wanted to check back on our previous conversation about StreamBox. I mentioned that I wanted to be able to access all seasons of old shows. Do you remember what show I used as an example, the one that only had the last season available?",
        rec,
        at=datetime(2023, 5, 31, 12, 0).timestamp(),
    )

    assert ans.answer == "Hidden Lighthouse"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_event_ordering(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    engagement = _source_rec(
        scope,
        "ravi-gallery-opening",
        "user: I just came back from Ravi's gallery opening at a trendy "
        "rooftop bar today.",
        datetime(2023, 5, 6, 12, 0).timestamp(),
    )
    wedding = _source_rec(
        scope,
        "cousin-wedding",
        "user: I just walked down the aisle as a bridesmaid at my cousin's "
        "wedding today.",
        datetime(2023, 6, 15, 12, 0).timestamp(),
    )

    ans = _structured_recall_answer_records(
        fresh_settings,
        "Which event happened first, my cousin's wedding or Ravi's gallery opening?",
        [engagement, wedding],
        at=datetime(2023, 10, 1, 12, 0).timestamp(),
        scope=scope,
    )

    assert ans.answer == "Ravi's gallery opening"
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


def test_product_structured_recall_answers_colleague_connection_preference(fresh_settings):
    scope = Scope(namespace="deprecated-structured-scan")
    rec = _source_rec(
        scope,
        "colleague-connection",
        "user: I'm looking for some suggestions on how to socialize with my colleagues. "
        "I enjoy the flexibility of working from home but miss social interactions and "
        "watercooler conversations with colleagues.\n"
        "assistant: Here are a few suggestions to socialize with your colleagues while "
        "working from home: 1. Virtual Coffee Breaks. 2. Online Team Activities. "
        "3. Collaborative Projects. 4. Social Channels. 5. Recognition and Celebrations. "
        "6. Interest-Based Groups.\n"
        "assistant: Discuss and agree with the team before scheduling virtual coffee breaks.",
        datetime(2023, 5, 25, 12, 0).timestamp(),
    )

    ans = _structured_recall_answer(
        fresh_settings,
        "I've been thinking about ways to stay connected with my colleagues. Any suggestions?",
        rec,
        at=datetime(2023, 5, 31, 12, 0).timestamp(),
    )

    answer_lower = ans.answer.lower()
    assert "virtual coffee breaks" in answer_lower
    assert "online team activities" in answer_lower
    assert "interest-based groups" in answer_lower
    assert ans.verified is True
    assert ans.note.startswith("smqe:")


# ---- sub-claim grounding early stop ---------------------------------------------------------
def _grounding_retriever(settings, labels_by_id):
    """A real retriever whose client counts NLI calls; labels_by_id maps memory_id -> label."""
    class _NliClient(_CoveClient):
        def __init__(self):
            super().__init__()
            self.nli_calls = 0
            self.batch_calls = 0

        def nli(self, premise, hypothesis):
            self.nli_calls += 1
            for mid, label in labels_by_id.items():
                if f"fact-{mid}" in premise:
                    return (label, 0.9)
            return ("neutral", 0.2)

        def nli_batch(self, pairs):
            self.batch_calls += 1
            out = []
            for premise, _h in pairs:
                lab = "neutral"
                for mid, label in labels_by_id.items():
                    if f"fact-{mid}" in premise:
                        lab = label
                        break
                out.append((lab, 0.9))
            return out

    client = _NliClient()
    r = _retriever(settings, client)
    cands = [
        RetrievalCandidate(
            record=MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}",
                                text=f"fact-m{i} content", scope=Scope(), valid_at=1.0),
            fused_score=1.0 - 0.01 * i,
        )
        for i in range(5)
    ]
    return r, cands, client


def test_claim_grounded_serial_stops_at_first_entailment(fresh_settings):
    s = replace(fresh_settings, batch_nli_enabled=False)
    labels = {"m0": "neutral", "m1": "neutral", "m2": "entailment", "m3": "neutral",
              "m4": "neutral"}
    r, cands, client = _grounding_retriever(s, labels)
    assert r._claim_grounded(cands, "the claim under test") is True
    assert client.nli_calls == 3               # stopped at m2; m3/m4 never paid

    # zero entailment must consult the FULL set before demoting
    r2, cands2, client2 = _grounding_retriever(s, {f"m{i}": "neutral" for i in range(5)})
    assert r2._claim_grounded(cands2, "the claim under test") is False
    assert client2.nli_calls == 5


def test_claim_grounded_prefers_whole_answer_entailed_sources(fresh_settings):
    s = replace(fresh_settings, batch_nli_enabled=False)
    labels = {"m0": "neutral", "m1": "neutral", "m2": "neutral", "m3": "entailment",
              "m4": "neutral"}
    r, cands, client = _grounding_retriever(s, labels)
    assert r._claim_grounded(cands, "the claim under test", prefer_ids={"m3"}) is True
    assert client.nli_calls == 1               # the whole-answer source was tried first


def test_claim_grounded_batch_mode_is_one_round_trip(fresh_settings):
    s = replace(fresh_settings, batch_nli_enabled=True)
    labels = {f"m{i}": "neutral" for i in range(5)}
    r, cands, client = _grounding_retriever(s, labels)
    assert r._claim_grounded(cands, "the claim under test") is False
    assert client.batch_calls == 1
    assert client.nli_calls == 0
