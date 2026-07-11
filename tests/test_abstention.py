"""Offline tests for calibrated abstention (Phase 2): pure signals + the answer/abstain decision."""
from __future__ import annotations

import types
from dataclasses import replace

import pytest

from eidetic.abstention import (channel_agreement, combine_confidence, pick_tau,
                                proof_completeness)


def test_answer_status_contract_and_abstention_factory():
    from eidetic.models import Answer, AnswerStatus

    verified = Answer(question="q", answer="a", verified=True)
    assert verified.status == AnswerStatus.VERIFIED
    abstained = Answer.abstain("q", note="no proof")
    assert abstained.status == AnswerStatus.ABSTAINED
    assert abstained.verified is False
    assert abstained.confidence == 0.0
    assert abstained.citations == []
    with pytest.raises(ValueError, match="disagree"):
        Answer(question="q", answer="a", status=AnswerStatus.VERIFIED, verified=False)


def test_engine_rejects_answer_verification_bypass(engine):
    with pytest.raises(ValueError, match="requires verification"):
        engine.ask("q", verify=False)


def test_empty_memory_is_explicit_abstention(engine):
    answer = engine.ask("what is my launch code?")
    assert answer.status.value == "ABSTAINED"
    assert answer.verified is False
    assert answer.citations == []
    assert answer.note.startswith("abstained")


def test_engine_rejects_dangling_citation_span(engine):
    from eidetic.models import Answer, Citation, MemoryRecord, NLILabel, Scope

    scope = Scope(namespace="dangling-proof")
    raw_text = "User: The launch code is BLUE-17."
    indexed_text = "User: The launch code is RED-99."
    content_hash, raw_uri = engine.substrate.put(raw_text.encode("utf-8"))
    record = MemoryRecord(
        memory_id="dangling-proof-memory",
        text=indexed_text,
        source="user",
        scope=scope,
        valid_at=1.0,
        content_hash=content_hash,
        raw_uri=raw_uri,
    )
    engine.store.upsert_record(record)
    answer = Answer(
        question="What is the launch code?",
        answer="RED-99",
        verified=True,
        citations=[Citation(
            memory_id=record.memory_id,
            content_hash=record.content_hash,
            raw_uri=record.raw_uri,
            source=record.source,
            valid_at=record.valid_at,
            snippet="User: The launch code is RED-99.",
            nli_label=NLILabel.ENTAILMENT,
            nli_score=1.0,
        )],
    )

    governed = engine._govern_answer(answer.question, answer, scope)

    assert governed.status.value == "ABSTAINED"
    assert governed.citations == []
    assert "immutable proof" in governed.note


def test_engine_rejects_citation_inactive_at_query_time(engine):
    from eidetic.models import Answer, Citation, MemoryRecord, NLILabel, Scope

    scope = Scope(namespace="inactive-proof")
    text = "User: My office is in Berlin."
    content_hash, raw_uri = engine.substrate.put(text.encode("utf-8"))
    record = MemoryRecord(
        memory_id="inactive-proof-memory",
        text=text,
        source="user",
        scope=scope,
        valid_at=1.0,
        invalid_at=2.0,
        content_hash=content_hash,
        raw_uri=raw_uri,
    )
    engine.store.upsert_record(record)
    answer = Answer(
        question="Where is my office?",
        answer="Berlin",
        verified=True,
        citations=[Citation(
            memory_id=record.memory_id,
            content_hash=record.content_hash,
            raw_uri=record.raw_uri,
            source=record.source,
            valid_at=record.valid_at,
            snippet=text,
            nli_label=NLILabel.ENTAILMENT,
            nli_score=1.0,
        )],
    )

    governed = engine._govern_answer(answer.question, answer, scope, at=3.0)

    assert governed.status.value == "ABSTAINED"
    assert governed.citations == []


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


def test_advice_answer_grounds_on_context_restatement(fresh_settings):
    """A recommendation answer is synthesis by design: fresh suggestions are never entailed by
    memory as a whole answer. The verifiable core is the answer's restatement of the user's
    stored context; sentence-level verification must ground the answer on that restatement
    instead of abstaining on entail=0."""
    import hashlib
    import re as _re

    import numpy as np

    from eidetic.engine import Engine
    from eidetic.models import NLILabel, Scope

    class _AdviceClient:
        def __init__(self, dim):
            self.dim = dim

        def _e(self, t):
            v = np.zeros(self.dim, np.float32)
            for tok in _re.findall(r"[a-z0-9]+", (t or "").lower()):
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
            return ("You are learning advanced glazing techniques with the kiln at home. "
                    "For next steps, try community studio workshops and mineral pigment guides.")

        def nli(self, premise, hypothesis):
            hyp = (hypothesis or "").lower()
            if "glazing" in hyp and "workshops" not in hyp:
                return ("entailment", 0.9)     # the context restatement grounds in memory
            return ("neutral", 0.2)            # fresh suggestions never entail

    s = replace(fresh_settings, rerank_enabled=False)
    e = Engine(s, client=_AdviceClient(s.embed_dim))
    ns = Scope(namespace="advice-ground")
    e.ingest_text("User: I'm trying to learn advanced glazing techniques with my kiln at home.",
                  scope=ns, consolidate_now=False)
    e.consolidate_pending(scope=ns, score_importance=False)

    ans = e.ask("Can you suggest some resources to learn more about pottery glazing?", scope=ns)

    assert not ans.note.startswith("abstained")
    assert "workshops" in ans.answer            # the synthesis is delivered, not swallowed
    assert ans.verified is True                 # grounded on the entailed restatement
    assert any(c.nli_label == NLILabel.ENTAILMENT for c in ans.citations)


def test_likely_inference_answer_grounds_on_premise_restatement(fresh_settings):
    """'Is it likely that X?' answers are labeled inference over cited premises - the same
    structural shape as advice: the yes/no marker never whole-answer-entails, so a correct
    synthesis died unverified. The sentence-level rescue must cover likelihood questions."""
    import hashlib
    import re as _re

    import numpy as np

    from eidetic.engine import Engine
    from eidetic.models import NLILabel, Scope

    class _LikelyClient:
        def __init__(self, dim):
            self.dim = dim

        def _e(self, t):
            v = np.zeros(self.dim, np.float32)
            for tok in _re.findall(r"[a-z0-9]+", (t or "").lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        def embed_text(self, t):
            return self._e(t)

        def embed_texts(self, ts):
            return (np.stack([self._e(t) for t in ts])
                    if ts else np.zeros((0, self.dim), np.float32))

        def extract_edges(self, text):
            return []

        def generate_answer(self, q, blocks, model=None):
            return ("Likely yes. Rowan mentions hiking with colleagues from the trail club. "
                    "That suggests an active social circle beyond his sister.")

        def nli(self, premise, hypothesis):
            hyp = (hypothesis or "").lower()
            if "trail club" in hyp and "suggests" not in hyp:
                return ("entailment", 0.9)     # the premise restatement grounds
            return ("neutral", 0.2)            # the inference marker never entails

    from dataclasses import replace as _replace
    s = _replace(fresh_settings, rerank_enabled=False)
    e = Engine(s, client=_LikelyClient(s.embed_dim))
    ns = Scope(namespace="likely-ground")
    e.ingest_text("User: Rowan went hiking with colleagues from the trail club again.",
                  scope=ns, consolidate_now=False)
    e.consolidate_pending(scope=ns, score_importance=False)

    ans = e.ask("Is it likely that Rowan has friends besides his sister?", scope=ns)

    assert not ans.note.startswith("abstained")
    assert ans.verified is True
    assert any(c.nli_label == NLILabel.ENTAILMENT for c in ans.citations)


def test_rescue_grounds_sentence_of_verbatim_quotes_across_records(fresh_settings):
    """A synthesis sentence that cites MULTIPLE verbatim quoted spans from different records
    ('mentions 'the trail club', 'old friends', ...') never single-record-entails - but the
    quotes ARE anchors. When every quoted span is found verbatim in some candidate, the
    sentence is grounded extractively, no model call."""
    import hashlib
    import re as _re

    import numpy as np

    from eidetic.engine import Engine
    from eidetic.models import NLILabel, Scope

    class _QuoteClient:
        def __init__(self, dim):
            self.dim = dim

        def _e(self, t):
            v = np.zeros(self.dim, np.float32)
            for tok in _re.findall(r"[a-z0-9]+", (t or "").lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        def embed_text(self, t):
            return self._e(t)

        def embed_texts(self, ts):
            return (np.stack([self._e(t) for t in ts])
                    if ts else np.zeros((0, self.dim), np.float32))

        def extract_edges(self, text):
            return []

        def generate_answer(self, q, blocks, model=None):
            return ("Likely yes. Rowan mentions hiking with 'colleagues from the trail club' "
                    "and grabbing lunch with 'friends from the chess league' regularly.")

        def nli(self, premise, hypothesis):
            return ("neutral", 0.2)            # NLI never fires: grounding must be extractive

    from dataclasses import replace as _replace
    s = _replace(fresh_settings, rerank_enabled=False)
    e = Engine(s, client=_QuoteClient(s.embed_dim))
    ns = Scope(namespace="quote-ground")
    e.ingest_text("User: Rowan went hiking with colleagues from the trail club again.",
                  scope=ns, consolidate_now=False)
    e.ingest_text("User: Rowan grabbed lunch with friends from the chess league.",
                  scope=ns, consolidate_now=False)
    e.consolidate_pending(scope=ns, score_importance=False)

    ans = e.ask("Is it likely that Rowan has friends besides his sister?", scope=ns)

    assert not ans.note.startswith("abstained")
    assert ans.verified is True
    assert any(c.nli_label == NLILabel.ENTAILMENT for c in ans.citations)


def test_quoted_anchor_falls_back_to_store_when_record_outside_shortlist(fresh_settings):
    """A quoted span whose source record fell outside the retrieval shortlist is still
    provable against the scoped active store - retrieval rank is not part of the
    verification contract."""
    from dataclasses import replace as _replace

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    s = _replace(fresh_settings, rerank_enabled=False)
    store = RecordStore(s.sqlite_path)
    scope = Scope(namespace="quote-store")
    in_short = MemoryRecord(memory_id="m1", content_hash="h1",
                            text="Rowan went hiking with colleagues from the trail club.",
                            scope=scope, valid_at=1.0)
    out_short = MemoryRecord(memory_id="m2", content_hash="h2",
                             text="Rowan grabbed lunch with friends from the chess league.",
                             scope=scope, valid_at=2.0)
    store.upsert_record(in_short)
    store.upsert_record(out_short)
    class _Sub:
        def get(self, h):
            raise KeyError(h)          # -> _ground_truth falls back to rec.text

    r = Retriever(store, object(), KnowledgeGraph(store), _Sub(), object(), s)

    sentence = ("Rowan mentions 'colleagues from the trail club' and "
                "'friends from the chess league' regularly.")
    only_first = [RetrievalCandidate(record=in_short, fused_score=1.0)]
    out = r._quoted_span_anchors(only_first, sentence, scope=scope, at=10.0)
    assert out is not None
    citations, entailed = out
    assert entailed == 2
    assert {c.memory_id for c in citations} == {"m1", "m2"}

    # without scope (no store fallback) the missing span fails the sentence
    assert r._quoted_span_anchors(only_first, sentence) is None


def test_abstentions_never_ship_citations_as_support(fresh_settings):
    """'I don't have enough verified evidence' followed by a source list reads as a
    contradiction (MCP UX exercise finding: abstention carried 4 citations). Abstained
    answers ship an empty citation list; the considered sources stay in telemetry."""
    from dataclasses import replace as _replace

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    class _NeutralClient:
        def generate_answer(self, q, blocks, model=None):
            return "Something unrelated to any source."

        def nli(self, premise, hypothesis):
            return ("neutral", 0.2)

    s = _replace(fresh_settings, rerank_enabled=False, abstention_threshold=1.0,
                 cascade_enabled=False)
    store = RecordStore(s.sqlite_path)

    class _Sub:
        def get(self, h):
            raise KeyError(h)

    r = Retriever(store, object(), KnowledgeGraph(store), _Sub(), _NeutralClient(), s)
    cands = [RetrievalCandidate(record=MemoryRecord(
        memory_id=f"m{i}", content_hash=f"h{i}", text=f"fact {i}",
        scope=Scope(), valid_at=1.0), dense_score=0.1) for i in range(4)]
    r._try_conflict_resolver = lambda *a, **k: None
    r.assemble_context = lambda *a, **k: ["[S0] fact"]

    ans = r.answer("what is my blood type", verify=True, precomputed=cands)
    assert ans.note.startswith("abstained")
    assert ans.citations == []


def _fast_abstain_retriever(fresh_settings, *, flag, floor=0.25, reader_calls=None, v2=False):
    from dataclasses import replace as _replace

    from eidetic.graph import KnowledgeGraph
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    class _CountingClient:
        def generate_answer(self, q, blocks, model=None):
            if reader_calls is not None:
                reader_calls.append(q)
            return "Something unrelated to any source."

        def nli(self, premise, hypothesis):
            return ("neutral", 0.2)

    s = _replace(fresh_settings, rerank_enabled=False, cascade_enabled=False,
                 fast_abstain_enabled=flag, fast_abstain_floor=floor,
                 abstention_threshold=0.4, abstention_v2_enabled=v2)
    store = RecordStore(s.sqlite_path)

    class _Sub:
        def get(self, h):
            raise KeyError(h)

    r = Retriever(store, object(), KnowledgeGraph(store), _Sub(), _CountingClient(), s)
    r._try_conflict_resolver = lambda *a, **k: None
    r.assemble_context = lambda *a, **k: ["[S0] fact"]
    return r


def _weak_candidates(dense=0.1, n=3):
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope

    return [RetrievalCandidate(record=MemoryRecord(
        memory_id=f"m{i}", content_hash=f"h{i}", text=f"fact {i}",
        scope=Scope(), valid_at=1.0), dense_score=dense) for i in range(n)]


def test_fast_abstain_skips_reader_below_floor(fresh_settings):
    """Coverage far under the threshold: the 5-7s reader call buys nothing -- the coverage
    gate discards its draft anyway. Pre-gate abstains in-process, no model call, no citations."""
    calls: list[str] = []
    r = _fast_abstain_retriever(fresh_settings, flag=True, reader_calls=calls)
    ans = r.answer("what is my blood type", verify=True,
                   precomputed=_weak_candidates(dense=0.1))
    assert ans.note == "abstained: insufficient evidence (coverage 0.10, pre-reader)"
    assert ans.citations == [] and not ans.verified and ans.confidence == 0.0
    assert calls == []                      # the reader was never invoked


def test_fast_abstain_above_floor_keeps_reader_rescue_path(fresh_settings):
    """Coverage between floor and threshold: the draft can still be NLI-rescued, so the
    reader must run; the ordinary coverage gate decides afterwards."""
    calls: list[str] = []
    r = _fast_abstain_retriever(fresh_settings, flag=True, reader_calls=calls)
    ans = r.answer("what is my blood type", verify=True,
                   precomputed=_weak_candidates(dense=0.3))
    assert len(calls) == 1                  # full pipeline ran
    assert ans.note == "abstained: insufficient evidence (coverage 0.30)"


def test_fast_abstain_flag_off_is_baseline(fresh_settings):
    calls: list[str] = []
    r = _fast_abstain_retriever(fresh_settings, flag=False, reader_calls=calls)
    ans = r.answer("what is my blood type", verify=True,
                   precomputed=_weak_candidates(dense=0.1))
    assert len(calls) == 1                  # baseline: reader always consulted
    assert ans.note == "abstained: insufficient evidence (coverage 0.10)"


def test_fast_abstain_defers_to_abstention_v2(fresh_settings):
    """ABSTENTION_V2's calibrated confidence needs the citation signals, so the pre-gate
    must not fire when v2 owns the decision."""
    calls: list[str] = []
    r = _fast_abstain_retriever(fresh_settings, flag=True, reader_calls=calls, v2=True)
    r.answer("what is my blood type", verify=True,
             precomputed=_weak_candidates(dense=0.1))
    assert len(calls) == 1                  # v2 path kept the full pipeline
