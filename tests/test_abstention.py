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
