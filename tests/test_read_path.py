"""Offline tests for S1 read-path latency: batched NLI, short-circuit verify, deferred re-embed."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.config import get_settings
from eidetic.dashscope_client import DashScopeClient
from eidetic.graph import KnowledgeGraph
from eidetic.models import (BrainEventType, MemoryRecord, NLILabel, RetrievalCandidate, Scope)
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore


class _FakeEmbed:
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


class _FakeSub:
    def get(self, h):
        raise KeyError(h)            # -> _ground_truth falls back to rec.text


def _cands(n):
    return [RetrievalCandidate(record=MemoryRecord(memory_id=f"m{i}", content_hash=f"h{i}",
            text=f"fact {i}", scope=Scope(), valid_at=1.0)) for i in range(n)]


# ---- batched NLI ---------------------------------------------------------------------------
def test_nli_batch_maps_by_index_and_fills_missing_as_neutral(monkeypatch):
    c = DashScopeClient(get_settings())
    monkeypatch.setattr(c, "chat_json", lambda *a, **k: {"results": [
        {"index": 0, "label": "entailment", "confidence": 0.9},
        {"index": 2, "label": "contradiction", "confidence": 0.8},   # index 1 omitted -> neutral
    ]})
    out = c.nli_batch([("p0", "h"), ("p1", "h"), ("p2", "h")])
    assert out == [("entailment", 0.9), ("neutral", 0.0), ("contradiction", 0.8)]


def test_batch_nli_verify_path_matches_labels(fresh_settings):
    s = replace(fresh_settings, batch_nli_enabled=True)
    store = RecordStore(s.sqlite_path)

    class C:
        def nli_batch(self, pairs):
            return [("entailment", 0.9)] * len(pairs)

    r = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), C(), s)
    cits, entailed = r._verify_candidates(_cands(3), "ans", verify=True)
    assert entailed == 3 and all(c.nli_label == NLILabel.ENTAILMENT for c in cits)


# ---- short-circuit verify ------------------------------------------------------------------
def test_fast_verify_short_circuits_after_cap(fresh_settings, monkeypatch):
    s = replace(fresh_settings, fast_verify_enabled=True, verify_citation_cap=2)
    store = RecordStore(s.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), object(), s)
    calls = {"n": 0}

    def fake_verify(rec, text):
        calls["n"] += 1
        return NLILabel.ENTAILMENT, 0.9

    monkeypatch.setattr(r, "verify_citation", fake_verify)
    cits, entailed = r._verify_candidates(_cands(5), "ans", verify=True)
    assert calls["n"] == 2 and entailed == 2          # stopped after the cap
    assert len(cits) == 5                             # the rest still get a (neutral) citation


def test_baseline_verify_unchanged_when_flags_off(fresh_settings, monkeypatch):
    store = RecordStore(fresh_settings.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), object(), fresh_settings)
    monkeypatch.setattr(r, "verify_citation", lambda rec, text: (NLILabel.ENTAILMENT, 0.7))
    cits, entailed = r._verify_candidates(_cands(4), "ans", verify=True)
    assert entailed == 4 and len(cits) == 4          # every candidate verified serially (baseline)


# ---- deferred re-embed ---------------------------------------------------------------------
def test_deferred_reembed_enqueue_and_drain(fresh_settings):
    from eidetic.engine import Engine
    e = Engine(replace(fresh_settings, defer_reembed_enabled=True, brain_events_enabled=True),
               client=_FakeEmbed(fresh_settings.embed_dim))
    scope = Scope(namespace="d")
    rec = e.ingest_text("alice likes green tea", scope=scope, consolidate_now=False)

    e._enqueue_reembed([rec.memory_id])
    assert rec.memory_id in e._reembed_queue
    out = e.drain_reembed_queue()
    assert out["reembedded"] == 1
    assert rec.memory_id not in e._reembed_queue          # queue drained
    assert e.brain_log.by_type(BrainEventType.REEMBED_DEFERRED)
    # a second drain is a clean no-op.
    assert e.drain_reembed_queue()["reembedded"] == 0


# ---- reader ordering guidance ----------------------------------------------------------------
def test_reader_prompt_anchors_event_order_questions(monkeypatch):
    """Ordering questions get an explicit date-anchoring instruction: an answer that merely
    echoes the question's event phrases is unprovable; dates make the ordering checkable."""
    c = DashScopeClient(replace(get_settings(), reader_cot_enabled=True))
    seen = {}

    def fake_chat_json(model, system, user, **kw):
        seen["system"] = system
        return {"answer": "ok"}

    monkeypatch.setattr(c, "chat_json", fake_chat_json)
    c.generate_answer(
        "Which three events happened in the order from first to last: the fair, the recital, "
        "and the workshop?",
        ["[2024-02-05] the fair", "[2024-03-01] the recital"],
    )
    assert "chronological" in seen["system"].lower()
    assert "date" in seen["system"].lower()

    c.generate_answer("What color is my bike?", ["the bike is red"])
    assert "chronological" not in seen["system"].lower()


def test_reader_prompt_excludes_established_items_on_other_questions(monkeypatch):
    """'What OTHER exercises ...' asks for items beyond the established ones; without the
    exclusion instruction the reader echoes the subject's current routine (observed live:
    strength-training/yoga returned where the asked-for additions were in the sources)."""
    c = DashScopeClient(replace(get_settings(), reader_cot_enabled=True))
    seen = {}

    def fake_chat_json(model, system, user, **kw):
        seen["system"] = system
        return {"answer": "ok"}

    monkeypatch.setattr(c, "chat_json", fake_chat_json)
    for q in ("What other exercises can help John with his basketball performance?",
              "Besides painting, what hobbies does Maya have?",
              "What else did she pack apart from the tent?"):
        c.generate_answer(q, ["[S0] ctx"])
        assert "ADDITIONAL" in seen["system"], q

    c.generate_answer("What color is my bike?", ["the bike is red"])
    assert "ADDITIONAL" not in seen["system"]


# ---- answer-path index save gating -----------------------------------------------------------
def test_ask_saves_index_only_when_reconsolidation_mutated_it(fresh_settings, monkeypatch):
    """The per-answer index.save() is an O(corpus) disk write under the write lock. It must run
    only when reconsolidation actually updated a vector this ask: with DEFER_REEMBED the update
    is queued for the idle drain, so saving an unchanged index buys nothing and serializes
    concurrent asks behind disk IO that grows with corpus size."""
    from dataclasses import replace as _replace

    from eidetic.engine import Engine
    from eidetic.models import Scope

    class _Client(_FakeEmbed):
        def extract_edges(self, text):
            return []

        def generate_answer(self, q, blocks, model=None):
            return blocks[0][:200] if blocks else "I do not have that in memory."

        def nli(self, premise, hypothesis):
            return ("entailment", 0.9)

    # deferred mode: the ask must NOT save (no index mutation happened on the answer path)
    s = _replace(fresh_settings, defer_reembed_enabled=True, rerank_enabled=False)
    e = Engine(s, client=_Client(s.embed_dim))
    scope = Scope(namespace="savegate")
    e.ingest_text("The greenhouse fan runs on circuit twelve.", scope=scope, consolidate_now=False)
    saves = {"n": 0}
    real_save = e.index.save
    monkeypatch.setattr(e.index, "save", lambda: (saves.__setitem__("n", saves["n"] + 1),
                                                  real_save())[1])
    ans = e.ask("which circuit runs the greenhouse fan", scope=scope)
    assert ans.citations
    assert saves["n"] == 0

    # inline mode: the re-embed mutates the index -> the save must still happen
    s2 = _replace(fresh_settings, defer_reembed_enabled=False, rerank_enabled=False)
    e2 = Engine(s2, client=_Client(s2.embed_dim))
    scope2 = Scope(namespace="savegate2")
    e2.ingest_text("The orchard pump runs on circuit nine.", scope=scope2, consolidate_now=False)
    saves2 = {"n": 0}
    real_save2 = e2.index.save
    monkeypatch.setattr(e2.index, "save", lambda: (saves2.__setitem__("n", saves2["n"] + 1),
                                                   real_save2())[1])
    ans2 = e2.ask("which circuit runs the orchard pump", scope=scope2)
    assert ans2.citations
    assert saves2["n"] >= 1


# ---- FAST_VERIFY under BATCH_NLI: wave batching ----------------------------------------------
class _WaveClient:
    """nli_batch client that labels by premise marker and counts round trips."""

    def __init__(self, labels_by_id):
        self.labels_by_id = labels_by_id
        self.batches: list[int] = []

    def nli_batch(self, pairs):
        self.batches.append(len(pairs))
        out = []
        for premise, _h in pairs:
            lab = "neutral"
            for mid, label in self.labels_by_id.items():
                if f"fact {mid[1:]}" in premise:
                    lab = label
                    break
            out.append((lab, 0.9))
        return out


def _wave_retriever(fresh_settings, labels_by_id, cap=2):
    s = replace(fresh_settings, batch_nli_enabled=True, fast_verify_enabled=True,
                verify_citation_cap=cap)
    store = RecordStore(s.sqlite_path)
    client = _WaveClient(labels_by_id)
    r = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), client, s)
    return r, client


def test_batch_fast_verify_caps_first_wave(fresh_settings):
    """Both flags on (the shipping profile): wave 1 batches only the top verify_citation_cap
    candidates; an entailment there leaves the tail NEUTRAL and unpaid."""
    labels = {f"m{i}": ("entailment" if i == 0 else "neutral") for i in range(8)}
    r, client = _wave_retriever(fresh_settings, labels, cap=2)
    cits, entailed = r._verify_candidates(_cands(8), "ans", verify=True)
    assert entailed >= 1
    assert client.batches == [2]                  # one wave, cap-sized -- 6 pairs never paid
    assert len(cits) == 8

def test_batch_fast_verify_escalates_on_zero_entailment(fresh_settings):
    """Zero entailments in wave 1 -> the remainder is batched before deciding: the abstention
    decision is computed over the full set, exactly as wide as before."""
    labels = {f"m{i}": ("entailment" if i == 7 else "neutral") for i in range(8)}
    r, client = _wave_retriever(fresh_settings, labels, cap=2)
    cits, entailed = r._verify_candidates(_cands(8), "ans", verify=True)
    assert entailed == 1                          # the tail entailment was found in wave 2
    assert client.batches == [2, 6]

def test_batch_fast_verify_escalates_on_contradiction(fresh_settings):
    """A contradiction in wave 1 forces the full picture even when an entailment was found:
    the advice-rescue kill and reconsolidation lapse need every contradicting source."""
    labels = {"m0": "entailment", "m1": "contradiction",
              **{f"m{i}": "neutral" for i in range(2, 8)}}
    r, client = _wave_retriever(fresh_settings, labels, cap=2)
    cits, entailed = r._verify_candidates(_cands(8), "ans", verify=True)
    assert client.batches == [2, 6]
    assert any(c.nli_label == NLILabel.CONTRADICTION for c in cits)

def test_batch_without_fast_verify_stays_full_width(fresh_settings):
    s = replace(fresh_settings, batch_nli_enabled=True, fast_verify_enabled=False)
    store = RecordStore(s.sqlite_path)
    client = _WaveClient({f"m{i}": "neutral" for i in range(8)})
    r = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), client, s)
    r._verify_candidates(_cands(8), "ans", verify=True)
    assert client.batches == [8]                  # unchanged: one full-width batch


def test_assemble_context_scans_active_records_once(fresh_settings, monkeypatch):
    """Five audit channels share one active-record snapshot: the O(corpus) store scan must run
    at most once per assemble_context, and not at all when every channel is off."""
    from dataclasses import replace as _replace

    s = _replace(fresh_settings, scratchpad_enabled=True, user_evidence_context_enabled=True,
                 assistant_evidence_context_enabled=True, temporal_evidence_audit_enabled=True,
                 list_audit_enabled=True, active_fact_context_enabled=False,
                 graph_bridge_context_enabled=False, gist_channel_enabled=False)
    store = RecordStore(s.sqlite_path)
    from eidetic.models import Scope as _Scope
    scope = _Scope(namespace="hoist")
    store.upsert_record(MemoryRecord(memory_id="m1", content_hash="h1",
                                     text="user: the fern needs weekly watering",
                                     scope=scope, valid_at=1.0, source="user"))
    r = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), object(), s)
    calls = {"n": 0}
    real = store.active_records_at

    def counting(at=None, scope=None, **kw):
        calls["n"] += 1
        return real(at, scope, **kw)

    monkeypatch.setattr(store, "active_records_at", counting)
    r.assemble_context("when does the fern need watering",
                       _cands(0) or [], at=2.0, scope=scope)
    assert calls["n"] == 1

    s_off = _replace(fresh_settings, scratchpad_enabled=False,
                     user_evidence_context_enabled=False,
                     assistant_evidence_context_enabled=False,
                     temporal_evidence_audit_enabled=False, list_audit_enabled=False,
                     active_fact_context_enabled=False, graph_bridge_context_enabled=False,
                     gist_channel_enabled=False)
    r2 = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), object(), s_off)
    calls["n"] = 0
    r2.assemble_context("when does the fern need watering", [], at=2.0, scope=scope)
    assert calls["n"] == 0


def test_rerank_span_input_flag(fresh_settings, monkeypatch):
    """RERANK_SPAN_INPUT: on -> the cross-encoder sees bounded query-centered spans; off ->
    byte-identical full texts."""
    from dataclasses import replace as _replace

    long_text = ("filler sentence about nothing in particular.\n" * 120
                 + "the fern needs weekly watering.\n"
                 + "more filler trailing away.\n" * 120)
    seen = {}

    class _RerankClient:
        def rerank(self, query, documents, top_n):
            seen["docs"] = list(documents)
            return [(i, 1.0 - 0.01 * i) for i in range(min(top_n, len(documents)))]

    def cand():
        return RetrievalCandidate(record=MemoryRecord(
            memory_id="m0", content_hash="h0", text=long_text,
            scope=Scope(), valid_at=1.0), fused_score=1.0)

    s_on = _replace(fresh_settings, rerank_enabled=True, rerank_span_input_enabled=True,
                    rerank_span_chars=800, rerank_skip_margin=0.0)
    store = RecordStore(s_on.sqlite_path)
    r = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), _RerankClient(), s_on)
    r._finalize("when does the fern need watering", [cand()])
    assert len(seen["docs"][0]) <= 1000
    assert "fern needs weekly watering" in seen["docs"][0]

    s_off = _replace(fresh_settings, rerank_enabled=True, rerank_span_input_enabled=False,
                     rerank_skip_margin=0.0)
    r2 = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), _RerankClient(), s_off)
    r2._finalize("when does the fern need watering", [cand()])
    assert seen["docs"][0] == long_text


def test_adaptive_context_scales_budget_with_difficulty(fresh_settings, monkeypatch):
    """ADAPTIVE_CONTEXT: an easy single-hop question gets the floor fraction of the token
    budget; a hard multi-hop question keeps the full budget. Flag off: byte-identical."""
    from dataclasses import replace as _replace

    import eidetic.retrieval as retrieval_mod
    from eidetic.models import Scope as _Scope

    captured = {}
    real_budget = retrieval_mod._budget_blocks

    def spy(blocks, budget):
        captured["budget"] = budget
        return real_budget(blocks, budget)

    monkeypatch.setattr(retrieval_mod, "_budget_blocks", spy)

    def build(settings):
        store = RecordStore(settings.sqlite_path)
        return Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), object(), settings)

    easy_q = "What color is the bike?"
    hard_q = ("Which of the three trips that Rowan and Priya planned together after the "
              "spring festival happened before the harvest market weekend in Lisbon?")

    s_on = _replace(fresh_settings, adaptive_context_enabled=True, adaptive_context_floor=0.45,
                    context_token_budget=8000)
    r = build(s_on)
    r.assemble_context(easy_q, [], at=2.0, scope=_Scope(namespace="ac"))
    easy_budget = captured["budget"]
    r.assemble_context(hard_q, [], at=2.0, scope=_Scope(namespace="ac"))
    hard_budget = captured["budget"]
    assert easy_budget < hard_budget
    assert easy_budget <= 8000 * 0.6
    assert hard_budget >= 8000 * 0.8

    s_off = _replace(fresh_settings, adaptive_context_enabled=False, context_token_budget=8000)
    r2 = build(s_off)
    r2.assemble_context(easy_q, [], at=2.0, scope=_Scope(namespace="ac"))
    assert captured["budget"] == 8000


def test_verify_citation_lru_memoizes_successful_verdicts(fresh_settings, monkeypatch):
    """VERIFY_NLI_CACHE: an identical (premise, hypothesis, model) NLI pair costs one client
    call; only successful verdicts memoize; flag off is byte-identical (every call paid)."""
    from dataclasses import replace as _replace

    calls = {"n": 0}

    class _NliClient:
        def nli(self, premise, hypothesis):
            calls["n"] += 1
            return ("entailment", 0.9)

    def build(enabled):
        s = _replace(fresh_settings, verify_nli_cache_enabled=enabled)
        store = RecordStore(s.sqlite_path)
        r = Retriever(store, object(), KnowledgeGraph(store), _FakeSub(), _NliClient(), s)
        return r

    rec = MemoryRecord(memory_id="m1", content_hash="h1",
                       text="The fern needs weekly watering in summer.",
                       scope=Scope(), valid_at=1.0)

    r = build(True)
    calls["n"] = 0
    r.verify_citation(rec, "the fern is watered weekly")
    r.verify_citation(rec, "the fern is watered weekly")
    r.verify_citation(rec, "THE FERN   is watered weekly")   # whitespace/case-normalized hit
    assert calls["n"] == 1
    r.verify_citation(rec, "the cactus is watered weekly")   # different hypothesis -> miss
    assert calls["n"] == 2

    r_off = build(False)
    calls["n"] = 0
    r_off.verify_citation(rec, "the fern is watered weekly")
    r_off.verify_citation(rec, "the fern is watered weekly")
    assert calls["n"] == 2


def test_usage_counters_accumulate_real_api_spend(fresh_settings, monkeypatch):
    """Model-spend accounting: chat responses' own usage numbers accumulate; snapshot/delta
    expose dollars-shaped tokens (the write_tokens column only ever counted content volume)."""
    c = DashScopeClient(fresh_settings)

    class _Usage:
        input_tokens = 120
        output_tokens = 30

    class _Resp:
        status_code = 200
        usage = _Usage()
        output = {"choices": [{"message": {"content": "ok"}}]}

    class _Gen:
        @staticmethod
        def call(**kw):
            return _Resp()

    monkeypatch.setattr(c, "_require_key", lambda: None)
    monkeypatch.setattr(c, "_ds", type("DS", (), {"Generation": _Gen}))
    before = c.usage_snapshot()
    c.chat("m", "sys", "user")
    c.chat("m", "sys", "user")
    delta = c.usage_delta(before, c.usage_snapshot())
    assert delta == {"input_tokens": 240, "output_tokens": 60, "calls": 2}


# ---- reader-path form floor ------------------------------------------------------------------
def test_reader_form_floor_demotes_verbatim_fragment_answers(fresh_settings):
    """Every verified-wrong row of rotation slice 2 came through the READER path: the
    photographic reader quotes sources verbatim, so a conversational fragment ('I'm
    reading') entails trivially and ships verified while answering nothing. The universal
    form floor demotes such answers to unverified; the coverage gate then abstains. Kill
    switch READER_FORM_FLOOR=0 restores the old behavior for one release."""
    from dataclasses import replace as _replace

    from eidetic.graph import KnowledgeGraph
    from eidetic.models import MemoryRecord, RetrievalCandidate, Scope
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    class _EchoClient:
        def generate_answer(self, q, blocks, model=None):
            return "I'm reading"

        def nli(self, premise, hypothesis):
            return ("entailment", 0.95)

    def _build(flag):
        s = _replace(fresh_settings, rerank_enabled=False, cascade_enabled=False,
                     abstention_threshold=1.0, reader_form_floor_enabled=flag)
        store = RecordStore(s.sqlite_path)

        class _Sub:
            def get(self, h):
                raise KeyError(h)

        r = Retriever(store, object(), KnowledgeGraph(store), _Sub(), _EchoClient(), s)
        r._try_conflict_resolver = lambda *a, **k: None
        r.assemble_context = lambda *a, **k: ["[S0] Tim: I'm reading"]
        r._verify_candidates = lambda cands, text, verify, **kw: (
            [], 0) if not verify else (
            [__import__("eidetic.models", fromlist=["Citation"]).Citation(
                memory_id="m0", content_hash="h0", raw_uri="", source="u", valid_at=1.0,
                nli_label=__import__("eidetic.models", fromlist=["NLILabel"]).NLILabel.ENTAILMENT,
                nli_score=0.95)], 1)
        cands = [RetrievalCandidate(record=MemoryRecord(
            memory_id="m0", content_hash="h0", text="Tim: I'm reading", scope=Scope(),
            valid_at=1.0), dense_score=0.2)]
        return r, cands

    r, cands = _build(True)
    ans = r.answer("What books has Tim read?", verify=True, precomputed=cands)
    assert ans.verified is False
    assert ans.note.startswith("abstained") or "non-responsive" in ans.note

    r2, cands2 = _build(False)                    # kill switch: old behavior
    ans2 = r2.answer("What books has Tim read?", verify=True, precomputed=cands2)
    assert ans2.verified is True


def test_reader_prompt_demands_complete_lists_on_plural_questions(monkeypatch):
    """Slice-2 shape: 'What books has Tim read?' answered with a sentence saying books
    exist. Plural-wh questions instruct the reader to enumerate every distinct item
    across ALL sources; non-plural questions get no such instruction."""
    c = DashScopeClient(replace(get_settings(), reader_cot_enabled=True))
    seen = {}

    def fake_chat_json(model, system, user, **kw):
        seen["system"] = system
        return {"answer": "ok"}

    monkeypatch.setattr(c, "chat_json", fake_chat_json)
    for q in ("What books has Tim read?",
              "Which activities has Maria done with her friends?",
              "What kinds of subjects does Evan enjoy painting?"):
        c.generate_answer(q, ["[S0] ctx"])
        assert "COMPLETE list" in seen["system"], q

    c.generate_answer("What is Jon working on opening?", ["ctx"])
    assert "COMPLETE list" not in seen["system"]
