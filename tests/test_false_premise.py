"""Track 3.3 false-premise abstention: don't answer a question whose presupposed relationship is
unsupported by memory. Deterministic + no model call on the abstain path (the fake reader RAISES
if the reader/NLI is touched, proving the gate short-circuits before any model call)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.models import BrainEventType


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


class _RaiseReader(_Embed):
    def generate_answer(self, *a, **k):
        raise AssertionError("reader was called on a false-premise abstain (should short-circuit)")

    def nli(self, *a, **k):
        raise AssertionError("nli was called on a false-premise abstain (should short-circuit)")


class _OkReader(_Embed):
    def generate_answer(self, q, blocks, model=None):
        return "I'm not sure."

    def nli(self, premise, hypothesis):
        return ("neutral", 0.2)


def _engine(fresh_settings, reader_cls=_RaiseReader, **kw):
    s = replace(fresh_settings, false_premise_enabled=True, semantic_cache_enabled=False, **kw)
    return Engine(s, client=reader_cls(s.embed_dim))


def test_check_flags_disconnected_entities(fresh_settings):
    e = _engine(fresh_settings)
    e.ingest_text("Alice is a software engineer at Microsoft", consolidate_now=False)
    fp = e.check_false_premise("Why did Alice leave Google?")
    assert fp is not None and fp["category"] == "missing_premise"
    assert "Alice" in fp["entities"] and "Google" in fp["entities"]


def test_check_passes_when_entities_co_occur_in_a_memory(fresh_settings):
    e = _engine(fresh_settings)
    e.ingest_text("Alice worked at Google for five years before leaving", consolidate_now=False)
    assert e.check_false_premise("Why did Alice leave Google?") is None   # premise supported


def test_check_passes_when_a_graph_edge_connects_the_entities(fresh_settings):
    from eidetic.models import Scope
    e = _engine(fresh_settings)
    rec = e.ingest_text("a record", consolidate_now=False)
    e.graph.add_fact("Alice", "worked_at", "Google", fact="Alice worked at Google",
                     source_memory_id=rec.memory_id, scope=Scope())
    assert e.check_false_premise("Why did Alice leave Google?") is None   # edge connects them


def test_check_skips_single_entity_questions(fresh_settings):
    e = _engine(fresh_settings)
    e.ingest_text("Bob enjoys hiking", consolidate_now=False)
    assert e.check_false_premise("Why did Alice leave?") is None          # <2 entities -> no check


def test_check_is_case_insensitive_against_stored_memory(fresh_settings):
    e = _engine(fresh_settings)
    e.ingest_text("alice worked at google last year", consolidate_now=False)  # lowercase storage
    assert e.check_false_premise("Why did Alice leave Google?") is None       # still matches


def test_ask_abstains_on_false_premise_with_no_model_call(fresh_settings):
    e = _engine(fresh_settings, brain_events_enabled=True)
    e.ingest_text("Alice is a software engineer at Microsoft", consolidate_now=False)
    ans = e.ask("Why did Alice leave Google?")   # _RaiseReader raises if any model call happens
    assert ans.note.startswith("abstained: false-premise")
    assert ans.verified is False and ans.confidence == 0.0
    assert e.brain_log.by_type(BrainEventType.ANSWER_ABSTAINED)


def test_flag_off_has_no_false_premise_gate(fresh_settings):
    e = Engine(replace(fresh_settings, false_premise_enabled=False, semantic_cache_enabled=False,
                       rerank_enabled=False), client=_OkReader(fresh_settings.embed_dim))
    e.ingest_text("Alice is a software engineer at Microsoft", consolidate_now=False)
    ans = e.ask("Why did Alice leave Google?")
    assert not ans.note.startswith("abstained: false-premise")   # gate off -> normal path runs
