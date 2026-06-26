"""Track 9 Task 4: Flow hub wired through ask + reflex_recall + ingest. One decay per ask (no
double-decay via the nested reflex_recall), confirmed recall commits to the field, the field is
read on the next turn, and flow works even with REFLEX_RECALL off (hybrid reads it)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.models import Scope


class _Reader:
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

    def generate_answer(self, q, blocks, model=None):
        return "Alice works at Acme Corporation"

    def nli(self, premise, hypothesis):
        return ("entailment", 0.9) if "acme" in (premise or "").lower() else ("neutral", 0.2)


def _eng(fresh_settings, **kw):
    s = replace(fresh_settings, semantic_cache_enabled=False, rerank_enabled=False, **kw)
    return Engine(s, client=_Reader(s.embed_dim))


def test_ask_commits_confirmed_recall_into_field(fresh_settings):
    e = _eng(fresh_settings, flow_activation_enabled=True, reflex_recall_enabled=True)
    e.ingest_text("Alice works at Acme Corporation", consolidate_now=False)
    ans = e.ask("where does Alice work")
    assert ans.verified is True
    assert any(v > 0.0 for v in e._flow_snapshot("default").values())   # field warmed by the recall


def test_one_ask_decays_field_exactly_once(fresh_settings, monkeypatch):
    # landmine: ask() begins the turn AND calls reflex_recall internally -> must decay only ONCE.
    e = _eng(fresh_settings, flow_activation_enabled=True, reflex_recall_enabled=True, flow_decay=0.5)
    e.ingest_text("Alice works at Acme Corporation", consolidate_now=False)
    calls = {"n": 0}
    orig = e.activation.decay

    def _counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(e.activation, "decay", _counted)
    e.ask("where does Alice work")
    assert calls["n"] == 1


def test_flow_commits_even_with_reflex_off(fresh_settings):
    # REFLEX_RECALL=0: hybrid is the reader, but the field is still committed (one writer).
    e = _eng(fresh_settings, flow_activation_enabled=True, reflex_recall_enabled=False)
    e.ingest_text("Alice works at Acme Corporation", consolidate_now=False)
    ans = e.ask("where does Alice work")
    assert ans.answer
    assert any(v > 0.0 for v in e._flow_snapshot("default").values())


def test_reflex_recall_field_seeds_activated_unnamed_memory(fresh_settings):
    e = _eng(fresh_settings, flow_activation_enabled=True, flow_field_seed=True,
             reflex_recall_enabled=True)
    rec = e.ingest_text("completely different beta wording", consolidate_now=False)
    e.activation.inject("default", [rec.memory_id], 1.0)
    p = e.reflex_recall("alpha keyword totally unrelated")
    assert rec.memory_id in p.candidate_ids()                 # instinct surfaced it system-wide
    assert p.scores[rec.memory_id].match_strength == 0.0      # but coverage stays content-only


def test_flow_off_ask_does_not_build_field(fresh_settings):
    e = _eng(fresh_settings, flow_activation_enabled=False, reflex_recall_enabled=True)
    e.ingest_text("Alice works at Acme Corporation", consolidate_now=False)
    e.ask("where does Alice work")
    assert e.activation is None                               # flag-off: no field at all
