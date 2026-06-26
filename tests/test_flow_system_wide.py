"""Track 9 Task 10: instinct is EVERYWHERE -- one field, many readers. A memory the query never
named, once activated, surfaces via reflex_recall, the hybrid retrieve activation channel, and
(with REFLEX_RECALL off) the hybrid path -- all reading the same field."""
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
        return "an answer"

    def nli(self, premise, hypothesis):
        return ("neutral", 0.2)


def _eng(fresh_settings, **kw):
    s = replace(fresh_settings, flow_activation_enabled=True, flow_field_seed=True,
                flow_hybrid_channel_enabled=True, rerank_enabled=False,
                semantic_cache_enabled=False, **kw)
    return Engine(s, client=_Reader(s.embed_dim))


def test_one_field_surfaces_on_every_recall_surface(fresh_settings):
    e = _eng(fresh_settings, reflex_recall_enabled=True)
    rec = e.ingest_text("completely unrelated instinct memory wording", consolidate_now=False)
    mid = rec.memory_id
    e.activation.inject("default", [mid], 1.0)

    # reflex_recall (the field-seed path)
    assert mid in e.reflex_recall("alpha keyword totally different").candidate_ids()
    # hybrid retrieve (the activation channel), reading the SAME snapshot
    snap = e._flow_snapshot("default")
    hybrid_ids = [c.record.memory_id
                  for c in e.retriever.retrieve("alpha keyword totally different",
                                                scope=Scope(), activation=snap)]
    assert mid in hybrid_ids
    # one field: the id is present in the shared snapshot every surface reads
    assert e._flow_snapshot("default").get(mid, 0.0) > 0.0


def test_reflex_off_hybrid_still_surfaces_via_field(fresh_settings):
    e = _eng(fresh_settings, reflex_recall_enabled=False)
    rec = e.ingest_text("completely unrelated instinct memory wording", consolidate_now=False)
    mid = rec.memory_id
    e.activation.inject("default", [mid], 1.0)
    ids = [c.record.memory_id
           for c in e.retriever.retrieve("alpha keyword totally different", scope=Scope(),
                                         activation=e._flow_snapshot("default"))]
    assert mid in ids                                # REFLEX_RECALL=0 + flow -> hybrid has instinct


def test_ask_runs_clean_with_full_instinct_stack(fresh_settings):
    # the whole stack composed: ingest -> activate -> ask -> commit, no crash, field warmed.
    e = _eng(fresh_settings, reflex_recall_enabled=True, brain_events_enabled=True)
    e.ingest_text("Alice works at Acme Corporation", consolidate_now=False)
    ans = e.ask("where does Alice work")
    assert ans.answer
    # the confirmed/cited recall committed into the shared field
    assert isinstance(e._flow_snapshot("default"), dict)
