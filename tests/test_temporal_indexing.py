"""Offline tests for strengthened event/temporal indexing (Phase 5)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from datetime import datetime

import numpy as np

from eidetic.events import EventRecord, event_chain, select_for_query
from eidetic.models import Scope


def _ev(subj, obj, start, mid):
    return EventRecord(subject=subj, verb="did", object=obj, start=start, end=start,
                       source_memory_id=mid, namespace="t", valid_at=start)


def test_event_chain_is_strictly_chronological():
    evs = [_ev("alice", "c", 300, "m3"), _ev("alice", "a", 100, "m1"), _ev("alice", "b", 200, "m2")]
    parsed = {"entities": ["alice"], "operation": "order", "ranges": []}
    assert [e.object for e in event_chain(evs, parsed)] == ["a", "b", "c"]


def test_event_chain_window_caps_length():
    evs = [_ev("alice", str(i), i * 100, f"m{i}") for i in range(10)]
    parsed = {"entities": ["alice"], "operation": "order", "ranges": []}
    assert len(event_chain(evs, parsed, window=3)) == 3


def test_select_for_query_interval_overlap_with_entity():
    may15 = datetime(2023, 5, 15).timestamp()
    jun15 = datetime(2023, 6, 15).timestamp()
    evs = [_ev("alice", "run", may15, "m1"), _ev("alice", "swim", jun15, "m2")]
    parsed = {"entities": ["alice"], "operation": "count",
              "ranges": [{"start": "2023-05-01T00:00:00", "end": "2023-05-31T23:59:59"}]}
    got = select_for_query(evs, parsed)
    assert [e.object for e in got] == ["run"]            # only the May event overlaps the range


# ---- consolidate_pending always indexes events (offline, fake client) ----------------------
class _FakeClient:
    def __init__(self, dim):
        self.dim = dim

    def _embed(self, t):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, t):
        return self._embed(t)

    def embed_texts(self, texts):
        return np.stack([self._embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def extract_edges(self, t):
        return [{"src": "Alice", "relation": "visited", "dst": "Paris", "fact": "Alice visited Paris"}]

    def score_importance(self, t):
        return 0.5


def test_consolidate_pending_always_indexes_events(fresh_settings):
    from eidetic.engine import Engine
    eng = Engine(fresh_settings, client=_FakeClient(fresh_settings.embed_dim))
    scope = Scope(namespace="t")
    eng.ingest_text("Alice visited Paris in May 2023.", scope=scope, consolidate_now=False)
    assert eng.store.events_in_scope("t") == []          # fast write: no events yet
    eng.consolidate_pending(scope=scope, score_importance=False)
    events = eng.store.events_in_scope("t")
    assert events and any(e.source_memory_id for e in events)   # sleep indexed the event


def test_event_chain_context_off_by_default(fresh_settings):
    assert fresh_settings.event_chain_context_enabled is False   # baseline context unchanged
