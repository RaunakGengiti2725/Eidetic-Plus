"""Offline tests for strengthened event/temporal indexing (Phase 5)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from datetime import datetime

import numpy as np

from eidetic.events import EventRecord, event_chain, normalize_dates, select_for_query
from eidetic.models import MemoryRecord, Scope
from eidetic.store import RecordStore


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


def test_events_in_scope_requires_named_source_memory_when_scoped(tmp_path):
    store = RecordStore(tmp_path / "events.sqlite")
    scope = Scope(namespace="t")
    may4 = datetime(2026, 5, 4).timestamp()
    store.add_event(EventRecord(
        subject="Alice", verb="attended", object="conference",
        fact="Alice attended conference",
        start=may4, end=may4, namespace="t", valid_at=may4,
        source_memory_id="missing-source",
    ))

    assert store.events_in_scope("t")
    assert store.events_in_scope("t", scope=scope, at=may4) == []


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
        low = t.lower()
        if "visited paris" in low:
            return [{"src": "Alice", "relation": "visited", "dst": "Paris", "fact": "Alice visited Paris"}]
        if "conference" in low:
            return [{
                "src": "Alice",
                "relation": "attended",
                "dst": "conference",
                "fact": "Alice attended the conference",
            }]
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


def test_consolidate_pending_resolves_event_relative_dates_from_prior_events(fresh_settings):
    from eidetic.engine import Engine
    eng = Engine(fresh_settings, client=_FakeClient(fresh_settings.embed_dim))
    scope = Scope(namespace="t")
    eng.ingest_text(
        "Alice attended the conference on May 4, 2026.",
        scope=scope,
        valid_at=datetime(2026, 5, 4, 12).timestamp(),
        consolidate_now=False,
    )
    eng.ingest_text(
        "The week after the conference, Alice visited Paris.",
        scope=scope,
        valid_at=datetime(2026, 6, 1, 12).timestamp(),
        consolidate_now=False,
    )
    eng.consolidate_pending(scope=scope, score_importance=False)
    events = eng.store.events_in_scope("t")
    paris = next(e for e in events if e.object == "Paris")
    assert datetime.fromtimestamp(paris.start).isoformat() == "2026-05-11T00:00:00"
    assert datetime.fromtimestamp(paris.end).isoformat() == "2026-05-17T23:59:59"


def test_consolidate_pending_resolves_counted_event_relative_dates(fresh_settings):
    from eidetic.engine import Engine
    eng = Engine(fresh_settings, client=_FakeClient(fresh_settings.embed_dim))
    scope = Scope(namespace="t")
    eng.ingest_text(
        "Alice attended the conference on May 4, 2026.",
        scope=scope,
        valid_at=datetime(2026, 5, 4, 12).timestamp(),
        consolidate_now=False,
    )
    eng.ingest_text(
        "Two days after the conference, Alice visited Paris.",
        scope=scope,
        valid_at=datetime(2026, 6, 1, 12).timestamp(),
        consolidate_now=False,
    )
    eng.consolidate_pending(scope=scope, score_importance=False)
    events = eng.store.events_in_scope("t")
    paris = next(e for e in events if e.object == "Paris")
    assert datetime.fromtimestamp(paris.start).isoformat() == "2026-05-06T00:00:00"
    assert datetime.fromtimestamp(paris.end).isoformat() == "2026-05-06T23:59:59"


def test_consolidate_pending_uses_local_dates_for_multiple_events_in_one_memory(fresh_settings):
    from eidetic.engine import Engine

    class MultiDateClient(_FakeClient):
        def extract_edges(self, t):
            return [
                {
                    "src": "Alice",
                    "relation": "attended",
                    "dst": "robotics conference",
                    "fact": "Alice attended the robotics conference",
                },
                {
                    "src": "Alice",
                    "relation": "attended",
                    "dst": "medical conference",
                    "fact": "Alice attended the medical conference",
                },
            ]

    eng = Engine(fresh_settings, client=MultiDateClient(fresh_settings.embed_dim))
    scope = Scope(namespace="t")
    eng.ingest_text(
        "Alice attended the robotics conference on May 4, 2026. "
        "Alice attended the medical conference on May 20, 2026.",
        scope=scope,
        valid_at=datetime(2026, 6, 1, 12).timestamp(),
        consolidate_now=False,
    )
    eng.consolidate_pending(scope=scope, score_importance=False)
    events = {e.object: e for e in eng.store.events_in_scope("t")}
    assert datetime.fromtimestamp(events["robotics conference"].start).isoformat() == "2026-05-04T00:00:00"
    assert datetime.fromtimestamp(events["medical conference"].start).isoformat() == "2026-05-20T00:00:00"


def test_consolidate_pending_resolves_same_memory_event_relative_date(fresh_settings):
    from eidetic.engine import Engine

    class SameRecordChainClient(_FakeClient):
        def extract_edges(self, t):
            return [
                {
                    "src": "Alice",
                    "relation": "attended",
                    "dst": "conference",
                    "fact": "Alice attended the conference",
                },
                {
                    "src": "Alice",
                    "relation": "visited",
                    "dst": "Paris",
                    "fact": "Alice visited Paris",
                },
            ]

    eng = Engine(fresh_settings, client=SameRecordChainClient(fresh_settings.embed_dim))
    scope = Scope(namespace="t")
    eng.ingest_text(
        "Alice attended the conference on May 4, 2026. "
        "The week after the conference, Alice visited Paris.",
        scope=scope,
        valid_at=datetime(2026, 6, 1, 12).timestamp(),
        consolidate_now=False,
    )
    eng.consolidate_pending(scope=scope, score_importance=False)
    paris = next(e for e in eng.store.events_in_scope("t") if e.object == "Paris")
    assert datetime.fromtimestamp(paris.start).isoformat() == "2026-05-11T00:00:00"
    assert datetime.fromtimestamp(paris.end).isoformat() == "2026-05-17T23:59:59"


def test_consolidate_pending_event_aliases_disambiguate_anchor_events(fresh_settings):
    from eidetic.engine import Engine

    class AliasClient(_FakeClient):
        def extract_edges(self, t):
            low = t.lower()
            if "robotics conference" in low:
                return [{
                    "src": "Alice",
                    "relation": "attended",
                    "dst": "conference",
                    "fact": "Alice attended conference",
                }]
            if "medical conference" in low:
                return [{
                    "src": "Alice",
                    "relation": "attended",
                    "dst": "conference",
                    "fact": "Alice attended conference",
                }]
            return super().extract_edges(t)

    settings = replace(fresh_settings, event_alias_expansion_enabled=True)
    eng = Engine(settings, client=AliasClient(settings.embed_dim))
    scope = Scope(namespace="t")
    eng.ingest_text(
        "Alice attended the annual robotics conference on May 4, 2026.",
        scope=scope,
        valid_at=datetime(2026, 5, 4, 12).timestamp(),
        consolidate_now=False,
    )
    eng.ingest_text(
        "Alice attended the medical conference on May 20, 2026.",
        scope=scope,
        valid_at=datetime(2026, 5, 20, 12).timestamp(),
        consolidate_now=False,
    )
    eng.consolidate_pending(scope=scope, score_importance=False)
    events = eng.store.events_in_scope("t")
    robotics = min(events, key=lambda e: e.start or 0)
    assert "annual robotics conference" in robotics.aliases
    got = {
        d["expr"]: (d["start"], d["end"])
        for d in normalize_dates(
            "the week after the annual robotics conference",
            datetime(2026, 6, 23, 12).timestamp(),
            events,
        )
    }
    assert got["the week after the annual robotics conference"] == (
        "2026-05-11T00:00:00",
        "2026-05-17T23:59:59",
    )


def test_event_chain_context_off_by_default(fresh_settings):
    assert fresh_settings.event_chain_context_enabled is False   # baseline context unchanged


def test_assemble_context_resolves_event_relative_timeline(tmp_path, fresh_settings):
    from eidetic.graph import KnowledgeGraph
    from eidetic.retrieval import Retriever
    from eidetic.store import RecordStore

    store = RecordStore(tmp_path / "db.sqlite")
    scope = Scope(namespace="t")
    may4 = datetime(2026, 5, 4).timestamp()
    may12 = datetime(2026, 5, 12).timestamp()
    store.upsert_record(MemoryRecord(
        memory_id="m1", content_hash="m1", text="Alice attended the conference",
        scope=scope, valid_at=may4,
    ))
    store.upsert_record(MemoryRecord(
        memory_id="m2", content_hash="m2", text="Alice visited Paris",
        scope=scope, valid_at=may12,
    ))
    store.add_event(EventRecord(
        subject="Alice", verb="attended", object="the conference",
        fact="Alice attended the conference",
        aliases=["work summit"],
        start=may4, end=may4, namespace="t", valid_at=may4, source_memory_id="m1",
    ))
    store.add_event(EventRecord(
        subject="Alice", verb="visited", object="Paris",
        fact="Alice visited Paris",
        start=may12, end=may12, namespace="t", valid_at=may12, source_memory_id="m2",
    ))
    settings = replace(fresh_settings, event_chain_context_enabled=True)
    retriever = Retriever(store, object(), KnowledgeGraph(store), object(), object(), settings)
    blocks = retriever.assemble_context(
        "What did Alice do the week after the conference?",
        [],
        at=datetime(2026, 6, 23).timestamp(),
        scope=scope,
    )
    joined = "\n".join(blocks)
    assert "Alice visited Paris" in joined
    assert "Event timeline (chronological): Alice visited Paris" in joined
    assert "Alice attended the conference" not in joined
