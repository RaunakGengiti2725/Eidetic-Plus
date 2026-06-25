"""Offline tests for deterministic version-aware conflict resolution (Phase 1).

The LLM (here a deterministic fake extractor) only extracts semantically matching candidates;
Python decides freshness from bi-temporal validity. These pin: latest-valid wins, as-of
time-travel, abstain-when-nothing-valid, created_at serial tiebreak, invalidated facts excluded,
and missing valid_at fails loud.
"""
from __future__ import annotations

import pytest

from eidetic.conflicts import resolve_current_value
from eidetic.models import MemoryRecord, RetrievalCandidate, Scope


def _cand(mid, company, valid_at, *, created_at=None, invalid_at=None):
    rec = MemoryRecord(memory_id=mid, content_hash=mid, text=f"Alice works at {company}",
                       scope=Scope(namespace="t"), valid_at=valid_at,
                       created_at=created_at if created_at is not None else valid_at,
                       invalid_at=invalid_at)
    return RetrievalCandidate(record=rec)


def _extract(query, payload):
    # Pure semantic extractor: returns every candidate, never compares timestamps.
    return [{"memory_id": p["memory_id"], "answer": p["text"], "timestamp": p["timestamp"]}
            for p in payload]


_Q = "where does Alice work now"


def test_latest_valid_wins_for_current():
    cands = [_cand("a", "Acme", 100), _cand("b", "Beta", 200), _cand("c", "Gamma", 300)]
    res = resolve_current_value(_Q, cands, _extract)
    assert res and res.records[0].memory_id == "c"            # latest valid = Gamma
    assert set(res.superseded) == {"a", "b"}                  # older values shown, not deleted


def test_as_of_time_travel_picks_value_valid_then():
    cands = [_cand("a", "Acme", 100), _cand("b", "Beta", 200), _cand("c", "Gamma", 300)]
    res = resolve_current_value(_Q, cands, _extract, as_of=250)
    assert res.records[0].memory_id == "b"                    # as of t=250, latest valid is Beta


def test_abstains_when_nothing_valid_as_of():
    cands = [_cand("a", "Acme", 100)]
    res = resolve_current_value(_Q, cands, _extract, as_of=50)  # before any fact was valid
    assert res.abstained and "valid as of" in res.note and res.records == []


def test_created_at_breaks_ties_at_equal_valid_at():
    cands = [_cand("a", "Acme", 100, created_at=10), _cand("b", "Beta", 100, created_at=20)]
    res = resolve_current_value(_Q, cands, _extract)
    assert res.records[0].memory_id == "b"                    # same valid_at -> later serial wins


def test_invalidated_fact_excluded_from_current():
    # Acme was closed (invalid_at=200); as of 250 it is no longer a valid current value.
    cands = [_cand("a", "Acme", 100, invalid_at=200), _cand("b", "Beta", 200)]
    res = resolve_current_value(_Q, cands, _extract, as_of=250)
    assert res.records[0].memory_id == "b"


def test_missing_valid_at_fails_loud():
    rec = MemoryRecord(memory_id="x", content_hash="x", text="Alice works at Acme",
                       scope=Scope(namespace="t"))
    rec.valid_at = None                                       # corrupt/missing timestamp
    with pytest.raises(ValueError, match="valid_at"):
        resolve_current_value(_Q, [RetrievalCandidate(record=rec)], _extract)
