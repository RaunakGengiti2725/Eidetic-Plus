"""Offline tests for the heuristic write-time memory manager (no key)."""
from __future__ import annotations

import types

import numpy as np

from eidetic.dreaming.manager import Fact, Operation, classify_operation, run_memory_manager


def _f(subj, rel, obj, valid_at=0.0, text="", vec=None):
    return Fact(subject=subj, relation=rel, object=obj, valid_at=valid_at,
                text=text or f"{subj} {rel} {obj}", vec=vec)


def test_noop_on_exact_duplicate():
    existing = [_f("Alice", "works_at", "Acme", valid_at=10)]
    assert classify_operation(_f("alice", "works_at", "acme", valid_at=10), existing) == Operation.NOOP


def test_noop_on_high_cosine_near_duplicate():
    v = np.array([1.0, 0.0])
    existing = [_f("Alice", "likes", "tea", valid_at=5, text="Alice likes tea", vec=v)]
    cand = _f("Alice", "likes", "tea", valid_at=5, text="Alice likes tea", vec=v * 1.001)
    assert classify_operation(cand, existing, dup_cosine=0.97) == Operation.NOOP


def test_update_on_newer_value_for_same_attribute():
    existing = [_f("Alice", "works_at", "Acme", valid_at=10)]
    cand = _f("Alice", "works_at", "Globex", valid_at=20)      # newer, different value
    assert classify_operation(cand, existing) == Operation.UPDATE


def test_add_on_older_historical_value():
    existing = [_f("Alice", "works_at", "Globex", valid_at=20)]
    cand = _f("Alice", "works_at", "Acme", valid_at=10)        # older -> keep as history
    assert classify_operation(cand, existing) == Operation.ADD


def test_delete_tombstone_on_hard_contradiction():
    existing = [_f("Sky", "is", "blue", valid_at=10)]
    cand = _f("Sky", "is", "green", valid_at=11)
    assert classify_operation(cand, existing, contradicts=True) == Operation.DELETE_TOMBSTONE


def test_add_on_novel_fact():
    existing = [_f("Alice", "works_at", "Acme", valid_at=10)]
    assert classify_operation(_f("Bob", "plays", "guitar", valid_at=5), existing) == Operation.ADD


def test_manager_is_a_noop_when_disabled():
    # flags-off: run_memory_manager returns immediately, before any store read or model call.
    fake_engine = types.SimpleNamespace(settings=types.SimpleNamespace(memory_manager_enabled=False))
    assert run_memory_manager(fake_engine) == {"skipped": "disabled"}
