"""Offline tests for the ActivationField: the per-namespace working-memory substrate. Pure local
math -- decay, additive clamped inject, bounded eviction, namespace isolation, and salience-aware
decay (a salient memory fades slower; salience is access-time only, never memory age)."""
from __future__ import annotations

from eidetic.activation import ActivationField


def test_inject_and_get():
    f = ActivationField(decay=0.85, cap=100)
    f.inject("ns", ["a", "b"], amount=1.0)
    s = f.snapshot("ns")
    assert s["a"] == 1.0 and s["b"] == 1.0
    assert f.get("ns", "a") == 1.0


def test_decay_multiplies_then_prunes_floor():
    f = ActivationField(decay=0.5, floor=0.1, cap=100)
    f.inject("ns", ["a"], amount=1.0)
    f.decay("ns")                          # 1.0 -> 0.5
    assert round(f.snapshot("ns")["a"], 3) == 0.5
    for _ in range(3):                     # 0.5 -> 0.25 -> 0.125 -> 0.0625 < floor
        f.decay("ns")
    assert "a" not in f.snapshot("ns")     # pruned below floor


def test_inject_is_additive_and_clamped():
    f = ActivationField(decay=0.9, cap=100)
    f.inject("ns", ["a"], amount=0.8)
    f.inject("ns", ["a"], amount=0.8)
    assert f.snapshot("ns")["a"] == 1.0    # clamped to 1.0


def test_cap_evicts_lowest():
    f = ActivationField(decay=0.9, cap=2)
    f.inject("ns", ["a"], amount=0.2)
    f.inject("ns", ["b"], amount=0.5)
    f.inject("ns", ["c"], amount=0.9)      # evicts 'a' (lowest)
    assert set(f.snapshot("ns")) == {"b", "c"}


def test_namespace_isolation():
    f = ActivationField(decay=0.9, cap=100)
    f.inject("A", ["x"], amount=1.0)
    assert f.snapshot("B") == {}
    assert f.get("B", "x") == 0.0


def test_empty_and_noop_inputs_are_safe():
    f = ActivationField()
    f.inject("ns", [])                     # no ids
    f.inject("ns", [None, ""])             # falsy ids skipped
    f.decay("ns")                          # decay an empty namespace
    assert f.snapshot("ns") == {}


def test_salience_decay_slows_a_salient_memory():
    f = ActivationField(decay=0.5, floor=0.1, cap=100)
    f.inject("ns", ["hot", "cold"], amount=1.0)
    sal = lambda mid: 1.0 if mid == "hot" else 0.0      # hot is maximally salient
    for _ in range(4):
        f.decay("ns", salience=sal)
    s = f.snapshot("ns")
    assert s.get("hot", 0.0) > s.get("cold", 0.0)        # salient memory outlives the cold one
    assert "cold" not in s                                # cold decayed below floor


def test_salience_none_is_plain_decay():
    f = ActivationField(decay=0.5, cap=100)
    f.inject("ns", ["a"], amount=1.0)
    f.decay("ns", salience=None)
    assert round(f.snapshot("ns")["a"], 3) == 0.5        # identical to factor-only decay
