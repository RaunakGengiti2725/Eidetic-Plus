"""Track 9 Task 2.5: the Engine Flow hub -- one writer (commit), many readers (snapshot). Inject
once, read the identical snapshot everywhere; one-hop spread; namespace isolation; single warm-state
(hotset retired when flow on); flag-off no-op."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace

import numpy as np

from eidetic.engine import Engine
from eidetic.models import MemoryRecord, Scope


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


def _eng(fresh_settings, **kw):
    s = replace(fresh_settings, **kw)
    return Engine(s, client=_Embed(s.embed_dim))


def test_flow_off_field_is_none(fresh_settings):
    e = _eng(fresh_settings, flow_activation_enabled=False)
    assert e.activation is None
    assert e._flow_snapshot("ns") is None
    e._flow_commit_recall("ns", ["m1"], Scope(namespace="ns"), 1.0)   # no-op, no crash


def test_commit_injects_and_snapshot_reads(fresh_settings):
    e = _eng(fresh_settings, flow_activation_enabled=True)
    e._flow_commit_recall("ns", ["m1", "m2"], Scope(namespace="ns"), 1.0)
    snap = e._flow_snapshot("ns")
    assert snap["m1"] > 0.0 and snap["m2"] > 0.0


def test_commit_spreads_one_hop_weaker(fresh_settings):
    e = _eng(fresh_settings, flow_activation_enabled=True, flow_inject_confirmed=1.0,
             flow_spread_factor=0.4)
    ns = Scope(namespace="ns")
    e.store.upsert_record(MemoryRecord(memory_id="m1", content_hash="h1", text="x", scope=ns, valid_at=1.0))
    e.store.upsert_record(MemoryRecord(memory_id="m2", content_hash="h2", text="y", scope=ns, valid_at=1.0))
    e.graph.link_memories(["m1", "m2"], scope=ns, valid_at=1.0)
    e._flow_commit_recall("ns", ["m1"], ns, 1.0)
    snap = e._flow_snapshot("ns")
    assert snap["m1"] == 1.0
    assert 0.0 < snap.get("m2", 0.0) < 1.0           # neighbor got weaker spread activation


def test_namespace_isolation(fresh_settings):
    e = _eng(fresh_settings, flow_activation_enabled=True)
    e._flow_commit_recall("A", ["m1"], Scope(namespace="A"), 1.0)
    assert e._flow_snapshot("B") == {}


def test_single_warm_state_hotset_retired_when_flow_on(fresh_settings):
    e = _eng(fresh_settings, flow_activation_enabled=True)
    e._touch_hotset("ns", ["ghost"])                 # retired -> no-op when flow on
    assert not e._hotset.get("ns")
    e._flow_commit_recall("ns", ["m1"], Scope(namespace="ns"), 1.0)
    assert "m1" in e._hotset_ids("ns")               # _hotset_ids now reads the field


def test_begin_turn_decays_once(fresh_settings):
    e = _eng(fresh_settings, flow_activation_enabled=True, flow_decay=0.5)
    e._flow_commit_recall("ns", ["m1"], Scope(namespace="ns"), 1.0)
    e._flow_begin_turn("ns", "q", scope=Scope(namespace="ns"), as_of=None)
    assert round(e._flow_snapshot("ns")["m1"], 3) == 0.5


def test_flow_off_hotset_unchanged(fresh_settings):
    e = _eng(fresh_settings, flow_activation_enabled=False, reflex_recall_enabled=True)
    e._touch_hotset("ns", ["m1"])
    assert "m1" in e._hotset_ids("ns")               # legacy binary hotset intact when flow off
