"""Rotating holdout: disjoint, stratified, never-reused test-split slices with a digest ledger."""
from __future__ import annotations

import json
import types

import pytest

from bench.rotating_holdout import draw_slice, slice_digest, stratified_ring


def _samples(n_per_cat: dict[str, int]) -> list:
    out = []
    for cat, n in n_per_cat.items():
        for i in range(n):
            out.append(types.SimpleNamespace(sample_id=f"{cat}_{i}", category=cat))
    return out


def test_ring_is_deterministic_and_complete():
    s = _samples({"temporal": 10, "single_hop": 6, "multi_hop": 4})
    r1 = stratified_ring(s, "locomo", 0)
    r2 = stratified_ring(s, "locomo", 0)
    assert r1 == r2                              # same (dataset, epoch) -> same ring
    assert sorted(r1) == sorted(x.sample_id for x in s)   # a permutation, nothing lost
    assert stratified_ring(s, "locomo", 1) != r1          # new epoch reshuffles


def test_windows_are_category_balanced():
    """Every consecutive window mirrors the corpus mix; no window is one-category clumped."""
    s = _samples({"temporal": 12, "single_hop": 12, "multi_hop": 12})
    ring = stratified_ring(s, "locomo", 0)
    for w in range(3):
        window = ring[w * 12:(w + 1) * 12]
        cats = {sid.rsplit("_", 1)[0] for sid in window}
        assert cats == {"temporal", "single_hop", "multi_hop"}


def test_draws_are_disjoint_and_ledgered(tmp_path):
    s = _samples({"a": 8, "b": 8})
    state = tmp_path / "state.json"
    d1 = draw_slice(s, dataset="locomo", n=4, state_path=state)
    d2 = draw_slice(s, dataset="locomo", n=4, state_path=state)
    assert not set(d1["sample_ids"]) & set(d2["sample_ids"])   # disjoint slices
    ledger = json.loads(state.read_text())["datasets"]["locomo"]["draws"]
    assert [d["digest"] for d in ledger] == [d1["digest"], d2["digest"]]
    # the ledger holds digests only -- raw sample IDs never land in a committed file
    assert "sample_ids" not in ledger[0]
    assert all(sid not in state.read_text() for sid in d1["sample_ids"])


def test_exhaustion_rolls_epoch_and_never_repeats_digest(tmp_path):
    s = _samples({"a": 4, "b": 4})
    state = tmp_path / "state.json"
    digests = set()
    rolled = False
    for _ in range(3):                       # 2 windows/epoch of n=4 -> third draw must roll over
        d = draw_slice(s, dataset="locomo", n=4, state_path=state)
        assert d["digest"] not in digests    # a used slice is never re-reported
        digests.add(d["digest"])
        rolled = rolled or d["rollover"]
    assert rolled                            # the rollover was recorded, not silent


def test_oversized_slice_fails_loud(tmp_path):
    s = _samples({"a": 3})
    with pytest.raises(ValueError, match="exceeds test split size"):
        draw_slice(s, dataset="locomo", n=5, state_path=tmp_path / "state.json")


def test_samples_file_matches_bench_format(tmp_path):
    s = _samples({"a": 6})
    out = tmp_path / "holdout.samples.json"
    d = draw_slice(s, dataset="locomo", n=3, state_path=tmp_path / "state.json", out_path=out)
    rows = json.loads(out.read_text())
    assert rows == [{"dataset": "locomo", "sample_id": sid} for sid in d["sample_ids"]]
