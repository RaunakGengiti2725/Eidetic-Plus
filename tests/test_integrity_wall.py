"""The benchmark integrity wall: no optimizer may ever read a test item.

These tests are fully offline (no key). They prove the two halves of the wall:
  * the deterministic dev/test split partition is stable and disjoint, and
  * the feedback buffer hands learners dev rows ONLY -- a benchmark namespace's
    feedback is recorded for audit but is unreachable by any learner.
"""
from __future__ import annotations

import numpy as np

from bench.datasets import (DEV_SPLIT_PCT, Sample, filter_split, split_of)
from eidetic.feedback import FeedbackBuffer, is_benchmark_namespace


def _sample(sid: str) -> Sample:
    return Sample(sample_id=sid, sessions=[], question="q", gold="g",
                  category="c", dataset="locomo")


def test_split_is_deterministic_and_stable():
    ids = [f"q{i}" for i in range(500)]
    first = {sid: split_of(sid) for sid in ids}
    # Re-computing gives identical assignments (stable across calls / machines).
    assert all(split_of(sid) == first[sid] for sid in ids)
    assert set(first.values()) <= {"dev", "test"}


def test_dev_and_test_are_disjoint_and_cover_everything():
    ids = [f"sample-{i}" for i in range(2000)]
    samples = [_sample(s) for s in ids]
    dev = {s.sample_id for s in filter_split(samples, "dev")}
    test = {s.sample_id for s in filter_split(samples, "test")}
    assert dev.isdisjoint(test)                 # the wall: no overlap
    assert dev | test == set(ids)               # and nothing is lost
    # Roughly DEV_SPLIT_PCT of items land on dev (within a tolerance for 2000 draws).
    frac = len(dev) / len(ids)
    assert abs(frac - DEV_SPLIT_PCT / 100) < 0.05


def test_filter_split_passthrough_and_rejects_typo():
    samples = [_sample(f"x{i}") for i in range(10)]
    assert len(filter_split(samples, None)) == 10
    assert len(filter_split(samples, "all")) == 10
    try:
        filter_split(samples, "deva")
    except ValueError:
        pass
    else:
        raise AssertionError("filter_split must reject an unknown split, not silently leak test")


def test_benchmark_namespace_detection():
    # Harness pattern {system}-{dataset}-g{n}-r{n} and bare dataset tokens are benchmark.
    assert is_benchmark_namespace("eidetic-plus-locomo-g0-r0")
    assert is_benchmark_namespace("mem0-longmemeval-g3-r9")
    assert is_benchmark_namespace("beam-g1-r0")
    # Real user/production namespaces are NOT benchmark.
    assert not is_benchmark_namespace("user-42")
    assert not is_benchmark_namespace("acme/project-x")
    assert not is_benchmark_namespace("default")


def test_feedback_buffer_hides_benchmark_rows_from_learners(tmp_path):
    fb = FeedbackBuffer(tmp_path / "feedback.sqlite")
    # A real (production/dev) namespace -> learnable.
    fb.append("user-7", "where do I live?", {"coverage": 0.8}, arm="rrf", reward=1.0,
              qvec=np.array([1.0, 0.0], dtype=np.float32))
    # A benchmark namespace -> recorded for audit, but must be UNREACHABLE by a learner.
    fb.append("eidetic-plus-locomo-g0-r0", "benchmark q", {"coverage": 0.9}, arm="rrf",
              reward=1.0)

    learnable = fb.sample(limit=100)
    assert len(learnable) == 1
    assert learnable[0].namespace == "user-7"
    assert learnable[0].is_dev == 1
    # qvec round-trips.
    assert learnable[0].qvec is not None and learnable[0].qvec.shape == (2,)

    # The benchmark row exists for audit (count includes it via dev_only=False) but is
    # excluded from the learnable count.
    assert fb.count(dev_only=True) == 1
    assert fb.count(dev_only=False) == 2

    # arm_stats aggregates dev rows only.
    stats = fb.arm_stats()
    assert stats["rrf"]["n"] == 1.0


def test_feedback_clear_is_scoped(tmp_path):
    fb = FeedbackBuffer(tmp_path / "feedback.sqlite")
    fb.append("user-1", "q1", {}, reward=1.0)
    fb.append("user-2", "q2", {}, reward=0.0)
    fb.clear("user-1")
    rows = fb.sample(limit=10)
    assert {r.namespace for r in rows} == {"user-2"}
