"""Offline tests for HaluMem operation-level grading (no key)."""
from __future__ import annotations

import pytest

from bench.halumem import (extraction_f1, extraction_precision, extraction_recall,
                           filter_points_by_split, load, qa_rates, update_accuracy)


def test_extraction_recall_precision_f1():
    gold = ["Alice works at Acme", "Bob likes tea", "Carol lives in Paris"]
    pred = ["alice works at  acme", "Bob likes tea", "Dave plays guitar"]   # 2/3 gold, 1 spurious
    assert abs(extraction_recall(gold, pred) - 2 / 3) < 1e-9
    assert abs(extraction_precision(gold, pred) - 2 / 3) < 1e-9
    assert abs(extraction_f1(gold, pred) - 2 / 3) < 1e-9


def test_extraction_empty_sets():
    assert extraction_recall([], ["x"]) == 0.0
    assert extraction_precision(["x"], []) == 0.0


def test_update_accuracy():
    rows = [{"applied_correct": True}, {"applied_correct": False}, {"applied_correct": True}]
    assert abs(update_accuracy(rows) - 2 / 3) < 1e-9
    assert update_accuracy([]) == 0.0


def test_qa_rates_hallucination_and_omission():
    rows = [
        {"answerable": True, "answered": True, "correct": True},     # good
        {"answerable": True, "answered": True, "correct": False},    # hallucination
        {"answerable": True, "answered": False, "correct": False},   # omission
        {"answerable": False, "answered": False, "correct": True},   # correct abstention
    ]
    r = qa_rates(rows)
    assert r["qa_accuracy"] == 0.5
    assert abs(r["hallucination_rate"] - 0.5) < 1e-9     # 1 wrong of 2 answered
    assert abs(r["omission_rate"] - 1 / 3) < 1e-9        # 1 omitted of 3 answerable


def test_split_routing_respects_the_wall():
    pts = [{"sample_id": f"hm{i}", "point": f"p{i}"} for i in range(200)]
    dev = filter_points_by_split(pts, "dev")
    test = filter_points_by_split(pts, "test")
    dev_ids = {p["sample_id"] for p in dev}
    test_ids = {p["sample_id"] for p in test}
    assert dev_ids.isdisjoint(test_ids)                  # op-level eval respects dev/test
    assert len(dev) + len(test) == len(pts)


def test_loader_is_fail_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)                                   # no mock; raises when data absent
