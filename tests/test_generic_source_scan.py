"""Dataset-neutral structured-recall invariants.

These tests intentionally use synthetic memories, not benchmark examples. The product can keep
answering narrow recall through SMQE without benchmark-shaped source scans.
"""
from __future__ import annotations

from datetime import datetime

from eidetic.models import MemoryRecord, Scope
from eidetic.retrieval import _structured_memory_answer, structured_record_recall


class _Store:
    def __init__(self, records):
        self.records = records

    def active_records_at(self, at, scope):
        return [rec for rec in self.records if rec.is_active_at(at) and rec.scope.visible_to(scope)]

    def active_claims_at(self, at, scope):
        return []

    def get_record(self, memory_id):
        for rec in self.records:
            if rec.memory_id == memory_id:
                return rec
        return None


class _Retriever:
    def __init__(self, records):
        self.store = _Store(records)

    def verify_citation(self, rec, atom):
        raise AssertionError("verify_citation should not run when verify=False")


def _rec(text: str, *, memory_id: str = "m1", valid_at: float = 1_700_000_000.0) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        content_hash=f"h-{memory_id}",
        text=text,
        source="test",
        scope=Scope(namespace="scan-test"),
        valid_at=valid_at,
    )


def test_structured_record_recall_extracts_unseen_degree_slot():
    answer, supports = structured_record_recall(
        "What degree did I graduate with?",
        [_rec("assistant: Any school history?\nuser: I graduated with a degree in Astrobiology.")],
        1_700_000_010.0,
    )

    assert answer == "Astrobiology"
    assert supports
    assert "degree in Astrobiology" in supports[0][1]


def test_structured_record_recall_extracts_unseen_place_slot():
    answer, supports = structured_record_recall(
        "Where did I redeem my coffee coupon?",
        [_rec("user: I redeemed my coffee coupon at Moon Market after work.")],
        1_700_000_010.0,
    )

    assert answer == "Moon Market"
    assert supports
    assert "Moon Market" in supports[0][1]


def test_structured_record_recall_computes_elapsed_days_from_source_date():
    session_time = datetime(2026, 3, 12, 9, 0).timestamp()
    question_time = datetime(2026, 3, 20, 9, 0).timestamp()
    answer, supports = structured_record_recall(
        "How many days ago did I visit the pottery fair?",
        [_rec("user: I visited the pottery fair yesterday.", valid_at=session_time)],
        question_time,
    )

    assert answer == "9"
    assert supports
    assert "pottery fair" in supports[0][1]


def test_structured_memory_answer_uses_smqe():
    rec = _rec("user: I graduated with a degree in Astrobiology.")
    ans = _structured_memory_answer(
        _Retriever([rec]),
        "What degree did I graduate with?",
        [rec],
        1_700_000_010.0,
        verify=False,
    )

    assert ans is not None
    assert ans.answer == "Astrobiology"
    assert ans.note.startswith("smqe:")


def test_legacy_dataset_scan_env_var_has_no_effect(monkeypatch):
    monkeypatch.delenv("EIDETIC_ENABLE_DATASET_SOURCE_SCANS", raising=False)
    query = "Would I probably prefer C. S. Lewis over John Green?"
    rec = _rec("user: Harry Potter is my favorite book.")

    baseline = _structured_memory_answer(
        _Retriever([rec]), query, [rec], 1_700_000_010.0, verify=False
    )

    monkeypatch.setenv("EIDETIC_ENABLE_DATASET_SOURCE_SCANS", "1")
    enabled = _structured_memory_answer(
        _Retriever([rec]), query, [rec], 1_700_000_010.0, verify=False
    )

    assert (enabled.answer if enabled else None) == (baseline.answer if baseline else None)
    assert enabled is None or enabled.note.startswith("smqe:")


def test_removed_source_scan_names_are_not_exported():
    import eidetic.retrieval as retrieval

    assert not hasattr(retrieval, "generic_" + "source_scan")
    assert not hasattr(retrieval, "_product_" + "source_scan_answer")
