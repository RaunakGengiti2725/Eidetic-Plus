"""Offline tests for the MemMA self-repair deterministic core (no key)."""
from __future__ import annotations

import types

from eidetic.dreaming.repair import (Diagnosis, RepairAction, diagnose, route_repair,
                                     run_sweep, select_repair_targets)


def test_diagnose_branches():
    # a passed probe needs no repair regardless of the other signals
    assert diagnose(True, coverage=0.1, contradicted=True, abstention_threshold=0.4) == Diagnosis.PASSED
    # contradiction dominates a failed probe
    assert diagnose(False, coverage=0.9, contradicted=True, abstention_threshold=0.4) == Diagnosis.CONTRADICTED
    # weak evidence -> the info is missing
    assert diagnose(False, coverage=0.2, contradicted=False, abstention_threshold=0.4) == Diagnosis.MISSING
    # present-but-unentailed evidence -> hard to retrieve
    assert diagnose(False, coverage=0.7, contradicted=False, abstention_threshold=0.4) == Diagnosis.HARD_TO_RETRIEVE


def test_route_maps_diagnosis_to_memma_action():
    assert route_repair(Diagnosis.PASSED) == RepairAction.SKIP
    assert route_repair(Diagnosis.MISSING) == RepairAction.INSERT
    assert route_repair(Diagnosis.HARD_TO_RETRIEVE) == RepairAction.MERGE
    assert route_repair(Diagnosis.CONTRADICTED) == RepairAction.MERGE


def test_select_repair_targets_ranks_by_anomaly():
    recs = [types.SimpleNamespace(memory_id=f"m{i}") for i in range(5)]
    anomaly = {"m0": 0.1, "m1": 0.9, "m2": 0.5, "m3": 0.05, "m4": 0.7}
    targets = select_repair_targets(recs, anomaly, topk=3)
    assert [t.memory_id for t in targets] == ["m1", "m4", "m2"]   # highest anomaly first


def test_select_targets_falls_back_to_order_without_anomaly():
    recs = [types.SimpleNamespace(memory_id=f"m{i}") for i in range(3)]
    assert [t.memory_id for t in select_repair_targets(recs, {}, topk=2)] == ["m0", "m1"]


def test_run_sweep_is_a_noop_when_disabled():
    fake_engine = types.SimpleNamespace(settings=types.SimpleNamespace(dream_repair_enabled=False))
    assert run_sweep(fake_engine) == {"skipped": "disabled"}


def test_engine_dream_repair_noop_when_disabled(engine):
    # default settings: DREAM_REPAIR off -> dream_repair returns skipped with NO model call.
    assert engine.settings.dream_repair_enabled is False
    assert engine.dream_repair() == {"skipped": "disabled"}
