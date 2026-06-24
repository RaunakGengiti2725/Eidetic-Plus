"""Offline tests for the EvolveMem auto-revert guard (no key)."""
from __future__ import annotations

import json
from pathlib import Path

from bench.guard import (guard_decision, load_champion, pooled_guard_inputs, run_guard,
                         save_champion)


# ---- pure decision logic -----------------------------------------------------
def test_guard_accepts_significant_improvement():
    d = guard_decision(0.50, 0.60, mcnemar_p=0.001, paired_n=100, min_delta_pp=1.0, alpha=0.05)
    assert d["accept"] is True and abs(d["delta_pp"] - 10.0) < 1e-9


def test_guard_rejects_small_delta():
    d = guard_decision(0.50, 0.505, mcnemar_p=0.0001, paired_n=100, min_delta_pp=1.0)
    assert d["accept"] is False and "delta" in d["reason"]


def test_guard_rejects_insignificant():
    d = guard_decision(0.50, 0.62, mcnemar_p=0.20, paired_n=100, min_delta_pp=1.0, alpha=0.05)
    assert d["accept"] is False and "significant" in d["reason"]


def test_guard_rejects_no_paired_items():
    d = guard_decision(0.5, 0.9, mcnemar_p=None, paired_n=0)
    assert d["accept"] is False


def test_guard_significance_can_be_disabled():
    d = guard_decision(0.5, 0.6, mcnemar_p=None, paired_n=50, min_delta_pp=1.0,
                       require_significance=False)
    assert d["accept"] is True


# ---- end-to-end on synthetic dev log dirs ------------------------------------
def _write_logs(d: Path, system: str, correct_ids: set, all_ids: list):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{system}__run0.jsonl", "w") as fh:
        for sid in all_ids:
            fh.write(json.dumps({
                "system": system, "dataset": "locomo", "category": "single_hop",
                "sample_id": sid, "correct": sid in correct_ids, "run_idx": 0,
            }) + "\n")


def test_run_guard_accepts_clear_dev_win(tmp_path):
    ids = [f"q{i}" for i in range(40)]
    champ = tmp_path / "champ"
    chal = tmp_path / "chal"
    _write_logs(champ, "eidetic-plus", set(ids[:20]), ids)        # champion 20/40
    _write_logs(chal, "eidetic-plus", set(ids[:35]), ids)         # challenger 35/40 (superset)
    res = run_guard(champ, chal, system="eidetic-plus", min_delta_pp=1.0, alpha=0.05)
    assert res["accept"] is True
    assert res["unpaired"] == 0                                   # same dev items -> McNemar valid
    assert res["challenger_acc"] > res["champion_acc"]


def test_run_guard_rejects_when_not_significant(tmp_path):
    ids = [f"q{i}" for i in range(40)]
    champ = tmp_path / "champ"
    chal = tmp_path / "chal"
    _write_logs(champ, "eidetic-plus", set(ids[:20]), ids)        # 20/40
    _write_logs(chal, "eidetic-plus", set(ids[:21]), ids)         # 21/40: one extra (b=0,c=1)
    res = run_guard(champ, chal, system="eidetic-plus", min_delta_pp=1.0, alpha=0.05)
    assert res["accept"] is False                                 # one discordant -> not significant


def test_run_guard_refuses_unpaired_dev_sets(tmp_path):
    champ = tmp_path / "champ"
    chal = tmp_path / "chal"
    _write_logs(champ, "eidetic-plus", set(), [f"q{i}" for i in range(0, 40)])
    _write_logs(chal, "eidetic-plus", set(), [f"q{i}" for i in range(5, 45)])  # shifted set
    res = run_guard(champ, chal, system="eidetic-plus")
    assert res["accept"] is False and "unpaired" in res["reason"]   # invalid paired comparison


def test_champion_registry_roundtrip(tmp_path):
    p = tmp_path / "champion.json"
    assert load_champion(p) is None
    save_champion(p, env={"FUSION_METHOD": "dbsf"}, dev_acc=0.62, n=200)
    got = load_champion(p)
    assert got["env"]["FUSION_METHOD"] == "dbsf" and got["dev_acc"] == 0.62


def test_pooled_inputs_flags_unpaired():
    fake = {"comparisons": {"eidetic-plus|locomo|single_hop": {
        "status": "unpaired",
        "control": {"correct": 10, "n": 20}, "experiment": {"correct": 15, "n": 20},
        "paired": {"control_only": 1, "experiment_only": 6, "paired_n": 18,
                   "unpaired_control_rows": 2, "unpaired_experiment_rows": 2},
    }}}
    inp = pooled_guard_inputs(fake)
    assert inp["unpaired"] == 4 and inp["paired_n"] == 18
    assert inp["champion_acc"] == 0.5 and inp["challenger_acc"] == 0.75
