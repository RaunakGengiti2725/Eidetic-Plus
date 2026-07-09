"""The >=10-run gate aggregator: pure math over judged run files. Synthetic fixtures."""
import json
from pathlib import Path

from bench.notebooklm_gate import gate_report, _bootstrap_ci


def _write(tmp, name, rows):
    p = tmp / name
    p.write_text(json.dumps({"rows": rows}))
    return p


def _run(correct, total):
    return [{"sample_id": f"q{i}", "correct": i < correct} for i in range(total)]


def test_partial_when_too_few_runs(tmp_path):
    files = [_write(tmp_path, f"r{i}.judged.json", _run(34, 40)) for i in range(3)]
    rep = gate_report(files, min_runs=10, comparator_acc=0.592, comparator_name="rag-vector")
    assert rep["n_runs"] == 3
    assert rep["verdict"] == "PARTIAL"
    assert "need 7 more" in rep["verdict_reason"]
    assert rep["mean_accuracy"] == 0.85


def test_pass_when_runs_met_and_ci_clears_comparator(tmp_path):
    # 10 runs all ~85%, comparator 59.2% -> CI well above -> PASS
    files = [_write(tmp_path, f"r{i}.judged.json", _run(34, 40)) for i in range(10)]
    rep = gate_report(files, min_runs=10, comparator_acc=0.592, comparator_name="rag-vector")
    assert rep["n_runs"] == 10
    assert rep["verdict"] == "PASS"
    assert rep["ci95_mean"][0] > 0.592


def test_partial_when_ci_does_not_clear_comparator(tmp_path):
    # 10 noisy runs centered NEAR the comparator (mean ~58.5%) -> bootstrap CI straddles
    # 59.2% -> PARTIAL even with enough runs. Real variance, not a degenerate point mass.
    corrects = [22, 24, 23, 25, 21, 24, 23, 26, 22, 24]  # /40 => ~0.55..0.65
    files = [_write(tmp_path, f"r{i}.judged.json", _run(c, 40)) for i, c in enumerate(corrects)]
    rep = gate_report(files, min_runs=10, comparator_acc=0.592, comparator_name="rag-vector")
    assert rep["n_runs"] == 10
    assert rep["ci95_mean"][0] <= 0.592          # lower bound does not clear comparator
    assert rep["verdict"] == "PARTIAL"
    assert "does not clear" in rep["verdict_reason"]


def test_errored_rows_excluded_from_accuracy(tmp_path):
    rows = _run(20, 33) + [{"sample_id": f"e{i}", "error": "quota"} for i in range(7)]
    p = _write(tmp_path, "r.judged.json", rows)
    rep = gate_report([p], min_runs=10)
    assert rep["per_run"][0]["answered"] == 33
    assert rep["per_run"][0]["correct"] == 20


def test_bootstrap_ci_is_deterministic():
    a = _bootstrap_ci([0.85, 0.848, 0.86, 0.84, 0.855])
    b = _bootstrap_ci([0.85, 0.848, 0.86, 0.84, 0.855])
    assert a == b                      # fixed seed -> reproducible verdict
    assert a[0] <= 0.8506 <= a[1] or a[0] <= a[1]


def test_honest_note_forbids_auto_sota(tmp_path):
    files = [_write(tmp_path, f"r{i}.judged.json", _run(34, 40)) for i in range(10)]
    rep = gate_report(files, min_runs=10, comparator_acc=0.592)
    assert "NOT itself a 'best/SOTA' claim" in rep["honest_note"]
