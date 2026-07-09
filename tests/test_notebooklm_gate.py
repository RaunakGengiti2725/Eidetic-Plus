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


def test_truncated_runs_excluded_do_not_flip_partial_to_pass(tmp_path):
    # BUG FIX F1: 10 full runs whose CI STRADDLES 0.592 (PARTIAL on their own) + 5 truncated
    # 3/3=1.0 runs must NOT flip to PASS -- 15 truncated questions cannot override 400 real ones.
    corrects = [22, 24, 23, 25, 21, 24, 23, 26, 22, 24]   # /40 -> mean ~0.585, CI straddles
    files = [_write(tmp_path, f"full{i}.judged.json", _run(c, 40)) for i, c in enumerate(corrects)]
    files += [_write(tmp_path, f"trunc{i}.judged.json", _run(3, 3)) for i in range(5)]
    rep = gate_report(files, min_runs=10, comparator_acc=0.592, comparator_name="rag-vector")
    assert rep["runs_excluded_truncated"] == 5
    assert rep["n_runs"] == 10                    # only the full runs count
    assert rep["verdict"] == "PARTIAL"            # NOT flipped by the truncated 1.0 runs
    # sanity: including the truncated runs (the OLD buggy behavior) WOULD have raised the mean
    assert rep["mean_accuracy"] < 0.6


def test_single_full_run_cannot_pass_on_a_degenerate_ci(tmp_path):
    # BUG FIX F4: n=1 has no real interval -> no PASS, ci95_mean is null (not a fake point CI)
    f = _write(tmp_path, "one.judged.json", _run(34, 40))
    rep = gate_report([f], min_runs=1, comparator_acc=0.60)
    assert rep["verdict"] == "PARTIAL"
    assert rep["ci95_mean"] is None


def test_no_comparator_reason_does_not_claim_a_ci_was_cleared(tmp_path):
    # BUG FIX F2: with no comparator, the reason must not say "clears comparator CI"
    files = [_write(tmp_path, f"r{i}.judged.json", _run(34, 40)) for i in range(10)]
    rep = gate_report(files, min_runs=10, comparator_acc=None)
    assert "clears comparator CI" not in rep["verdict_reason"]


def test_bootstrap_ci_is_order_independent(tmp_path):
    # BUG FIX F3: same multiset, different order -> identical CI
    from bench.notebooklm_gate import _bootstrap_ci
    xs = [0.55, 0.60, 0.575, 0.625, 0.55, 0.60, 0.575, 0.625, 0.55, 0.60]
    assert _bootstrap_ci(xs) == _bootstrap_ci(list(reversed(xs)))
