"""Offline tests for the neutral harness: dataset loaders, run-grouping, age computation,
and scoreboard rendering. These need NO model key (they exercise the harness plumbing, not
the live runs). The live 3-system scoreboard is produced by `bash bench/reproduce.sh`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.datasets import Sample, Session, Turn, category_counts
from bench.harness import _age_days, _group_by_sessions
from bench import scoreboard


def _sample(sid, sessions, q="q", gold="g", cat="single-hop", ds="locomo", qtime=None):
    return Sample(sample_id=sid, sessions=sessions, question=q, gold=gold,
                  category=cat, dataset=ds, question_time=qtime)


def test_grouping_shares_sessions():
    sess = [Session("s0", [Turn("user", "hi")], session_time=1000.0)]
    a, b = _sample("c0_q0", sess), _sample("c0_q1", sess)  # same sessions object
    c = _sample("c1_q0", [Session("s1", [Turn("user", "yo")])])
    groups = _group_by_sessions([a, b, c])
    assert len(groups) == 2
    sizes = sorted(len(qs) for _, qs in groups)
    assert sizes == [1, 2]


def test_age_days_from_session_times():
    sess = [Session("s0", [Turn("user", "x")], session_time=0.0),
            Session("s1", [Turn("user", "y")], session_time=10 * 86400.0)]
    s = _sample("c_q", sess, qtime=20 * 86400.0)
    assert _age_days(s) == pytest.approx(20.0)
    # No times -> None
    assert _age_days(_sample("c_q", [Session("s", [Turn("user", "x")])])) is None


def test_locomo_loader_counts_if_cached():
    path = Path("data/bench/locomo/locomo10.json")
    if not path.exists():
        pytest.skip("LoCoMo not downloaded; run the harness once to fetch it.")
    from bench.datasets import locomo
    samples = locomo.load()
    counts = category_counts(samples)
    assert "adversarial" not in counts                 # validated categories only
    assert set(counts) <= {"single-hop", "multi-hop", "temporal", "open-domain"}
    assert sum(counts.values()) == 1540                # matches the dossier's ~1,540 QA


def test_scoreboard_pending_when_no_logs(tmp_path):
    md = scoreboard.render(tmp_path)
    assert md.exists()
    assert "Pending run" in md.read_text()             # never fabricates numbers


def test_scoreboard_aggregates_from_real_logs(tmp_path):
    # Two systems, one category, two runs -> mean/std must compute correctly.
    rows = []
    for run in (0, 1):
        for sysname, correctness in (("eidetic-plus", [True, True]), ("mem0", [True, False])):
            for i, ok in enumerate(correctness):
                rows.append({
                    "system": sysname, "dataset": "locomo", "category": "single-hop",
                    "sample_id": f"c0_q{i}", "question": "q", "gold": "g", "predicted": "p",
                    "correct": ok, "write_tokens": 100, "query_tokens": 50,
                    "search_ms": 10.0, "e2e_ms": 100.0, "abstained": False,
                    "run_idx": run, "age_days": 5.0, "n_sessions": 2, "extra": {},
                })
    (tmp_path / "eidetic-plus__run0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows if r["system"] == "eidetic-plus" and r["run_idx"] == 0))
    (tmp_path / "eidetic-plus__run1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows if r["system"] == "eidetic-plus" and r["run_idx"] == 1))
    (tmp_path / "mem0__run0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows if r["system"] == "mem0" and r["run_idx"] == 0))
    (tmp_path / "mem0__run1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows if r["system"] == "mem0" and r["run_idx"] == 1))

    md = scoreboard.render(tmp_path, {"judge_model": "qwen3-max", "judge_backend": "dashscope"})
    text = md.read_text()
    assert "single-hop" in text and "eidetic-plus" in text and "mem0" in text
    data = json.loads((tmp_path / "scoreboard.json").read_text())
    # eidetic-plus: 100% both runs -> mean 1.0; mem0: 50% both runs -> mean 0.5
    eid = data["accuracy"]["eidetic-plus|locomo|single-hop"]
    mem0 = data["accuracy"]["mem0|locomo|single-hop"]
    assert eid["mean"] == pytest.approx(1.0)
    assert mem0["mean"] == pytest.approx(0.5)
    assert eid["successes"] == 4 and eid["n"] == 4
    assert len(eid["ci95"]) == 2
    h2h = data["head_to_head"]["eidetic-plus|mem0|locomo|single-hop"]
    assert h2h["a_only"] == 2 and h2h["b_only"] == 0
    assert "eidetic-plus vs mem0" in text
    assert data["survival"]["eidetic-plus|mem0|locomo|single-hop"]["status"] == "survives"


def test_fixed_reader_cot_extracts_answer(tmp_path, monkeypatch):
    from bench import reader
    from eidetic.config import get_settings

    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("READER_COT", "1")
    get_settings.cache_clear()

    class FakeClient:
        def __init__(self):
            self.system = ""

        def chat_json(self, _model, system, _user, **_kw):
            self.system = system
            return {"notes": [{"source": "S0", "relevant": True, "note": "Paris"}],
                    "answer": "Paris [S0]"}

    fake = FakeClient()
    monkeypatch.setattr(reader, "get_client", lambda: fake)
    assert reader.answer_with_fixed_reader("Where?", ["Alice moved to Paris."]) == "Paris [S0]"
    assert "Reply ONLY as JSON" in fake.system
    get_settings.cache_clear()


def test_memoryagentbench_normalized_loader(tmp_path):
    from bench.datasets import memoryagentbench

    data_dir = tmp_path / "memoryagentbench"
    data_dir.mkdir()
    row = {
        "sample_id": "fc1",
        "category": "factconsolidation",
        "question": "Where does Alice work now?",
        "answer": "Globex",
        "sessions": [{"session_id": "s1", "turns": [{"role": "user", "content": "Alice moved to Globex."}]}],
    }
    (data_dir / "factconsolidation.jsonl").write_text(json.dumps(row) + "\n")
    samples = memoryagentbench.load(data_dir=data_dir)
    assert len(samples) == 1
    assert samples[0].dataset == "memoryagentbench"
    assert memoryagentbench.verify(samples)["has_target_task"]


def test_beam_normalized_loader(tmp_path):
    from bench.datasets import beam

    data_dir = tmp_path / "beam"
    data_dir.mkdir()
    row = {
        "id": "b1",
        "ability": "contradiction_resolution",
        "query": "What is the current project code?",
        "gold": "NIM-9",
        "turns": [{"role": "user", "content": "The current project code is NIM-9."}],
    }
    (data_dir / "beam_1m.jsonl").write_text(json.dumps(row) + "\n")
    samples = beam.load(data_dir=data_dir)
    assert len(samples) == 1
    assert samples[0].dataset == "beam"
    assert beam.verify(samples)["has_contradiction_resolution"]


def test_deterministic_memory_judges_do_not_need_api():
    from bench.judge import Judge, exact_match, substring_exact_match

    assert exact_match("Globex", ["globex"])
    assert substring_exact_match("Alice works at Globex now.", ["Globex"])
    judge = Judge()
    assert judge.judge_memoryagentbench("", "Alice works at Globex now.", {"gold_aliases": ["Globex"]})
    assert judge.judge_beam("NIM-9", "NIM-9", {})


def test_harness_routes_memory_datasets_to_deterministic_judges():
    from bench.harness import _judge_sample
    from bench.datasets import Sample, Session, Turn

    class FakeJudge:
        def judge_memoryagentbench(self, gold, answer, meta):
            return gold == "Globex" and "Globex" in answer and meta["gold_aliases"] == ["Globex"]

        def judge_beam(self, gold, answer, meta):
            return gold == answer and meta == {}

    mab = Sample("m1", [Session("s", [Turn("user", "x")])], "q", "Globex",
                 "factconsolidation", "memoryagentbench", meta={"gold_aliases": ["Globex"]})
    beam_sample = Sample("b1", [Session("s", [Turn("user", "x")])], "q", "NIM-9",
                         "contradiction_resolution", "beam")
    assert _judge_sample(FakeJudge(), mab, "Alice works at Globex.")
    assert _judge_sample(FakeJudge(), beam_sample, "NIM-9")


def _write_gate_manifest(path: Path, reader_model: str = "qwen-plus") -> None:
    (path / "run_manifest.json").write_text(json.dumps({
        "systems": "mem0",
        "dataset": "locomo",
        "subset": 10,
        "sample_offset": 0,
        "runs": 1,
        "run_offset": 0,
        "variant": "longmemeval_s",
        "render_only": False,
        "judge": {"judge_model": "qwen3-max"},
        "sample_count": 10,
        "category_counts": {"single-hop": 10},
        "sample_rows": [
            {"dataset": "locomo", "sample_id": f"s{i}", "category": "single-hop"}
            for i in range(10)
        ],
        "env": {"READER_MODEL": reader_model, "JUDGE_MODEL": "qwen3-max"},
    }))


def test_mem0_gate_passes_only_against_real_logs(tmp_path):
    from bench.gate import run_gate

    _write_gate_manifest(tmp_path)
    rows = [
        {"system": "mem0", "dataset": "locomo", "category": "single-hop",
         "sample_id": f"s{i}", "correct": i < 8, "run_idx": 0,
         "query_tokens": 10, "write_tokens": 20, "search_ms": 3.0, "e2e_ms": 30.0}
        for i in range(10)
    ]
    (tmp_path / "mem0__run0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({
        "min_n": 10,
        "tolerance_points": 2.0,
        "categories": {"single-hop": 80.0},
    }))
    res = run_gate(tmp_path, expected)
    assert res["status"] == "PASS"
    assert res["comparisons"]["single-hop"]["n"] == 10
    assert res["comparisons"]["single-hop"]["search_p95"] == pytest.approx(3.0)


def test_mem0_gate_fails_on_missing_category_or_small_n(tmp_path):
    from bench.gate import run_gate

    _write_gate_manifest(tmp_path)
    (tmp_path / "mem0__run0.jsonl").write_text(json.dumps({
        "system": "mem0", "dataset": "locomo", "category": "single-hop",
        "sample_id": "s0", "correct": True, "run_idx": 0,
    }) + "\n")
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({
        "min_n": 50,
        "categories": {"single-hop": 100.0, "temporal": 50.0},
    }))
    res = run_gate(tmp_path, expected)
    assert res["status"] == "FAIL"
    assert res["comparisons"]["temporal"]["status"] == "missing"


def test_mem0_gate_fails_without_strong_reader_or_bool_correct(tmp_path):
    from bench.gate import run_gate

    _write_gate_manifest(tmp_path, reader_model="qwen-flash")
    (tmp_path / "mem0__run0.jsonl").write_text(json.dumps({
        "system": "mem0", "dataset": "locomo", "category": "single-hop",
        "sample_id": "s0", "correct": True, "run_idx": 0,
    }) + "\n")
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({
        "min_n": 1,
        "categories": {"single-hop": 100.0},
    }))
    res = run_gate(tmp_path, expected)
    assert res["status"] == "FAIL"
    assert "reader mismatch" in res["reason"]

    _write_gate_manifest(tmp_path)
    (tmp_path / "mem0__run0.jsonl").write_text(json.dumps({
        "system": "mem0", "dataset": "locomo", "category": "single-hop",
        "sample_id": "s0", "correct": "true", "run_idx": 0,
    }) + "\n")
    with pytest.raises(ValueError, match="correct"):
        run_gate(tmp_path, expected)


def test_load_samples_supports_offset_without_changing_both_semantics(monkeypatch):
    from bench import run
    from bench.datasets import Sample, Session, Turn

    def samples(ds):
        return [
            Sample(f"{ds}{i}", [Session("s", [Turn("user", "x")])], "q", "a", "cat", ds)
            for i in range(5)
        ]

    monkeypatch.setattr(run.longmemeval, "load", lambda **_kw: samples("longmemeval"))
    monkeypatch.setattr(run.locomo, "load", lambda **_kw: samples("locomo"))
    out = run.load_samples("both", subset=2, variant="v", offset=1)
    assert [s.sample_id for s in out] == ["longmemeval1", "longmemeval2", "locomo1", "locomo2"]


def test_run_system_honors_run_offset(tmp_path):
    from bench.adapters.base import AnswerResult, MemorySystem, WriteResult
    from bench.datasets import Sample, Session, Turn
    from bench.harness import run_system

    class FakeSystem(MemorySystem):
        name = "fake"

        def reset(self, namespace):
            self.namespace = namespace

        def ingest_session(self, namespace, session_id, turns, session_time=None):
            return WriteResult(tokens=1, ms=0.0)

        def answer(self, namespace, question, as_of=None):
            return AnswerResult(answer="yes", context_tokens=1)

    class FakeJudge:
        def judge_locomo(self, question, gold, hypothesis):
            return True

    samples = [Sample("s_q0", [Session("sess", [Turn("user", "x")])], "q", "yes",
                      "single-hop", "locomo")]
    run_system(FakeSystem(), samples, FakeJudge(), runs=1, out_dir=tmp_path, run_offset=3)
    assert (tmp_path / "fake__run3.jsonl").exists()
    with pytest.raises(FileExistsError):
        run_system(FakeSystem(), samples, FakeJudge(), runs=1, out_dir=tmp_path, run_offset=3)


def test_run_cli_rejects_invalid_offsets(monkeypatch):
    from bench import run

    monkeypatch.setattr("sys.argv", ["bench.run", "--sample-offset", "-1"])
    with pytest.raises(SystemExit):
        run.main()
    monkeypatch.setattr("sys.argv", ["bench.run", "--run-offset", "-1"])
    with pytest.raises(SystemExit):
        run.main()
    monkeypatch.setattr("sys.argv", ["bench.run", "--runs", "0"])
    with pytest.raises(SystemExit):
        run.main()


def test_compare_dirs_reports_flag_deltas(tmp_path):
    from bench.compare import compare_dirs, render_markdown

    control = tmp_path / "control"
    experiment = tmp_path / "experiment"
    control.mkdir()
    experiment.mkdir()

    def rows(ok0, ok1, query_tokens, search_ms):
        return [
            {"system": "eidetic-plus", "dataset": "locomo", "category": "temporal",
             "sample_id": "s0", "correct": ok0, "run_idx": 0,
             "query_tokens": query_tokens, "search_ms": search_ms, "e2e_ms": search_ms + 100},
            {"system": "eidetic-plus", "dataset": "locomo", "category": "temporal",
             "sample_id": "s1", "correct": ok1, "run_idx": 0,
             "query_tokens": query_tokens, "search_ms": search_ms, "e2e_ms": search_ms + 100},
        ]

    (control / "eidetic-plus__run0.jsonl").write_text("\n".join(json.dumps(r) for r in rows(True, False, 100, 20)))
    (experiment / "eidetic-plus__run0.jsonl").write_text("\n".join(json.dumps(r) for r in rows(True, True, 80, 10)))
    res = compare_dirs(control, experiment, system="eidetic-plus")
    item = res["comparisons"]["eidetic-plus|locomo|temporal"]
    assert item["delta_accuracy_points"] == pytest.approx(50.0)
    assert item["delta_tokens_per_query"] == pytest.approx(-20.0)
    assert item["delta_write_tokens_per_conversation"] is None
    assert item["paired"]["experiment_only"] == 1
    md = render_markdown(res, tmp_path / "compare.md")
    assert md.exists()
    assert (tmp_path / "compare.json").exists()


def test_compare_dirs_fails_on_duplicate_rows(tmp_path):
    from bench.compare import compare_dirs

    control = tmp_path / "control"
    experiment = tmp_path / "experiment"
    control.mkdir()
    experiment.mkdir()
    row = {"system": "eidetic-plus", "dataset": "locomo", "category": "temporal",
           "sample_id": "s0", "correct": True, "run_idx": 0}
    (control / "eidetic-plus__run0.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    (experiment / "eidetic-plus__run0.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate row key"):
        compare_dirs(control, experiment)


def test_compare_dirs_missing_metrics_are_null(tmp_path):
    from bench.compare import compare_dirs

    control = tmp_path / "control"
    experiment = tmp_path / "experiment"
    control.mkdir()
    experiment.mkdir()
    row = {"system": "eidetic-plus", "dataset": "locomo", "category": "temporal",
           "sample_id": "s0", "correct": True, "run_idx": 0}
    (control / "eidetic-plus__run0.jsonl").write_text(json.dumps(row) + "\n")
    (experiment / "eidetic-plus__run0.jsonl").write_text(json.dumps(row) + "\n")
    item = compare_dirs(control, experiment)["comparisons"]["eidetic-plus|locomo|temporal"]
    assert item["control"]["tokens_per_query"] is None
    assert item["delta_tokens_per_query"] is None


def test_compare_dirs_marks_disjoint_slices_unpaired(tmp_path):
    from bench.compare import compare_dirs, render_markdown

    control = tmp_path / "control"
    experiment = tmp_path / "experiment"
    control.mkdir()
    experiment.mkdir()
    c_row = {"system": "eidetic-plus", "dataset": "locomo", "category": "temporal",
             "sample_id": "slice0", "correct": True, "run_idx": 0}
    e_row = dict(c_row, sample_id="slice50")
    (control / "eidetic-plus__run0.jsonl").write_text(json.dumps(c_row) + "\n")
    (experiment / "eidetic-plus__run0.jsonl").write_text(json.dumps(e_row) + "\n")
    res = compare_dirs(control, experiment, system="eidetic-plus")
    item = res["comparisons"]["eidetic-plus|locomo|temporal"]
    assert res["status"] == "unpaired"
    assert item["status"] == "unpaired"
    assert item["paired"]["paired_n"] == 0
    md = render_markdown(res, tmp_path / "compare.md")
    assert "unpaired control" in md.read_text()


def test_run_manifest_records_flags(tmp_path, monkeypatch):
    from argparse import Namespace
    from bench.datasets import Sample, Session, Turn
    from bench.run import write_manifest

    monkeypatch.setenv("READER_COT", "1")
    monkeypatch.setenv("CONTEXT_COMPRESS", "1")
    monkeypatch.setenv("RERANK_FAIL_OPEN", "1")
    monkeypatch.setenv("RRF_W_BM25", "1.2")
    args = Namespace(
        systems="eidetic", dataset="locomo", subset=10, sample_offset=5,
        runs=1, run_offset=2, variant="longmemeval_s", render_only=False,
    )
    samples = [Sample("s0", [Session("sess", [Turn("user", "x")])], "q", "a",
                      "single-hop", "locomo")]
    path = write_manifest(tmp_path, args, {"judge_model": "qwen-plus"}, samples=samples)
    data = json.loads(path.read_text())
    assert data["sample_offset"] == 5
    assert data["run_offset"] == 2
    assert data["env"]["READER_COT"] == "1"
    assert data["env"]["CONTEXT_COMPRESS"] == "1"
    assert data["env"]["RERANK_FAIL_OPEN"] == "1"
    assert data["env"]["RRF_W_BM25"] == "1.2"
    assert data["sample_rows"] == [{"dataset": "locomo", "sample_id": "s0", "category": "single-hop"}]


def test_manifest_records_all_sweep_env_vars(tmp_path):
    from argparse import Namespace
    from bench.run import write_manifest
    from bench.sweep import STAGES, stage_assignment

    args = Namespace(
        systems="eidetic", dataset="locomo", subset=10, sample_offset=0,
        runs=1, run_offset=0, variant="longmemeval_s", render_only=False,
    )
    data = json.loads(write_manifest(tmp_path, args, {}, samples=[]).read_text())
    expected = set()
    for stage, values in STAGES:
        for value in values:
            expected.update(stage_assignment(stage, value))
    assert expected <= set(data["env"])


def test_sweep_cli_accepts_current_dataset_choices(monkeypatch):
    from bench import sweep

    monkeypatch.setattr("sys.argv", ["bench.sweep", "--dry-run", "--dataset", "all", "--subset", "1"])
    assert sweep.main() == 0


def test_sweep_rejects_empty_or_malformed_rows(tmp_path):
    from bench.sweep import _validate_rows, score_rows

    with pytest.raises(RuntimeError, match="produced no rows"):
        _validate_rows([], tmp_path)
    with pytest.raises(RuntimeError, match="missing fields"):
        score_rows([{"system": "eidetic-plus"}])


def test_sweep_writes_stage_ab_artifact(tmp_path):
    from bench.sweep import write_stage_comparison

    control = tmp_path / "READER_COT=0"
    experiment = tmp_path / "READER_COT=1"
    control.mkdir()
    experiment.mkdir()
    row = {"system": "eidetic-plus", "dataset": "locomo", "category": "temporal",
           "sample_id": "s0", "correct": True, "run_idx": 0,
           "query_tokens": 100, "search_ms": 20.0, "e2e_ms": 120.0}
    better = dict(row, query_tokens=90, search_ms=10.0, e2e_ms=100.0)
    (control / "eidetic-plus__run0.jsonl").write_text(json.dumps(row) + "\n")
    (experiment / "eidetic-plus__run0.jsonl").write_text(json.dumps(better) + "\n")

    md, result = write_stage_comparison(control, experiment, stage="READER_COT", value="1")
    assert md.exists()
    data = json.loads(md.with_suffix(".json").read_text())
    assert result["status"] == "ok"
    assert data["sweep_stage"] == "READER_COT"
    assert data["sweep_value"] == "1"


def test_live_sweep_requires_mem0_gate_paths(monkeypatch):
    from bench import sweep
    from eidetic.config import get_settings

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    get_settings.cache_clear()
    monkeypatch.setattr("sys.argv", ["bench.sweep", "--dataset", "locomo", "--subset", "1"])
    with pytest.raises(SystemExit, match="Mem0 reproduction gate"):
        sweep.main()
    get_settings.cache_clear()
