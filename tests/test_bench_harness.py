"""Offline tests for the neutral harness: dataset loaders, run-grouping, age computation,
and scoreboard rendering. These need NO model key (they exercise the harness plumbing, not
the live runs). The live 3-system scoreboard is produced by `bash bench/reproduce.sh`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.adapters.base import AnswerResult, MemorySystem, WriteResult
from bench.datasets import Sample, Session, Turn, category_counts
from bench.harness import _age_days, _as_of_time, _group_by_sessions, run_system
from bench import scoreboard


def _sample(sid, sessions, q="q", gold="g", cat="single-hop", ds="locomo", qtime=None):
    return Sample(sample_id=sid, sessions=sessions, question=q, gold=gold,
                  category=cat, dataset=ds, question_time=qtime)


def test_grouping_shares_sessions():
    sess = [Session("s0", [Turn("user", "hi")], session_time=1000.0)]
    a, b = _sample("toy0_q0", sess), _sample("toy0_q1", sess)  # same sessions object
    c = _sample("toy1_q0", [Session("s1", [Turn("user", "yo")])])
    groups = _group_by_sessions([a, b, c])
    assert len(groups) == 2
    sizes = sorted(len(qs) for _, qs in groups)
    assert sizes == [1, 2]


def test_run_system_logs_write_failures_per_question(tmp_path):
    class FailingWriteSystem(MemorySystem):
        name = "broken-write"

        def reset(self, namespace: str) -> None:
            return None

        def ingest_session(self, namespace: str, session_id: str, turns: list[dict],
                           session_time=None) -> WriteResult:
            raise TimeoutError("embedding service stalled")

        def answer(self, namespace: str, question: str, as_of=None) -> AnswerResult:
            raise AssertionError("answer should not be called after write failure")

    class Judge:
        def judge_locomo(self, question: str, gold: str, answer: str) -> bool:
            raise AssertionError("judge should not be called after write failure")

    sess = [Session("s0", [Turn("user", "hello")], session_time=1000.0)]
    samples = [
        _sample("toy0_q0", sess, q="q0", gold="g"),
        _sample("toy0_q1", sess, q="q1", gold="g"),
    ]

    results = run_system(
        FailingWriteSystem(),
        samples,
        Judge(),
        runs=1,
        out_dir=tmp_path,
        overwrite=True,
    )

    assert len(results) == 2
    assert all("write/consolidate failed: TimeoutError" in r.error for r in results)
    assert all(r.correct is False for r in results)
    rows = [
        json.loads(line)
        for line in (tmp_path / "broken-write__run0.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert all("embedding service stalled" in row["error"] for row in rows)


def test_age_days_from_session_times():
    sess = [Session("s0", [Turn("user", "x")], session_time=0.0),
            Session("s1", [Turn("user", "y")], session_time=10 * 86400.0)]
    s = _sample("c_q", sess, qtime=20 * 86400.0)
    assert _age_days(s) == pytest.approx(20.0)
    # No times -> None
    assert _age_days(_sample("c_q", [Session("s", [Turn("user", "x")])])) is None


def test_as_of_time_falls_back_to_latest_session_time():
    sess = [Session("s0", [Turn("user", "x")], session_time=1000.0),
            Session("s1", [Turn("user", "y")], session_time=3000.0)]
    assert _as_of_time(_sample("c_q", sess)) == 3000.0
    assert _as_of_time(_sample("c_q", sess, qtime=2000.0)) == 2000.0
    assert _as_of_time(_sample("c_q", [Session("s", [Turn("user", "x")])])) is None


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


def test_longmemeval_loader_preserves_haystack_session_ids(tmp_path):
    from bench.datasets import longmemeval

    d = tmp_path / "longmemeval"
    d.mkdir()
    (d / "mini.json").write_text(json.dumps([{
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "What degree did I graduate with?",
        "answer": "Business Administration",
        "answer_session_ids": ["answer_1"],
        "haystack_session_ids": ["noise_1", "answer_1"],
        "haystack_dates": ["2023/05/01 (Mon) 09:00", "2023/05/02 (Tue) 09:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "I graduated with a degree in Business Administration."}],
        ],
    }]))

    [sample] = longmemeval.load(variant="mini", data_dir=d)

    assert [s.session_id for s in sample.sessions] == ["noise_1", "answer_1"]
    assert sample.meta["answer_session_ids"] == ["answer_1"]


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
                extra = {}
                if sysname == "eidetic-plus":
                    extra = {"consolidate": {"consolidate_pending": {
                        "pending_processed": 2,
                        "facts_extracted": 1,
                        "events_indexed": 1,
                        "extraction_timed_out": 1,
                        "extraction_deferred": 0,
                    }}}
                rows.append({
                    "system": sysname, "dataset": "locomo", "category": "single-hop",
                    "sample_id": f"c0_q{i}", "question": "q", "gold": "g", "predicted": "p",
                    "correct": ok, "write_tokens": 100, "query_tokens": 50,
                    "search_ms": 10.0, "e2e_ms": 100.0, "abstained": False,
                    "run_idx": run, "age_days": 5.0, "n_sessions": 2, "extra": extra,
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
    assert "Consolidation Health" in text
    assert data["consolidation"]["eidetic-plus"]["groups"] == 2
    assert data["consolidation"]["eidetic-plus"]["pending_processed"] == 4
    assert data["consolidation"]["eidetic-plus"]["extraction_timed_out"] == 2
    assert data["log_fingerprint"]["file_count"] == 4
    assert data["log_fingerprint"]["combined_sha256"]


def test_curves_single_age_slice_renders_without_warnings(tmp_path):
    import warnings
    from bench import curves

    row = {
        "system": "eidetic-plus", "dataset": "longmemeval", "category": "single-session-user",
        "sample_id": "q0", "question": "q", "gold": "g", "predicted": "g", "correct": True,
        "write_tokens": 1, "query_tokens": 1, "search_ms": 1.0, "e2e_ms": 2.0,
        "abstained": False, "run_idx": 0, "age_days": 10.0, "n_sessions": 1, "extra": {},
    }
    (tmp_path / "eidetic-plus__run0.jsonl").write_text(json.dumps(row) + "\n")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        res = curves.render(tmp_path)
    assert res["ok"] is True
    assert res["slopes"] == {}
    assert "Need at least two" in res["note"]
    assert seen == []
    assert (tmp_path / "recall_vs_age.png").exists()


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
    from bench.judge import Judge, exact_match, short_answer_exact_match, substring_exact_match

    assert exact_match("Globex", ["globex"])
    assert exact_match("Summer Vibes [S4]", ["Summer Vibes"])
    assert substring_exact_match("Alice works at Globex now.", ["Globex"])
    assert not substring_exact_match("13", ["3"])       # token-contiguous, not char substring
    assert short_answer_exact_match("Summer Vibes", "Summer Vibes [S4]")
    assert not short_answer_exact_match("3", "13 [S0]")
    judge = Judge()
    assert judge.judge_memoryagentbench("", "Alice works at Globex now.", {"gold_aliases": ["Globex"]})
    assert judge.judge_beam("NIM-9", "NIM-9", {})


def test_longmemeval_answerable_decline_is_never_correct(monkeypatch):
    from bench.judge import Judge

    judge = Judge()
    monkeypatch.setattr(judge, "_call", lambda *_args, **_kw: "yes")

    assert judge.judge_longmemeval(
        "What degree did I graduate with?",
        "Business Administration",
        "I don't have enough verified evidence in memory to answer that confidently.",
        "single-session-user",
    ) is False


def test_longmemeval_short_exact_match_skips_llm_judge():
    from bench.judge import Judge

    class NoCallJudge(Judge):
        def _call(self, system, user):
            raise AssertionError("LLM judge should not be called for exact short answer")

    assert NoCallJudge().judge_longmemeval(
        "What is the Spotify playlist called?",
        "Summer Vibes",
        "Summer Vibes [S4]",
        "single-session-user",
    )


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
    from bench.gate import render_markdown, run_gate

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
    assert res["log_fingerprint"]["file_count"] == 1
    assert res["log_fingerprint"]["combined_sha256"]
    out = render_markdown(res, tmp_path / "mem0_gate.md")
    assert "Mem0 Reproduction Gate" in out.read_text()
    rendered = json.loads((tmp_path / "mem0_gate.json").read_text())
    assert rendered["status"] == "PASS"
    assert rendered["log_fingerprint"] == res["log_fingerprint"]


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


def test_load_samples_stratified_round_robins_conversation_categories(monkeypatch):
    from bench import run
    from bench.datasets import Sample, Session, Turn

    sess = [Session("s", [Turn("user", "x")])]
    loaded = [
        Sample("toy0_q0", sess, "q", "a", "temporal", "locomo"),
        Sample("toy0_q1", sess, "q", "a", "temporal", "locomo"),
        Sample("toy0_q2", sess, "q", "a", "multi-hop", "locomo"),
        Sample("toy1_q0", sess, "q", "a", "temporal", "locomo"),
        Sample("toy1_q1", sess, "q", "a", "temporal", "locomo"),
        Sample("toy2_q0", sess, "q", "a", "open-domain", "locomo"),
    ]

    monkeypatch.setattr(run.locomo, "load", lambda **_kw: loaded)
    out = run.load_samples(
        "locomo", subset=5, variant="v", split="all", sample_strategy="stratified"
    )

    assert [s.sample_id for s in out] == ["toy0_q0", "toy0_q2", "toy2_q0", "toy1_q0", "toy0_q1"]


def test_load_samples_stratified_balances_category_sorted_longmemeval(monkeypatch):
    from bench import run
    from bench.datasets import Sample, Session, Turn

    sess = [Session("s", [Turn("user", "x")])]
    loaded = [
        Sample("u0", sess, "q", "a", "single-session-user", "longmemeval"),
        Sample("u1", sess, "q", "a", "single-session-user", "longmemeval"),
        Sample("u2", sess, "q", "a", "single-session-user", "longmemeval"),
        Sample("m0", sess, "q", "a", "multi-session", "longmemeval"),
        Sample("m1", sess, "q", "a", "multi-session", "longmemeval"),
        Sample("t0", sess, "q", "a", "temporal-reasoning", "longmemeval"),
        Sample("k0", sess, "q", "a", "knowledge-update", "longmemeval"),
    ]

    monkeypatch.setattr(run.longmemeval, "load", lambda **_kw: loaded)
    out = run.load_samples(
        "longmemeval", subset=6, variant="v", split="all", sample_strategy="stratified"
    )

    assert [s.sample_id for s in out] == ["u0", "m0", "t0", "k0", "u1", "m1"]


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


def test_run_system_logs_consolidation_report(tmp_path):
    from bench.adapters.base import AnswerResult, MemorySystem, WriteResult
    from bench.datasets import Sample, Session, Turn
    from bench.harness import run_system

    class FakeSystem(MemorySystem):
        name = "fake"

        def reset(self, namespace):
            self.namespace = namespace

        def ingest_session(self, namespace, session_id, turns, session_time=None):
            return WriteResult(tokens=1, ms=0.0)

        def consolidate(self, namespace):
            return {"consolidate_pending": {
                "pending_processed": 3,
                "facts_extracted": 2,
                "events_indexed": 1,
                "extraction_timed_out": 1,
                "extraction_deferred": 0,
            }}

        def answer(self, namespace, question, as_of=None):
            return AnswerResult(answer="yes", context_tokens=1, extra={"verified": True})

    class FakeJudge:
        def judge_locomo(self, question, gold, hypothesis):
            return True

    samples = [Sample("s_q0", [Session("sess", [Turn("user", "x")])], "q", "yes",
                      "single-hop", "locomo")]
    results = run_system(FakeSystem(), samples, FakeJudge(), runs=1, out_dir=tmp_path)
    assert results[0].extra["verified"] is True
    assert results[0].extra["consolidate"]["consolidate_pending"]["extraction_timed_out"] == 1
    row = json.loads((tmp_path / "fake__run0.jsonl").read_text())
    assert row["extra"]["consolidate"]["consolidate_pending"]["pending_processed"] == 3


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

    def rows(ok0, ok1, query_tokens, search_ms, *, timeout=False):
        out = [
            {"system": "eidetic-plus", "dataset": "locomo", "category": "temporal",
             "sample_id": "s0", "correct": ok0, "run_idx": 0,
             "query_tokens": query_tokens, "search_ms": search_ms, "e2e_ms": search_ms + 100},
            {"system": "eidetic-plus", "dataset": "locomo", "category": "temporal",
             "sample_id": "s1", "correct": ok1, "run_idx": 0,
             "query_tokens": query_tokens, "search_ms": search_ms, "e2e_ms": search_ms + 100},
        ]
        if timeout:
            out[1]["extra"] = {"consolidate": {"consolidate_pending": {
                "pending_processed": 2,
                "facts_extracted": 1,
                "events_indexed": 1,
                "extraction_timed_out": 1,
                "extraction_deferred": 0,
            }}}
        return out

    (control / "eidetic-plus__run0.jsonl").write_text("\n".join(json.dumps(r) for r in rows(True, False, 100, 20)))
    (experiment / "eidetic-plus__run0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows(True, True, 80, 10, timeout=True)))
    res = compare_dirs(control, experiment, system="eidetic-plus")
    item = res["comparisons"]["eidetic-plus|locomo|temporal"]
    assert item["delta_accuracy_points"] == pytest.approx(50.0)
    assert item["delta_tokens_per_query"] == pytest.approx(-20.0)
    assert item["delta_write_tokens_per_conversation"] is None
    assert item["paired"]["experiment_only"] == 1
    assert res["consolidation"]["experiment"]["eidetic-plus"]["extraction_timed_out"] == 1
    assert res["consolidation"]["delta"]["eidetic-plus"]["delta"]["extraction_timed_out"] == 1
    # Per-question flip attribution: s1 went wrong->right (a gain); nothing regressed.
    assert item["paired"]["gained"] == [{"sample_id": "s1", "run_idx": 0}]
    assert item["paired"]["regressed"] == []
    md = render_markdown(res, tmp_path / "compare.md")
    assert md.exists()
    assert (tmp_path / "compare.json").exists()
    # The flip table names the actual questions so a judge can check the attribution.
    text = md.read_text()
    assert "Per-question flips" in text
    assert "Consolidation deltas" in text
    assert "s1" in text


def test_compare_flip_table_lists_gains_and_regressions(tmp_path):
    from bench.compare import compare_dirs, render_markdown

    control = tmp_path / "control"
    experiment = tmp_path / "experiment"
    control.mkdir()
    experiment.mkdir()

    def row(sid, ok):
        return {"system": "eidetic-plus", "dataset": "locomo", "category": "temporal",
                "sample_id": sid, "correct": ok, "run_idx": 0,
                "query_tokens": 100, "search_ms": 20, "e2e_ms": 120}

    # control: q0 right, q1 wrong, q2 right. experiment: q0 wrong (regress), q1 right (gain), q2 right.
    (control / "eidetic-plus__run0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in (row("q0", True), row("q1", False), row("q2", True))))
    (experiment / "eidetic-plus__run0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in (row("q0", False), row("q1", True), row("q2", True))))
    res = compare_dirs(control, experiment, system="eidetic-plus")
    paired = res["comparisons"]["eidetic-plus|locomo|temporal"]["paired"]
    assert [g["sample_id"] for g in paired["gained"]] == ["q1"]
    assert [r["sample_id"] for r in paired["regressed"]] == ["q0"]
    md = render_markdown(res, tmp_path / "compare.md").read_text()
    assert "Per-question flips" in md


def test_compare_flip_table_dedupes_question_ids_across_runs(tmp_path):
    # Review finding 2: with >1 run, a question that flips in both runs must be listed ONCE in the
    # flip table (distinct questions), not "q1, q1".
    from bench.compare import compare_dirs, render_markdown

    control = tmp_path / "control"
    experiment = tmp_path / "experiment"
    control.mkdir()
    experiment.mkdir()

    def row(ok, run):
        return {"system": "eidetic-plus", "dataset": "locomo", "category": "temporal",
                "sample_id": "q1", "correct": ok, "run_idx": run,
                "query_tokens": 100, "search_ms": 20, "e2e_ms": 120}

    for run in (0, 1):
        (control / f"eidetic-plus__run{run}.jsonl").write_text(json.dumps(row(False, run)))
        (experiment / f"eidetic-plus__run{run}.jsonl").write_text(json.dumps(row(True, run)))
    res = compare_dirs(control, experiment, system="eidetic-plus")
    paired = res["comparisons"]["eidetic-plus|locomo|temporal"]["paired"]
    assert paired["experiment_only"] == 2          # 2 discordant question-runs (McNemar is per-run)
    md = render_markdown(res, tmp_path / "compare.md").read_text()
    assert "q1, q1" not in md                       # q1 listed once, not duplicated per run
    assert "| 1 | 0 | q1 | - |" in md              # 1 distinct gained question, 0 regressed


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
        sample_strategy="stratified", runs=1, run_offset=2, variant="longmemeval_s",
        render_only=False,
    )
    samples = [Sample("s0", [Session("sess", [Turn("user", "x")])], "q", "a",
                      "single-hop", "locomo")]
    path = write_manifest(tmp_path, args, {"judge_model": "qwen-plus"}, samples=samples)
    data = json.loads(path.read_text())
    assert data["sample_offset"] == 5
    assert data["sample_strategy"] == "stratified"
    assert data["run_offset"] == 2
    assert data["system_failures"] == []
    assert data["env"]["READER_COT"] == "1"
    assert data["env"]["CONTEXT_COMPRESS"] == "1"
    assert data["env"]["RERANK_FAIL_OPEN"] == "1"
    assert data["env"]["RRF_W_BM25"] == "1.2"
    assert data["sample_rows"] == [{"dataset": "locomo", "sample_id": "s0", "category": "single-hop"}]


def test_run_manifest_records_system_failures(tmp_path):
    from argparse import Namespace
    from bench.run import write_manifest

    args = Namespace(
        systems="eidetic,graphiti", dataset="locomo", subset=10, sample_offset=0,
        sample_strategy="contiguous", runs=1, run_offset=0, variant="longmemeval_s",
        render_only=False,
    )
    failures = [{
        "system": "graphiti",
        "error_type": "RuntimeError",
        "error": "Neo4j DNS failure",
    }]
    data = json.loads(write_manifest(
        tmp_path, args, {}, samples=[], system_failures=failures
    ).read_text())
    assert data["system_failures"] == failures


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
