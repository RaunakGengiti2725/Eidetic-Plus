"""Offline tests for benchmark co-activation wiring."""
from __future__ import annotations

from dataclasses import replace

from bench.adapters.base import AnswerResult, MemorySystem, WriteResult
from bench.datasets import Sample, Session, Turn
from bench.harness import run_system
from eidetic.models import Scope


def test_engine_link_coactivated_writes_graph(engine):
    scope = Scope(namespace="co")
    res = engine.link_coactivated(["m1", "m2", "m3"], scope=scope, valid_at=123.0)
    assert res["linked"] == 3
    assert set(engine.graph.linked_memories("m1", scope, 124.0)) == {"m2", "m3"}


def test_adapter_after_answer_respects_bench_flag(fresh_settings):
    from eidetic.engine import Engine
    from bench.adapters.eidetic_adapter import EideticSystem

    settings = replace(fresh_settings, bench_coactivation_enabled=True)
    system = EideticSystem(Engine(settings, client=object()))
    ar = AnswerResult(
        answer="ok",
        extra={"candidate_memory_ids": ["m1", "m2"]},
    )
    post = system.after_answer("bench-ns", "q", ar, correct=False, as_of=50.0)
    assert post["linked"] == 1
    assert system.engine.graph.linked_memories("m1", Scope(namespace="bench-ns"), 51.0) == ["m2"]


def test_harness_calls_after_answer_hook(tmp_path):
    class DummySystem(MemorySystem):
        name = "dummy"

        def __init__(self):
            self.called = 0

        def reset(self, namespace: str) -> None:
            self.namespace = namespace

        def ingest_session(self, namespace: str, session_id: str, turns: list[dict],
                           session_time=None) -> WriteResult:
            return WriteResult(tokens=1, ms=0.0)

        def answer(self, namespace: str, question: str, as_of=None) -> AnswerResult:
            return AnswerResult(answer="gold", extra={"candidate_memory_ids": ["a", "b"]})

        def after_answer(self, namespace: str, question: str, result: AnswerResult, *,
                         correct=None, as_of=None) -> dict:
            self.called += 1
            return {"called": self.called, "correct_seen": bool(correct)}

    class Judge:
        def judge_locomo(self, question, gold, answer):
            return answer == gold

    sess = [Session("s", [Turn("user", "x")], session_time=1.0)]
    sample = Sample("s_q0", sess, "q", "gold", "single-hop", "locomo", question_time=2.0)
    system = DummySystem()
    res = run_system(system, [sample], Judge(), runs=1, out_dir=tmp_path)
    assert system.called == 1
    assert res[0].extra["post_answer"] == {"called": 1, "correct_seen": True}
