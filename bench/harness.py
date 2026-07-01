"""The runner. Drives every system through the IDENTICAL loop and the one fixed judge,
logging one JSON line per question so every number reproduces from the raw logs.

Protocol: samples that share a conversation (same `sessions` object) are grouped, so the
history is ingested ONCE per conversation/haystack and every question on it is answered
against the same store -- faithful to both LongMemEval (per-question haystack) and LoCoMo
(shared conversation). Each group runs in its own isolated scope (namespace).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .adapters.base import MemorySystem
from .datasets import Sample
from .judge import Judge


@dataclass
class QResult:
    system: str
    dataset: str
    category: str
    sample_id: str
    question: str
    gold: str
    predicted: str
    correct: bool
    write_tokens: int
    query_tokens: int
    search_ms: float
    e2e_ms: float
    abstained: bool
    run_idx: int
    age_days: Optional[float] = None
    n_sessions: int = 0
    extra: dict = field(default_factory=dict)
    error: str = ""          # transport/runtime error on this question (excluded from accuracy)


def _group_by_sessions(samples: list[Sample]) -> list[tuple[list, list[Sample]]]:
    groups: dict[int, tuple[list, list[Sample]]] = {}
    for s in samples:
        key = id(s.sessions)
        if key not in groups:
            groups[key] = (s.sessions, [])
        groups[key][1].append(s)
    return list(groups.values())


def _age_days(sample: Sample) -> Optional[float]:
    times = [s.session_time for s in sample.sessions if s.session_time is not None]
    if not times:
        return None
    q = sample.question_time if sample.question_time is not None else max(times)
    return max(0.0, (q - min(times)) / 86400.0)


def _as_of_time(sample: Sample) -> Optional[float]:
    """Question timestamp for bi-temporal reads.

    Some benchmark rows provide dated haystack sessions but no explicit question date. In that
    case, relative phrases such as "past few months" should resolve against the conversation end,
    not the machine's wall clock years later. This is passed to every system equally.
    """
    if sample.question_time is not None:
        return sample.question_time
    times = [s.session_time for s in sample.sessions if s.session_time is not None]
    return max(times) if times else None


def _judge_sample(judge: Judge, sample: Sample, answer: str) -> bool:
    if sample.dataset == "longmemeval":
        return judge.judge_longmemeval(sample.question, sample.gold, answer, sample.category)
    if sample.dataset == "locomo":
        return judge.judge_locomo(sample.question, sample.gold, answer)
    if sample.dataset == "memoryagentbench":
        return judge.judge_memoryagentbench(sample.gold, answer, sample.meta)
    if sample.dataset == "beam":
        return judge.judge_beam(sample.gold, answer, sample.meta)
    return judge.judge_generic_memory(sample.question, sample.gold, answer, sample.category)


def run_system(system: MemorySystem, samples: list[Sample], judge: Judge, *,
               runs: int, out_dir: Path, run_offset: int = 0,
               overwrite: bool = False) -> list[QResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if runs <= 0:
        raise ValueError("runs must be positive")
    if run_offset < 0:
        raise ValueError("run_offset must be >= 0")
    if not samples:
        raise ValueError("no samples loaded")
    groups = _group_by_sessions(samples)
    results: list[QResult] = []

    for run_idx in range(run_offset, run_offset + runs):
        log_path = out_dir / f"{system.name}__run{run_idx}.jsonl"
        if log_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing log {log_path}. Use a new --run-offset, "
                "a separate --out directory, or --overwrite."
            )
        with open(log_path, "w") as fh:
            for gi, (sessions, qs) in enumerate(groups):
                ns = f"{system.name}-{qs[0].dataset}-g{gi}-r{run_idx}"
                write_tokens = 0
                consolidate_report = {}
                group_error = ""
                try:
                    system.reset(ns)
                    for sess in sessions:
                        turns = [{"role": t.role, "content": t.content, "timestamp": t.timestamp}
                                 for t in sess.turns]
                        wr = system.ingest_session(ns, sess.session_id, turns, sess.session_time)
                        write_tokens += wr.tokens
                    consolidate_report = system.consolidate(ns) or {}
                except Exception as e:  # noqa: BLE001 - write/sleep failures are logged per row
                    group_error = f"{type(e).__name__}: {str(e)[:200]}"

                for s in qs:
                    # Per-question resilience: a transient transport/runtime error on ONE question
                    # must not abort the whole system's run (losing every other answerable question).
                    # Record it with an `error` flag and continue; analysis excludes errored rows
                    # from accuracy (never silently counted right or wrong).
                    try:
                        if group_error:
                            raise RuntimeError(f"write/consolidate failed: {group_error}")
                        q_as_of = _as_of_time(s)
                        ar = system.answer(ns, s.question, as_of=q_as_of)
                        if consolidate_report:
                            ar.extra = {**(ar.extra or {}), "consolidate": consolidate_report}
                        correct = _judge_sample(judge, s, ar.answer)
                        try:
                            post = system.after_answer(
                                ns, s.question, ar, correct=bool(correct), as_of=q_as_of
                            )
                            if post:
                                ar.extra = {**(ar.extra or {}), "post_answer": post}
                        except Exception as hook_error:  # optional hook, never abort a scored row
                            ar.extra = {
                                **(ar.extra or {}),
                                "post_answer_error": f"{type(hook_error).__name__}: {str(hook_error)[:200]}",
                            }
                        qr = QResult(
                            system=system.name, dataset=s.dataset, category=s.category,
                            sample_id=s.sample_id, question=s.question, gold=s.gold,
                            predicted=ar.answer, correct=bool(correct),
                            write_tokens=write_tokens, query_tokens=ar.context_tokens,
                            search_ms=ar.search_ms, e2e_ms=ar.e2e_ms, abstained=ar.abstained,
                            run_idx=run_idx, age_days=_age_days(s), n_sessions=len(sessions),
                            extra=ar.extra,
                        )
                    except Exception as e:  # noqa: BLE001 - record + continue, never abort the run
                        qr = QResult(
                            system=system.name, dataset=s.dataset, category=s.category,
                            sample_id=s.sample_id, question=s.question, gold=s.gold,
                            predicted="", correct=False, write_tokens=write_tokens,
                            query_tokens=0, search_ms=0.0, e2e_ms=0.0, abstained=False,
                            run_idx=run_idx, age_days=_age_days(s), n_sessions=len(sessions),
                            error=f"{type(e).__name__}: {str(e)[:200]}",
                            extra={"consolidate": consolidate_report} if consolidate_report else {},
                        )
                    fh.write(json.dumps(asdict(qr)) + "\n")
                    fh.flush()
                    results.append(qr)
    system.teardown()
    return results


def load_logs(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(Path(out_dir).glob("*__run*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows
