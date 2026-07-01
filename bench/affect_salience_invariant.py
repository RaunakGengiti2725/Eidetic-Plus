"""Rotating affect-salience retrieval invariant.

This sidecar proves the salience hook behaves like a memory-priority signal, not a timestamp leak:

* dense-only retrieval must prefer the stronger lexical/vector match;
* enabling affect salience must surface the high-salience memory even when it is older;
* equal-salience memories must receive equal boost regardless of age;
* every boost must stay within the configured bounded fraction of the pre-boost top score.

No benchmark questions, model calls, or adapter shortcuts are used. The eval runs the real Retriever
fusion path with a fake dense index over rotating synthetic memories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from bench.seed_policy import resolve_seed
from eidetic.config import Settings
from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.retrieval import Retriever
from eidetic.store import RecordStore


class _DenseIndex:
    def __init__(self, dense: list[tuple[str, float]]):
        self.dense = dense

    def __len__(self) -> int:
        return max(len(self.dense), 1)

    def search(self, _qvec, k: int, allowed_ids=None, ef=None):
        del ef
        items = [
            (mid, score)
            for mid, score in self.dense
            if allowed_ids is None or mid in allowed_ids
        ]
        return sorted(items, key=lambda item: -item[1])[:k]

    def get_vectors(self, ids):
        return {}


class _NoopClient:
    def embed_text(self, _text: str):
        return np.array([1.0, 0.0], dtype=np.float32)


class _NoopReranker:
    pass


@dataclass(frozen=True)
class AffectCase:
    case_id: str
    topic: str
    old_salience: float
    new_salience: float
    old_dense: float
    new_dense: float
    old_valid_at: float
    new_valid_at: float
    equal_salience: float
    equal_old_valid_at: float
    equal_new_valid_at: float


_TOPICS = [
    "passport renewal", "studio access", "medical forms", "school pickup",
    "travel charger", "client deadline", "allergy note", "field permit",
    "key handoff", "tax packet", "garden schedule", "therapy appointment",
]


def _settings(*, affect_on: bool, lambda_salience: float) -> Settings:
    return replace(
        Settings(),
        affect_salience_enabled=affect_on,
        lambda_salience=lambda_salience,
        persistent_bm25_enabled=False,
        rerank_enabled=False,
        parallel_channels_enabled=False,
        rocchio_enabled=False,
        adaptive_ef_enabled=False,
        temporal_rerank_enabled=False,
        active_fact_context_enabled=False,
        memory_typing_enabled=False,
        struct_channel_enabled=False,
        event_ranking_enabled=False,
        gist_channel_enabled=False,
        graph_vocab_seeding=False,
        coactivation_channel_enabled=False,
        flow_hybrid_channel_enabled=False,
        temporal_evidence_audit_enabled=False,
        aggregation_audit_enabled=False,
        list_audit_enabled=False,
        user_evidence_context_enabled=False,
        assistant_evidence_context_enabled=False,
        ann_topk=10,
        final_topk=10,
    )


def _record(memory_id: str, text: str, *, scope: Scope, valid_at: float,
            salience: float) -> MemoryRecord:
    digest = hashlib.sha256(
        f"{scope.namespace}\0{memory_id}\0{text}\0{valid_at}\0{salience}".encode("utf-8")
    ).hexdigest()
    return MemoryRecord(
        memory_id=memory_id,
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        salience=salience,
        content_hash=digest,
        raw_uri=f"mem://affect-salience/{digest}",
    )


def _scores(store: RecordStore, scope: Scope, dense: list[tuple[str, float]], *,
            affect_on: bool, lambda_salience: float) -> dict[str, float]:
    retriever = Retriever(
        store,
        _DenseIndex(dense),
        KnowledgeGraph(store),
        _NoopClient(),
        _NoopReranker(),
        _settings(affect_on=affect_on, lambda_salience=lambda_salience),
    )
    cands = retriever.retrieve(
        "which memory should be routed first",
        scope=scope,
        at=3_000_000_000.0,
        qvec=np.array([1.0, 0.0], dtype=np.float32),
        use_recency=False,
        skip_rerank=True,
    )
    return {cand.record.memory_id: cand.fused_score for cand in cands}


def _top(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda item: item[1])[0] if scores else ""


def generate_cases(seed: int, cases: int) -> list[AffectCase]:
    rng = random.Random(seed)
    out: list[AffectCase] = []
    for idx in range(cases):
        topic = rng.choice(_TOPICS)
        old_salience = rng.uniform(0.90, 0.98)
        new_salience = rng.uniform(0.02, 0.10)
        new_dense = rng.uniform(0.84, 0.90)
        old_dense = new_dense - rng.uniform(0.02, 0.04)
        base_time = 1_500_000_000 + idx * 10_000
        out.append(AffectCase(
            case_id=f"affect-salience-{idx}",
            topic=topic,
            old_salience=old_salience,
            new_salience=new_salience,
            old_dense=old_dense,
            new_dense=new_dense,
            old_valid_at=base_time,
            new_valid_at=base_time + rng.randint(80_000_000, 1_000_000_000),
            equal_salience=rng.uniform(0.60, 0.95),
            equal_old_valid_at=base_time - rng.randint(40_000_000, 900_000_000),
            equal_new_valid_at=base_time + rng.randint(40_000_000, 900_000_000),
        ))
    rng.shuffle(out)
    return out


def _run_case(store: RecordStore, case: AffectCase, *,
              lambda_salience: float) -> tuple[int, int, dict]:
    scope = Scope(namespace=f"affect-salience-{case.case_id}")
    old_id = f"{case.case_id}-old-important"
    new_id = f"{case.case_id}-new-routine"
    equal_old_id = f"{case.case_id}-equal-old"
    equal_new_id = f"{case.case_id}-equal-new"
    rows = [
        _record(
            old_id,
            f"Important memory about {case.topic}: keep this routed first.",
            scope=scope,
            valid_at=case.old_valid_at,
            salience=case.old_salience,
        ),
        _record(
            new_id,
            f"Routine memory about {case.topic}: useful but not emotionally important.",
            scope=scope,
            valid_at=case.new_valid_at,
            salience=case.new_salience,
        ),
        _record(
            equal_old_id,
            f"Equal-salience old control for {case.topic}.",
            scope=scope,
            valid_at=case.equal_old_valid_at,
            salience=case.equal_salience,
        ),
        _record(
            equal_new_id,
            f"Equal-salience new control for {case.topic}.",
            scope=scope,
            valid_at=case.equal_new_valid_at,
            salience=case.equal_salience,
        ),
    ]
    for rec in rows:
        store.upsert_record(rec)

    dense = [
        (new_id, case.new_dense),
        (old_id, case.old_dense),
        (equal_old_id, 0.70),
        (equal_new_id, 0.70),
    ]
    off = _scores(store, scope, dense, affect_on=False, lambda_salience=lambda_salience)
    on = _scores(store, scope, dense, affect_on=True, lambda_salience=lambda_salience)

    checks = 0
    correct = 0

    def check(ok: bool) -> None:
        nonlocal checks, correct
        checks += 1
        if ok:
            correct += 1

    baseline_top = _top(off)
    salience_top = _top(on)
    old_delta = on.get(old_id, 0.0) - off.get(old_id, 0.0)
    new_delta = on.get(new_id, 0.0) - off.get(new_id, 0.0)
    equal_old_delta = on.get(equal_old_id, 0.0) - off.get(equal_old_id, 0.0)
    equal_new_delta = on.get(equal_new_id, 0.0) - off.get(equal_new_id, 0.0)
    max_base = max(off.values()) if off else 0.0
    observed_boosts = {
        old_id: old_delta,
        new_id: new_delta,
        equal_old_id: equal_old_delta,
        equal_new_id: equal_new_delta,
    }
    salience_by_id = {
        old_id: case.old_salience,
        new_id: case.new_salience,
        equal_old_id: case.equal_salience,
        equal_new_id: case.equal_salience,
    }
    bounded = all(
        -1e-12 <= delta <= lambda_salience * salience_by_id[mid] * max_base + 1e-12
        for mid, delta in observed_boosts.items()
    )

    check(baseline_top == new_id)
    check(salience_top == old_id)
    check(case.old_valid_at < case.new_valid_at)
    check(case.old_salience > case.new_salience)
    check(old_delta > new_delta)
    check(abs(equal_old_delta - equal_new_delta) < 1e-12)
    check(bounded)

    return correct, checks, {
        "case_id": case.case_id,
        "topic": case.topic,
        "baseline_top": baseline_top,
        "salience_top": salience_top,
        "old_memory_id": old_id,
        "new_memory_id": new_id,
        "old_salience": round(case.old_salience, 6),
        "new_salience": round(case.new_salience, 6),
        "old_valid_at": case.old_valid_at,
        "new_valid_at": case.new_valid_at,
        "age_gap_seconds": case.new_valid_at - case.old_valid_at,
        "old_delta": old_delta,
        "new_delta": new_delta,
        "equal_old_delta": equal_old_delta,
        "equal_new_delta": equal_new_delta,
        "max_base_score": max_base,
        "max_boost_ratio": (
            max((abs(v) for v in observed_boosts.values()), default=0.0) / max_base
            if max_base > 0.0 else 0.0
        ),
        "bounded": bounded,
    }


def run_eval(*, seed: Optional[int] = None, cases: int = 24,
             lambda_salience: float = 0.5) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    if lambda_salience <= 0.0:
        raise ValueError("lambda_salience must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    correct = 0
    checks = 0
    failures: list[dict] = []
    max_boost_ratio = 0.0
    min_age_gap_seconds: float | None = None
    with tempfile.TemporaryDirectory(prefix="affect-salience-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        for case in generated:
            got, local_checks, detail = _run_case(store, case, lambda_salience=lambda_salience)
            correct += got
            checks += local_checks
            max_boost_ratio = max(max_boost_ratio, float(detail["max_boost_ratio"]))
            age_gap = float(detail["age_gap_seconds"])
            min_age_gap_seconds = age_gap if min_age_gap_seconds is None else min(min_age_gap_seconds, age_gap)
            if got != local_checks:
                failures.append(detail)
    flip_checks = cases * 2
    age_free_checks = cases
    bounded_checks = cases
    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        "checks": checks,
        "correct": correct,
        "flip_checks": flip_checks,
        "age_free_checks": age_free_checks,
        "bounded_checks": bounded_checks,
        "lambda_salience": lambda_salience,
        "max_boost_ratio": round(max_boost_ratio, 6),
        "min_age_gap_seconds": min_age_gap_seconds,
        "case_type_counts": {"affect_salience_retrieval": cases},
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None, help="repro seed; omitted means random")
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--lambda-salience", type=float, default=0.5)
    ap.add_argument("--out", default="", help="optional JSON report path")
    args = ap.parse_args()
    report = run_eval(seed=args.seed, cases=args.cases, lambda_salience=args.lambda_salience)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
