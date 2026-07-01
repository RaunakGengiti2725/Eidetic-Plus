"""Rotating reflex-recall invariant.

Reflex recall is the tiny local memory packet that can seed answers before ANN or a reader run.
This sidecar proves, on fresh synthetic scopes each run, that the packet is:

* local and proof-ready: candidates carry content hashes, raw URIs, and score contracts;
* scoped and bi-temporal: invalidated, expired, future, and other-agent records are excluded;
* associative: a query-hit memory can pull in a co-activated silent memory;
* fast: packet build stays inside the configured local latency budget.

No benchmark questions, model clients, embeddings, NLI, or answer generation are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import quantiles
from typing import Optional

from bench.seed_policy import resolve_seed
from eidetic.config import Settings
from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Scope
from eidetic.reflex_activation import build_memory_packet
from eidetic.reflex_index import ReflexIndex
from eidetic.store import RecordStore


@dataclass(frozen=True)
class ReflexCase:
    case_id: str
    topic: str
    at: float


_TOPICS = [
    "passport", "studio alarm", "school handoff", "client escalation",
    "medical allergy", "garden permit", "travel receipt", "therapy schedule",
    "key transfer", "tax packet", "field permit", "family ritual",
]


def _settings() -> Settings:
    return replace(
        Settings(),
        reflex_topk=12,
        reflex_max_seeds=200,
        reflex_coact_seeds=4,
        reflex_budget_ms=100,
    )


def _hash(scope: Scope, memory_id: str, text: str) -> str:
    return hashlib.sha256(
        f"{scope.namespace}\0{scope.agent_id}\0{scope.project_id}\0{memory_id}\0{text}".encode("utf-8")
    ).hexdigest()


def _record(memory_id: str, text: str, *, scope: Scope, valid_at: float,
            invalid_at: float | None = None,
            expired_at: float | None = None) -> MemoryRecord:
    h = _hash(scope, memory_id, text)
    return MemoryRecord(
        memory_id=memory_id,
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        content_hash=h,
        raw_uri=f"cas://{h}",
    )


def generate_cases(seed: int, cases: int) -> list[ReflexCase]:
    rng = random.Random(seed)
    out: list[ReflexCase] = []
    for idx in range(cases):
        out.append(ReflexCase(
            case_id=f"reflex-{idx}",
            topic=rng.choice(_TOPICS),
            at=2_000_000_000 + idx * 10_000,
        ))
    rng.shuffle(out)
    return out


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(quantiles(values, n=20, method="inclusive")[18])


def _run_case(store: RecordStore, graph: KnowledgeGraph, case: ReflexCase,
              settings: Settings) -> tuple[int, int, float, dict]:
    scope = Scope(
        namespace=f"reflex-invariant-{case.case_id}",
        agent_id=f"agent-{case.case_id}",
        project_id="project-main",
    )
    other_scope = Scope(
        namespace=scope.namespace,
        agent_id=f"other-agent-{case.case_id}",
        project_id="project-main",
    )
    target_id = f"{case.case_id}-target"
    silent_id = f"{case.case_id}-silent-coactivated"
    invalid_id = f"{case.case_id}-invalid"
    expired_id = f"{case.case_id}-expired"
    future_id = f"{case.case_id}-future"
    other_scope_id = f"{case.case_id}-other-scope"
    target_text = f"alpha {case.topic} quartz route packet proof"
    silent_text = "Sapphire backup drawer; no query vocabulary here."
    records = [
        _record(target_id, target_text, scope=scope, valid_at=case.at - 300),
        _record(silent_id, silent_text, scope=scope, valid_at=case.at - 280),
        _record(invalid_id, target_text, scope=scope, valid_at=case.at - 500,
                invalid_at=case.at - 10),
        _record(expired_id, target_text, scope=scope, valid_at=case.at - 450,
                expired_at=case.at - 8),
        _record(future_id, target_text, scope=scope, valid_at=case.at + 500),
        _record(other_scope_id, target_text, scope=other_scope, valid_at=case.at - 250),
    ]
    for idx in range(80):
        records.append(_record(
            f"{case.case_id}-decoy-{idx}",
            f"background local memory {idx} about unrelated archive material",
            scope=scope,
            valid_at=case.at - 200 + idx,
        ))
    for rec in records:
        store.upsert_record(rec)
    graph.link_memories([target_id, silent_id], scope=scope, valid_at=case.at - 100)

    index = ReflexIndex()
    index.rebuild_from_store(store)
    query = f"alpha {case.topic} quartz route packet proof"
    packet = build_memory_packet(
        query,
        scope,
        store=store,
        graph=graph,
        index=index,
        settings=settings,
        as_of=case.at,
    )
    ids = packet.candidate_ids()
    id_set = set(ids)
    by_id = {item.memory_id: item for item in packet.items}
    candidates = packet.to_candidates()
    candidate_ids = [cand.record.memory_id for cand in candidates]
    latency_ms = float(packet.latency_ms.get("total", 0.0) or 0.0)

    checks = 0
    correct = 0

    def check(ok: bool) -> None:
        nonlocal checks, correct
        checks += 1
        if ok:
            correct += 1

    check(target_id in id_set)
    check(target_id in by_id and "lexical" in by_id[target_id].retrieval_paths)
    check(silent_id in id_set)
    check(silent_id in by_id and "coactivation" in by_id[silent_id].retrieval_paths)
    check(invalid_id not in id_set)
    check(expired_id not in id_set)
    check(future_id not in id_set)
    check(other_scope_id not in id_set)
    check(all(item.content_hash and item.raw_uri.startswith("cas://") for item in packet.items))
    check(candidate_ids == ids and all(c.fused_score > 0.0 for c in candidates))
    check(packet.coverage >= settings.reflex_min_coverage)
    check(latency_ms <= settings.reflex_budget_ms)

    detail = {
        "case_id": case.case_id,
        "topic": case.topic,
        "candidate_ids": ids,
        "target_id": target_id,
        "silent_id": silent_id,
        "forbidden_ids": [invalid_id, expired_id, future_id, other_scope_id],
        "latency_ms": round(latency_ms, 6),
        "coverage": packet.coverage,
        "paths": {mid: by_id[mid].retrieval_paths for mid in by_id},
        "content_hashes": packet.content_hashes,
    }
    return correct, checks, latency_ms, detail


def run_eval(*, seed: Optional[int] = None, cases: int = 24) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    settings = _settings()
    correct = 0
    checks = 0
    latencies: list[float] = []
    failures: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="reflex-recall-invariant-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        graph = KnowledgeGraph(store)
        for case in generated:
            got, local_checks, latency_ms, detail = _run_case(store, graph, case, settings)
            correct += got
            checks += local_checks
            latencies.append(latency_ms)
            if got != local_checks:
                failures.append(detail)
    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        "checks": checks,
        "correct": correct,
        "direct_hit_checks": cases * 2,
        "coactivation_checks": cases * 2,
        "active_scope_filter_checks": cases * 4,
        "proof_link_checks": cases,
        "score_contract_checks": cases * 2,
        "latency_budget_checks": cases,
        "max_latency_ms": round(max(latencies, default=0.0), 6),
        "p95_latency_ms": round(_p95(latencies), 6),
        "latency_budget_ms": settings.reflex_budget_ms,
        "case_type_counts": {"reflex_recall_proof_surface": cases},
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None, help="repro seed; omitted means random")
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--out", default="", help="optional JSON report path")
    args = ap.parse_args()
    report = run_eval(seed=args.seed, cases=args.cases)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
