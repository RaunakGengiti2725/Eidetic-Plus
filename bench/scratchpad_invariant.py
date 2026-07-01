"""Rotating scratchpad invariant.

The scratchpad is a tiny host-agent context surface, so it must be selective and auditable:

* only active records in the requested scope can appear;
* low-salience, future, expired, invalidated, and other-scope records are excluded;
* selected entries are ordered by salience plus verified-helpful tie-breaks;
* every selected entry carries immutable proof pointers (`content_hash` and `raw_uri`).
* scratchpad-on retrieval surfaces those same facts as verification candidates.

The eval is fully offline and dataset-neutral. Each run invents fresh scopes/topics from a seed.
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
from eidetic.scratchpad import select_scratchpad
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
class ScratchpadCase:
    case_id: str
    topic: str
    at: float
    top_salience: float
    lower_salience: float
    min_salience: float


_TOPICS = [
    "passport appointment", "studio alarm", "school pickup", "client escalation",
    "medical allergy", "garden deadline", "travel document", "therapy note",
    "key handoff", "tax filing", "field permit", "family ritual",
]


def _hash(scope: Scope, memory_id: str, text: str) -> str:
    return hashlib.sha256(
        f"{scope.namespace}\0{scope.agent_id}\0{scope.project_id}\0{memory_id}\0{text}".encode("utf-8")
    ).hexdigest()


def _record(memory_id: str, text: str, *, scope: Scope, valid_at: float,
            salience: float, helpful: int = 0, invalid_at: float | None = None,
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
        salience=salience,
        verified_helpful_count=helpful,
        content_hash=h,
        raw_uri=f"cas://{h}",
    )


def _settings(*, scratchpad_on: bool, min_salience: float) -> Settings:
    return replace(
        Settings(),
        scratchpad_enabled=scratchpad_on,
        scratchpad_topk=3,
        scratchpad_min_salience=min_salience,
        scratchpad_channel_weight=0.6,
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
        raw_span_audit_enabled=False,
        user_evidence_context_enabled=False,
        assistant_evidence_context_enabled=False,
        graph_bridge_context_enabled=False,
        recall_trace_enabled=True,
        ann_topk=6,
        final_topk=6,
    )


def _retriever(store: RecordStore, dense: list[tuple[str, float]], *,
               scratchpad_on: bool, min_salience: float) -> Retriever:
    return Retriever(
        store,
        _DenseIndex(dense),
        KnowledgeGraph(store),
        _NoopClient(),
        _NoopReranker(),
        _settings(scratchpad_on=scratchpad_on, min_salience=min_salience),
    )


def generate_cases(seed: int, cases: int) -> list[ScratchpadCase]:
    rng = random.Random(seed)
    out: list[ScratchpadCase] = []
    for idx in range(cases):
        min_salience = rng.uniform(0.55, 0.65)
        top_salience = rng.uniform(0.86, 0.98)
        lower_salience = rng.uniform(min_salience + 0.04, top_salience - 0.08)
        out.append(ScratchpadCase(
            case_id=f"scratchpad-{idx}",
            topic=rng.choice(_TOPICS),
            at=1_800_000_000 + idx * 10_000,
            top_salience=top_salience,
            lower_salience=lower_salience,
            min_salience=min_salience,
        ))
    rng.shuffle(out)
    return out


def _run_case(store: RecordStore, case: ScratchpadCase) -> tuple[int, int, dict]:
    scope = Scope(
        namespace=f"scratchpad-{case.case_id}",
        agent_id=f"agent-{case.case_id}",
        project_id="project-main",
    )
    other_scope = Scope(
        namespace=scope.namespace,
        agent_id=f"agent-other-{case.case_id}",
        project_id="project-main",
    )
    expected_ids = [
        f"{case.case_id}-tie-helpful",
        f"{case.case_id}-tie-less-helpful",
        f"{case.case_id}-lower-active",
    ]
    records = [
        _record(
            expected_ids[1],
            f"Remember this active high-salience fact about {case.topic}.",
            scope=scope,
            valid_at=case.at - 300,
            salience=case.top_salience,
            helpful=1,
        ),
        _record(
            expected_ids[0],
            f"Remember this repeatedly confirmed fact about {case.topic}.",
            scope=scope,
            valid_at=case.at - 250,
            salience=case.top_salience,
            helpful=4,
        ),
        _record(
            expected_ids[2],
            f"Still salient but lower priority fact about {case.topic}.",
            scope=scope,
            valid_at=case.at - 200,
            salience=case.lower_salience,
            helpful=8,
        ),
        _record(
            f"{case.case_id}-low-salience",
            f"Routine low-salience note about {case.topic}.",
            scope=scope,
            valid_at=case.at - 180,
            salience=case.min_salience - 0.05,
            helpful=20,
        ),
        _record(
            f"{case.case_id}-invalidated",
            f"Superseded high-salience fact about {case.topic}.",
            scope=scope,
            valid_at=case.at - 400,
            invalid_at=case.at - 10,
            salience=0.99,
            helpful=9,
        ),
        _record(
            f"{case.case_id}-expired",
            f"Expired high-salience fact about {case.topic}.",
            scope=scope,
            valid_at=case.at - 400,
            expired_at=case.at - 5,
            salience=0.99,
            helpful=9,
        ),
        _record(
            f"{case.case_id}-future",
            f"Future high-salience fact about {case.topic}.",
            scope=scope,
            valid_at=case.at + 500,
            salience=0.99,
            helpful=9,
        ),
        _record(
            f"{case.case_id}-other-scope",
            f"Other agent high-salience fact about {case.topic}.",
            scope=other_scope,
            valid_at=case.at - 100,
            salience=0.99,
            helpful=9,
        ),
        _record(
            f"{case.case_id}-retrieval-decoy",
            "scratchpad proof item surfaced decoy only.",
            scope=scope,
            valid_at=case.at - 50,
            salience=case.min_salience - 0.08,
            helpful=0,
        ),
    ]
    for rec in records:
        store.upsert_record(rec)

    active = store.active_records_at(case.at, scope)
    scratchpad = select_scratchpad(
        active,
        top_k=3,
        min_salience=case.min_salience,
    )
    ids = [entry["memory_id"] for entry in scratchpad]
    forbidden_ids = {
        f"{case.case_id}-low-salience",
        f"{case.case_id}-invalidated",
        f"{case.case_id}-expired",
        f"{case.case_id}-future",
        f"{case.case_id}-other-scope",
        f"{case.case_id}-retrieval-decoy",
    }
    hash_by_id = {rec.memory_id: rec.content_hash for rec in records}
    uri_by_id = {rec.memory_id: rec.raw_uri for rec in records}

    checks = 0
    correct = 0

    def check(ok: bool) -> None:
        nonlocal checks, correct
        checks += 1
        if ok:
            correct += 1

    check(ids == expected_ids)
    check(not forbidden_ids.intersection(ids))
    check(all(entry["content_hash"] == hash_by_id[entry["memory_id"]] for entry in scratchpad))
    check(all(entry.get("raw_uri") == uri_by_id[entry["memory_id"]] for entry in scratchpad))
    check(all(len(str(entry.get("content_hash", ""))) == 64 for entry in scratchpad))
    check(all(str(entry.get("raw_uri", "")).startswith("cas://") for entry in scratchpad))
    check(len(scratchpad) == 3)

    query = "Which scratchpad proof item should be surfaced?"
    dense = [(f"{case.case_id}-retrieval-decoy", 0.95)]
    off_retriever = _retriever(
        store, dense, scratchpad_on=False, min_salience=case.min_salience
    )
    on_retriever = _retriever(
        store, dense, scratchpad_on=True, min_salience=case.min_salience
    )
    off = off_retriever.retrieve(
        query,
        scope=scope,
        at=case.at,
        qvec=np.array([1.0, 0.0], dtype=np.float32),
        use_recency=False,
        skip_rerank=True,
    )
    on = on_retriever.retrieve(
        query,
        scope=scope,
        at=case.at,
        qvec=np.array([1.0, 0.0], dtype=np.float32),
        use_recency=False,
        skip_rerank=True,
    )
    off_ids = {cand.record.memory_id for cand in off}
    on_ids = {cand.record.memory_id for cand in on}
    trace = on_retriever.last_trace
    trace_scratchpad = (
        list(getattr(trace, "channel_results", {}).get("scratchpad", []))
        if trace is not None else []
    )
    check(expected_ids[0] not in off_ids)
    check(expected_ids[0] in on_ids)
    check(trace is not None and "scratchpad" in getattr(trace, "enabled_channels", []))
    check(expected_ids[0] in trace_scratchpad)

    detail = {
        "case_id": case.case_id,
        "topic": case.topic,
        "expected_ids": expected_ids,
        "actual_ids": ids,
        "retrieval_off_ids": sorted(off_ids),
        "retrieval_on_ids": sorted(on_ids),
        "scratchpad_trace_ids": trace_scratchpad,
        "forbidden_returned": sorted(forbidden_ids.intersection(ids)),
        "proof_linked": all(entry.get("content_hash") and entry.get("raw_uri") for entry in scratchpad),
        "min_salience": round(case.min_salience, 6),
        "returned": scratchpad,
    }
    return correct, checks, detail


def run_eval(*, seed: Optional[int] = None, cases: int = 24) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    correct = 0
    checks = 0
    failures: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="scratchpad-invariant-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        for case in generated:
            got, local_checks, detail = _run_case(store, case)
            correct += got
            checks += local_checks
            if got != local_checks:
                failures.append(detail)
    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        "checks": checks,
        "correct": correct,
        "ordering_checks": cases,
        "active_scope_filter_checks": cases,
        "proof_link_checks": cases * 4,
        "top_k_checks": cases,
        "retrieval_channel_checks": cases * 4,
        "case_type_counts": {"scratchpad_active_proof_surface": cases},
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
