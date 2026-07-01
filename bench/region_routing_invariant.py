"""Rotating memory-region/cocoon routing invariant.

Memory regions are only useful if they route recall to the right neighborhood without becoming
uncited synthetic truth. This sidecar proves, on fresh synthetic scopes each run, that:

* the gist/region channel recovers raw memories that dense retrieval misses;
* nested cocoons resolve back to active raw memories;
* invalidated, expired, future, and other-scope members are filtered out;
* every emitted hint carries immutable proof pointers through content hashes and raw URIs;
* trace telemetry observes the same route-only hints used by context assembly.

No benchmark questions, model calls, or dataset-specific entities are used.
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
from eidetic.models import DerivedRecord, MemoryRecord, Scope
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
class RegionCase:
    case_id: str
    topic: str
    at: float
    dense_score: float


_TOPICS = [
    "passport pickup", "studio alarm", "school handoff", "client escalation",
    "medical allergy", "garden inspection", "travel receipt", "therapy schedule",
    "key transfer", "tax packet", "field permit", "family ritual",
]


def _settings(*, gist_on: bool) -> Settings:
    return replace(
        Settings(),
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
        gist_channel_enabled=gist_on,
        rrf_w_gist=3.0,
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
        scratchpad_enabled=False,
        recall_trace_enabled=True,
        ann_topk=8,
        final_topk=8,
        context_token_budget=1200,
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


def _retriever(store: RecordStore, dense: list[tuple[str, float]], *, gist_on: bool) -> Retriever:
    return Retriever(
        store,
        _DenseIndex(dense),
        KnowledgeGraph(store),
        _NoopClient(),
        _NoopReranker(),
        _settings(gist_on=gist_on),
    )


def generate_cases(seed: int, cases: int) -> list[RegionCase]:
    rng = random.Random(seed)
    out: list[RegionCase] = []
    for idx in range(cases):
        out.append(RegionCase(
            case_id=f"region-{idx}",
            topic=rng.choice(_TOPICS),
            at=1_900_000_000 + idx * 10_000,
            dense_score=rng.uniform(0.82, 0.94),
        ))
    rng.shuffle(out)
    return out


def _run_case(store: RecordStore, case: RegionCase) -> tuple[int, int, dict]:
    scope = Scope(
        namespace=f"region-routing-{case.case_id}",
        agent_id=f"agent-{case.case_id}",
        project_id="project-main",
    )
    other_scope = Scope(
        namespace=scope.namespace,
        agent_id=f"other-agent-{case.case_id}",
        project_id="project-main",
    )
    target_id = f"{case.case_id}-target"
    nested_id = f"{case.case_id}-nested"
    dense_id = f"{case.case_id}-dense-decoy"
    invalid_id = f"{case.case_id}-invalid"
    expired_id = f"{case.case_id}-expired"
    future_id = f"{case.case_id}-future"
    other_scope_id = f"{case.case_id}-other-scope"
    records = [
        _record(
            target_id,
            "Copper folder.",
            scope=scope,
            valid_at=case.at - 500,
        ),
        _record(
            nested_id,
            "Quartz backup code.",
            scope=scope,
            valid_at=case.at - 450,
        ),
        _record(
            dense_id,
            f"High dense decoy with no useful {case.topic} route answer.",
            scope=scope,
            valid_at=case.at - 400,
        ),
        _record(
            invalid_id,
            "Old invalidated member says use the red folder.",
            scope=scope,
            valid_at=case.at - 700,
            invalid_at=case.at - 5,
        ),
        _record(
            expired_id,
            "Expired member says use the blue folder.",
            scope=scope,
            valid_at=case.at - 650,
            expired_at=case.at - 4,
        ),
        _record(
            future_id,
            "Future member says use the silver folder.",
            scope=scope,
            valid_at=case.at + 600,
        ),
        _record(
            other_scope_id,
            "Other-scope member says use the green folder.",
            scope=other_scope,
            valid_at=case.at - 300,
        ),
    ]
    for rec in records:
        store.upsert_record(rec)

    child_cid = f"{case.case_id}-child-cocoon"
    parent_cid = f"{case.case_id}-parent-cocoon"
    direct_cid = f"{case.case_id}-direct-region"
    store.add_derived(DerivedRecord(
        cid=direct_cid,
        kind="gist",
        namespace=scope.namespace,
        level=1,
        text=f"{case.topic} direct route region copper folder",
        member_ids=[target_id, invalid_id],
        vector=[1.0, 0.0],
    ))
    store.add_derived(DerivedRecord(
        cid=child_cid,
        kind="gist",
        namespace=scope.namespace,
        level=1,
        text=f"{case.topic} nested child cocoon quartz backup",
        member_ids=[nested_id, expired_id, future_id, other_scope_id],
        vector=[1.0, 0.0],
    ))
    store.add_derived(DerivedRecord(
        cid=parent_cid,
        kind="gist",
        namespace=scope.namespace,
        level=2,
        text=f"{case.topic} parent cocoon route neighborhood",
        member_ids=[child_cid],
        vector=[1.0, 0.0],
    ))
    store.add_derived(DerivedRecord(
        cid=f"{case.case_id}-dead-decoy-cocoon",
        kind="gist",
        namespace=scope.namespace,
        level=1,
        text=f"{case.topic} decoy cocoon should not emit",
        member_ids=[invalid_id, expired_id, future_id, other_scope_id],
        vector=[1.0, 0.0],
    ))

    dense = [(dense_id, case.dense_score)]
    query = f"Which {case.topic} route and cocoon memory should I use?"
    qvec = np.array([1.0, 0.0], dtype=np.float32)
    off = _retriever(store, dense, gist_on=False).retrieve(
        query,
        scope=scope,
        at=case.at,
        qvec=qvec,
        use_recency=False,
        skip_rerank=True,
    )
    on_retriever = _retriever(store, dense, gist_on=True)
    on = on_retriever.retrieve(
        query,
        scope=scope,
        at=case.at,
        qvec=qvec,
        use_recency=False,
        skip_rerank=True,
    )
    blocks = on_retriever.assemble_context(query, on, at=case.at, scope=scope)
    telemetry = on_retriever.last_context_telemetry
    trace = on_retriever.last_trace

    off_ids = {cand.record.memory_id for cand in off}
    on_ids = {cand.record.memory_id for cand in on}
    forbidden = {invalid_id, expired_id, future_id, other_scope_id}
    hints = telemetry.get("region_hints", []) if isinstance(telemetry, dict) else []
    hint_ids = {str(h.get("region_id", "")) for h in hints if isinstance(h, dict)}
    hint_members = {
        str(mid)
        for h in hints if isinstance(h, dict)
        for mid in (h.get("members", []) or [])
    }
    hint_hashes = {
        str(ch)
        for h in hints if isinstance(h, dict)
        for ch in (h.get("content_hashes", []) or [])
    }
    hint_raw_uris = {
        str(uri)
        for h in hints if isinstance(h, dict)
        for uri in (h.get("raw_uris", []) or [])
    }
    hash_by_id = {rec.memory_id: rec.content_hash for rec in records}
    uri_by_id = {rec.memory_id: rec.raw_uri for rec in records}
    blocks_text = " ".join(blocks)

    checks = 0
    correct = 0

    def check(ok: bool) -> None:
        nonlocal checks, correct
        checks += 1
        if ok:
            correct += 1

    check(target_id not in off_ids and nested_id not in off_ids)
    check(target_id in on_ids)
    check(nested_id in on_ids)
    check(not forbidden.intersection(on_ids))
    check(direct_cid in hint_ids)
    check(parent_cid in hint_ids)
    check({target_id, nested_id}.issubset(hint_members))
    check(not forbidden.intersection(hint_members))
    check(all(hash_by_id[mid][:16] in hint_hashes for mid in (target_id, nested_id)))
    check(all(uri_by_id[mid] in hint_raw_uris for mid in (target_id, nested_id)))
    check("Memory region hint" in blocks_text and "verify with source memories" in blocks_text)
    check(trace is not None and any(
        isinstance(h, dict) and h.get("region_id") == direct_cid
        for h in getattr(trace, "region_hints", [])
    ))

    detail = {
        "case_id": case.case_id,
        "topic": case.topic,
        "target_id": target_id,
        "nested_id": nested_id,
        "forbidden_ids": sorted(forbidden),
        "dense_only_ids": sorted(off_ids),
        "gist_on_ids": sorted(on_ids),
        "hint_region_ids": sorted(hint_ids),
        "hint_members": sorted(hint_members),
        "hint_hashes": sorted(hint_hashes),
        "hint_raw_uris": sorted(hint_raw_uris),
        "telemetry": telemetry,
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
    with tempfile.TemporaryDirectory(prefix="region-routing-invariant-") as tmp:
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
        "dense_miss_recovery_checks": cases * 3,
        "active_scope_filter_checks": cases * 2,
        "nested_cocoon_checks": cases * 2,
        "proof_link_checks": cases * 2,
        "telemetry_trace_checks": cases * 2,
        "route_only_context_checks": cases,
        "case_type_counts": {"region_routing_cocoon_proof": cases},
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
