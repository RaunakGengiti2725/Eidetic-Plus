"""Single structured-answer pipeline for SMQE.

This module is the public implementation point: plan once, execute claims first and records
second, then convert only verified structured results into normal Eidetic answers.
"""
from __future__ import annotations

from typing import Optional

import time

from eidetic.models import Answer, MemoryRecord, Scope, now

from .executor import execute_plan
from .planner import plan_query
from .verify import answer_from_result


def _record_active_at(rec: MemoryRecord, t: float, scope: Scope) -> bool:
    return (
        rec.scope.visible_to(scope)
        and rec.valid_at <= t
        and (rec.invalid_at is None or rec.invalid_at > t)
        and (rec.expired_at is None or rec.expired_at > t)
    )


def structured_answer(
    retriever,
    query: str,
    records: Optional[list[MemoryRecord]] = None,
    at: Optional[float] = None,
    *,
    verify: bool = True,
    scope: Optional[Scope] = None,
) -> Optional[Answer]:
    out = structured_recall(
        retriever,
        query,
        records=records,
        at=at,
        verify=verify,
        scope=scope,
    )
    return out.get("_answer_model")


def structured_recall(
    retriever,
    query: str,
    records: Optional[list[MemoryRecord]] = None,
    at: Optional[float] = None,
    *,
    verify: bool = True,
    scope: Optional[Scope] = None,
) -> dict:
    """Run SMQE and return the auditable plan/execute/verify trace.

    This is the host-facing structured memory surface. It performs no generation: claims are tried
    first, active raw records second, and any confident answer must verify against source memory
    proof atoms before it is marked answered.
    """
    t0 = time.perf_counter()
    at = now() if at is None else at
    scope = scope or (records[0].scope if records else Scope())
    store = getattr(retriever, "store", None)
    if store is None:
        return {
            "ok": False,
            "answered": False,
            "abstained": True,
            "failure_reason": "missing_store",
            "query": query,
            "scope": scope.model_dump(),
            "as_of": at,
            "latency_ms": {"total": (time.perf_counter() - t0) * 1000.0},
        }
    active_records_at = getattr(store, "active_records_at", None)
    active_records = [rec for rec in records if _record_active_at(rec, at, scope)] if records is not None else (
        active_records_at(at, scope) if callable(active_records_at) else []
    )
    claims = []
    active_claims = getattr(store, "active_claims_at", None)
    if callable(active_claims):
        claims = active_claims(at, scope)
    plan = plan_query(query, at)
    base = {
        "ok": True,
        "answered": False,
        "abstained": True,
        "query": query,
        "scope": scope.model_dump(),
        "as_of": at,
        "plan": plan.model_dump(),
        "op": plan.op,
        "backend": "",
        "answer": "",
        "verified": False,
        "confidence": 0.0,
        "supports": [],
        "citations": [],
        "generated_by": "smqe",
        "note": "",
        "failure_reason": "",
        "active_record_count": len(active_records),
        "active_claim_count": len(claims),
    }
    if not active_records:
        return {
            **base,
            "failure_reason": "no_active_records",
            "latency_ms": {"total": (time.perf_counter() - t0) * 1000.0},
        }
    result = execute_plan(plan, query, records=active_records, claims=claims)
    if result is None:
        return {
            **base,
            "failure_reason": "no_structured_result",
            "latency_ms": {"total": (time.perf_counter() - t0) * 1000.0},
        }
    answer = answer_from_result(retriever, query, result, verify=verify)
    supports = [s.model_dump() for s in result.supports]
    if answer is None:
        return {
            **base,
            "backend": result.backend,
            "answer": result.answer,
            "confidence": result.confidence,
            "supports": supports,
            "note": result.note or f"smqe:{result.op}:{result.backend}",
            "failure_reason": "verification_failed" if verify else "answer_conversion_failed",
            "latency_ms": {"total": (time.perf_counter() - t0) * 1000.0},
        }
    return {
        **base,
        "_answer_model": answer,
        "answered": True,
        "abstained": False,
        "backend": result.backend,
        "answer": answer.answer,
        "verified": answer.verified,
        "confidence": answer.confidence,
        "supports": supports,
        "citations": [c.model_dump(mode="json") for c in answer.citations],
        "note": answer.note,
        "failure_reason": "",
        "latency_ms": {"total": (time.perf_counter() - t0) * 1000.0},
    }
