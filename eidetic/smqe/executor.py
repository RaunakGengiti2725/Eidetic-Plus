"""SMQE executor: claims first, raw records second, retrieval fallback outside this module."""
from __future__ import annotations

from typing import Iterable

from eidetic.models import ClaimRecord, ExecutionPlan, MemoryRecord, StructuredAnswerResult

from .record_ops import execute_claim_op, execute_record_op


def execute_plan(
    plan: ExecutionPlan,
    query: str,
    *,
    records: Iterable[MemoryRecord],
    claims: Iterable[ClaimRecord],
) -> StructuredAnswerResult | None:
    claim_result = execute_claim_op(plan, query, claims)
    if claim_result is not None:
        return claim_result
    return execute_record_op(plan, query, records)
