"""Rotating agent/project sub-scope invariant eval for SMQE.

Namespace isolation is necessary but not sufficient for multi-agent memory. This sidecar
loads conflicting memories into the same namespace and varies only agent_id/project_id,
then requires the identical question to resolve only against the requested sub-scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from bench.seed_policy import resolve_seed
import tempfile
from pathlib import Path
from typing import Optional

from bench.smqe_scope_invariant import ScopeCase, ScopeSide, _answer_ok, generate_cases
from bench.smqe_synthetic_invariant import _Retriever, _proof_excludes_terms
from eidetic.models import MemoryRecord, Scope
from eidetic.smqe import structured_answer
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.store import RecordStore


def _record(text: str, *, scope: Scope, valid_at: float) -> MemoryRecord:
    digest = hashlib.sha256(f"{scope.key()}\0{text}\0{valid_at}".encode("utf-8")).hexdigest()
    return MemoryRecord(
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        content_hash=f"h-{digest}",
        raw_uri="mem://synthetic-smqe-subscope",
    )


def _load_side(store: RecordStore, scope: Scope, side: ScopeSide, *, add_claims: bool) -> None:
    for text, valid_at in side.rows:
        rec = _record(text, scope=scope, valid_at=valid_at)
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))


def _run_case(case: ScopeCase, *, claims_present: bool) -> tuple[int, int, list[dict], int]:
    with tempfile.TemporaryDirectory(prefix="smqe-subscope-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        namespace = f"smqe-subscope-{case.case_id}"
        left_scope = Scope(namespace=namespace, agent_id="agent-left", project_id="project-left")
        right_scope = Scope(namespace=namespace, agent_id="agent-right", project_id="project-right")
        _load_side(store, left_scope, case.left, add_claims=claims_present)
        _load_side(store, right_scope, case.right, add_claims=claims_present)
        correct = 0
        proof_tokens = 0
        failures: list[dict] = []
        for side_name, scope, side in (("left", left_scope, case.left), ("right", right_scope, case.right)):
            ans = structured_answer(retriever, case.question, at=1_900_000_000, verify=True, scope=scope)
            proof = " ".join(c.snippet for c in (ans.citations if ans else []))
            proof_tokens += sum(max(0, len(c.snippet or "") // 4) for c in (ans.citations if ans else []))
            ok = (
                ans is not None
                and ans.verified
                and _answer_ok(ans.answer, side.expected)
                and _proof_excludes_terms(proof, side.forbidden_in_proof)
            )
            if ok:
                correct += 1
                continue
            failures.append({
                "side": side_name,
                "scope": scope.key(),
                "expected": side.expected,
                "actual": ans.answer if ans else "",
                "note": ans.note if ans else "",
                "verified": bool(ans and ans.verified),
                "proof": proof[:500],
                "forbidden_in_proof": side.forbidden_in_proof,
            })
        return correct, 2, failures, proof_tokens


def run_eval(*, seed: Optional[int] = None, cases: int = 24) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    failures: list[dict] = []
    operator_counts: dict[str, int] = {}
    backend_counts = {"record": 0, "claim": 0}
    total_checks = 0
    correct = 0
    proof_tokens = 0
    for case in generated:
        operator_counts[case.op] = operator_counts.get(case.op, 0) + 1
        for claims_present in (False, True):
            backend = "claim" if claims_present else "record"
            got, checks, local_failures, tokens = _run_case(case, claims_present=claims_present)
            total_checks += checks
            correct += got
            proof_tokens += tokens
            if got == checks:
                backend_counts[backend] += checks
            else:
                failures.append({
                    "case_id": case.case_id,
                    "op": case.op,
                    "question": case.question,
                    "backend": backend,
                    "failures": local_failures,
                })
    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        "checks": total_checks,
        "correct": correct,
        "record_backend_correct": backend_counts["record"],
        "claim_backend_correct": backend_counts["claim"],
        "operator_counts": dict(sorted(operator_counts.items())),
        "backend_counts": {k: v for k, v in sorted(backend_counts.items())},
        "avg_proof_tokens": round(proof_tokens / max(1, total_checks), 2),
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
