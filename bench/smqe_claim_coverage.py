"""Rotating synthetic claim-coverage eval for SMQE.

Unlike ``smqe_synthetic_invariant``, this forces the answer to come from claims generated from
source-backed memory records. It keeps pressure on tier-1 coverage so the product does not quietly
depend on raw record scanning for every structured memory question.
"""
from __future__ import annotations

import argparse
import json
import random

from bench.seed_policy import resolve_seed
import tempfile
from pathlib import Path
from typing import Optional

from bench.smqe_synthetic_invariant import (
    _Retriever,
    _answer_matches,
    _proof_excludes_terms,
    _record,
    generate_cases,
)
from eidetic.models import Scope
from eidetic.smqe import structured_answer
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.store import RecordStore


def run_eval(*, seed: Optional[int] = None, cases: int = 24) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    failures = []
    op_counts: dict[str, int] = {}
    claim_backend_operator_counts: dict[str, int] = {}
    claim_type_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    proof_tokens = 0
    total_claims = 0
    with tempfile.TemporaryDirectory(prefix="smqe-claimcov-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        for case in generated:
            op_counts[case.op] = op_counts.get(case.op, 0) + 1
            scope = Scope(namespace=f"smqe-claimcov-{case.case_id}")
            case_claims = 0
            case_types: dict[str, int] = {}
            for text, valid_at in case.rows:
                rec = _record(text, scope=scope, valid_at=valid_at)
                store.upsert_record(rec)
                claims = claims_for_record(rec)
                for claim in claims:
                    case_types[claim.claim_type] = case_types.get(claim.claim_type, 0) + 1
                    claim_type_counts[claim.claim_type] = claim_type_counts.get(claim.claim_type, 0) + 1
                case_claims += store.add_claims(claims)
            total_claims += case_claims
            ans = structured_answer(retriever, case.question, at=1_800_000_000, verify=True, scope=scope)
            proof = " ".join(c.snippet for c in (ans.citations if ans else []))
            note = ans.note if ans else ""
            _parts = note.split(":") if note.startswith("smqe:") else []
            backend = _parts[2] if len(_parts) >= 3 else ""
            if backend:
                backend_counts[backend] = backend_counts.get(backend, 0) + 1
            if backend == "claim":
                claim_backend_operator_counts[case.op] = (
                    claim_backend_operator_counts.get(case.op, 0) + 1
                )
            if ans is not None:
                proof_tokens += sum(max(0, len(c.snippet or "") // 4) for c in ans.citations)
            if getattr(case, "expect_abstain", False):
                # P0 fail-closed (2026-07-09): a DERIVED count/sum no longer verifies via the
                # claim backend (eidetic/smqe/verify.py) -- it abstains. Claims are still
                # extracted (case_claims may be >0); only the verified answer is withheld.
                ok = ans is None and case_claims > 0
            else:
                ok = (
                    ans is not None
                    and ans.verified
                    and backend == "claim"
                    and _answer_matches(ans.answer, case.expected)
                    and case_claims > 0
                    and _proof_excludes_terms(proof, case.forbidden_in_proof)
                )
            if not ok:
                failures.append({
                    "case_id": case.case_id,
                    "op": case.op,
                    "question": case.question,
                    "expected": case.expected,
                    "actual": ans.answer if ans else "",
                    "note": note,
                    "verified": bool(ans and ans.verified),
                    "claims": case_claims,
                    "claim_types": dict(sorted(case_types.items())),
                    "proof": proof[:500],
                })
    expected_abstain_cases = sum(1 for c in generated if getattr(c, "expect_abstain", False))
    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        # P0 fail-closed (2026-07-09): derived count/sum cases assert abstention, so they can
        # never be claim-backed. The gate computes claim-backend rate over ANSWERABLE cases
        # (cases - expected_abstain_cases); publishing the count here keeps that scoping honest
        # and auditable rather than silently shrinking the denominator.
        "expected_abstain_cases": expected_abstain_cases,
        "correct": cases - len(failures),
        "claim_backend_correct": cases - len(failures),
        "claims_extracted": total_claims,
        "avg_claims_per_case": round(total_claims / cases, 2),
        "failures": failures,
        "operator_counts": dict(sorted(op_counts.items())),
        "claim_backend_operator_counts": dict(sorted(claim_backend_operator_counts.items())),
        "claim_type_counts": dict(sorted(claim_type_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": round(proof_tokens / cases, 2),
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
