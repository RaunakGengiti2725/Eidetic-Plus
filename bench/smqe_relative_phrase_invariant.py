"""Rotating source-relative date phrase invariant for SMQE.

This sidecar checks that source-relative temporal language is normalized from the memory
timestamp rather than returned as an unparsed sentence. It covers both past and future phrasing
across record-only and source-backed claim modes.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import random

from bench.seed_policy import resolve_seed
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from bench.smqe_synthetic_invariant import _Retriever, _answer_matches, _proof_excludes_terms
from eidetic.models import MemoryRecord, Scope
from eidetic.smqe import structured_answer
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.store import RecordStore


@dataclass
class RelativePhraseCase:
    case_id: str
    case_type: str
    question: str
    expected: str
    rows: list[tuple[str, float]]
    forbidden_in_proof: list[str] = field(default_factory=list)


_ITEMS = [
    "cedar permit", "orchid catalog", "harbor map", "kiln checklist", "linen ledger",
    "field badge", "garden invoice", "studio roster",
]
_DISTRACTORS = [
    "backup badge", "travel charger", "river ticket", "archive receipt", "blue notebook",
    "repair token", "market pass", "gallery key",
]


def _pick(rng: random.Random, values: list[str], suffix: str = "") -> str:
    return rng.choice(values) + suffix


def _record(text: str, *, scope: Scope, valid_at: float) -> MemoryRecord:
    digest = hashlib.sha256(f"{scope.namespace}\0{text}\0{valid_at}".encode("utf-8")).hexdigest()
    return MemoryRecord(
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        content_hash=f"h-{digest}",
        raw_uri="mem://synthetic-smqe-relative-phrase",
    )


def _ref(idx: int, *, month: int = 5) -> datetime:
    return datetime(2024, month, 18 + (idx % 5), 12, 0)


def _shift_months(ref: datetime, months_delta: int) -> datetime:
    month_index = ref.month - 1 + months_delta
    year = ref.year + month_index // 12
    month = month_index % 12 + 1
    day = min(ref.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day, ref.hour, ref.minute)


def _ago_days_case(rng: random.Random, idx: int) -> RelativePhraseCase:
    item = _pick(rng, _ITEMS, f" {idx}")
    other = _pick(rng, _DISTRACTORS, f" {idx}")
    days = rng.randint(2, 6)
    ref = _ref(idx)
    expected = (ref - timedelta(days=days)).date().isoformat()
    t = ref.timestamp()
    return RelativePhraseCase(
        case_id=f"ago-days-{idx}",
        case_type="ago_days",
        question=f"When did I file the {item}?",
        expected=expected,
        rows=[
            (f"User: {days} days ago I filed the {item}.", t),
            (f"User: Yesterday I filed the {other}.", t + 1),
        ],
        forbidden_in_proof=[other],
    )


def _ago_weeks_case(rng: random.Random, idx: int) -> RelativePhraseCase:
    item = _pick(rng, _ITEMS, f" {idx}")
    other = _pick(rng, _DISTRACTORS, f" {idx}")
    weeks = rng.randint(2, 4)
    ref = _ref(idx)
    expected = (ref - timedelta(days=weeks * 7)).date().isoformat()
    t = ref.timestamp()
    word = {2: "Two", 3: "Three", 4: "Four"}[weeks]
    return RelativePhraseCase(
        case_id=f"ago-weeks-{idx}",
        case_type="ago_weeks",
        question=f"When did I pick up the {item}?",
        expected=expected,
        rows=[
            (f"User: {word} weeks ago I picked up the {item}.", t),
            (f"User: Tomorrow I will pick up the {other}.", t + 1),
        ],
        forbidden_in_proof=[other],
    )


def _fortnight_case(rng: random.Random, idx: int) -> RelativePhraseCase:
    item = _pick(rng, _ITEMS, f" {idx}")
    other = _pick(rng, _DISTRACTORS, f" {idx}")
    ref = _ref(idx)
    expected = (ref - timedelta(days=14)).date().isoformat()
    t = ref.timestamp()
    return RelativePhraseCase(
        case_id=f"fortnight-{idx}",
        case_type="fortnight_ago",
        question=f"When did I mail the {item}?",
        expected=expected,
        rows=[
            (f"User: A fortnight ago I mailed the {item}.", t),
            (f"User: Last week I mailed the {other}.", t + 1),
        ],
        forbidden_in_proof=[other],
    )


def _in_days_case(rng: random.Random, idx: int) -> RelativePhraseCase:
    item = _pick(rng, _ITEMS, f" {idx}")
    other = _pick(rng, _DISTRACTORS, f" {idx}")
    days = rng.randint(2, 6)
    ref = _ref(idx)
    expected = (ref + timedelta(days=days)).date().isoformat()
    t = ref.timestamp()
    return RelativePhraseCase(
        case_id=f"in-days-{idx}",
        case_type="in_days",
        question=f"When will I inspect the {item}?",
        expected=expected,
        rows=[
            (f"User: In {days} days I will inspect the {item}.", t),
            (f"User: Today I inspected the {other}.", t + 1),
        ],
        forbidden_in_proof=[other],
    )


def _next_week_case(rng: random.Random, idx: int) -> RelativePhraseCase:
    item = _pick(rng, _ITEMS, f" {idx}")
    other = _pick(rng, _DISTRACTORS, f" {idx}")
    ref = _ref(idx)
    start = (ref + timedelta(days=7)).date().isoformat()
    end = (ref + timedelta(days=13)).date().isoformat()
    t = ref.timestamp()
    return RelativePhraseCase(
        case_id=f"next-week-{idx}",
        case_type="next_week",
        question=f"When will I review the {item}?",
        expected=f"the week of {start} to {end}",
        rows=[
            (f"User: Next week I will review the {item}.", t),
            (f"User: Last week I reviewed the {other}.", t + 1),
        ],
        forbidden_in_proof=[other],
    )


def _next_month_case(rng: random.Random, idx: int) -> RelativePhraseCase:
    item = _pick(rng, _ITEMS, f" {idx}")
    other = _pick(rng, _DISTRACTORS, f" {idx}")
    ref = _ref(idx, month=rng.randint(2, 10))
    shifted = _shift_months(ref, 1)
    t = ref.timestamp()
    return RelativePhraseCase(
        case_id=f"next-month-{idx}",
        case_type="next_month",
        question=f"When will I schedule the {item}?",
        expected=f"{calendar.month_name[shifted.month]} {shifted.year:04d}",
        rows=[
            (f"User: Next month I will schedule the {item}.", t),
            (f"User: Last month I scheduled the {other}.", t + 1),
        ],
        forbidden_in_proof=[other],
    )


_GENERATORS: list[Callable[[random.Random, int], RelativePhraseCase]] = [
    _ago_days_case,
    _ago_weeks_case,
    _fortnight_case,
    _in_days_case,
    _next_week_case,
    _next_month_case,
]


def generate_cases(seed: int, cases: int) -> list[RelativePhraseCase]:
    rng = random.Random(seed)
    out = [_GENERATORS[idx % len(_GENERATORS)](rng, idx) for idx in range(cases)]
    rng.shuffle(out)
    return out


def _load_case(store: RecordStore, scope: Scope, case: RelativePhraseCase, *, add_claims: bool) -> None:
    for text, valid_at in case.rows:
        rec = _record(text, scope=scope, valid_at=valid_at)
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))


def _run_once(case: RelativePhraseCase, *, backend: str) -> tuple[bool, dict, int]:
    with tempfile.TemporaryDirectory(prefix=f"smqe-relative-phrase-{backend}-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        scope = Scope(namespace=f"smqe-relative-phrase-{backend}-{case.case_id}")
        _load_case(store, scope, case, add_claims=(backend == "claim"))
        ans = structured_answer(retriever, case.question, at=1_900_000_000, verify=True, scope=scope)
        note = ans.note if ans else ""
        actual_backend = note.split(":")[-1] if note.startswith("smqe:") else ""
        proof = " ".join(c.snippet for c in (ans.citations if ans else []))
        ok = (
            ans is not None
            and ans.verified
            and actual_backend == backend
            and _answer_matches(ans.answer, case.expected)
            and _proof_excludes_terms(proof, case.forbidden_in_proof)
        )
        proof_tokens = sum(max(0, len(c.snippet or "") // 4) for c in (ans.citations if ans else []))
        return ok, {
            "actual": ans.answer if ans else "",
            "note": note,
            "verified": bool(ans and ans.verified),
            "proof": proof[:500],
        }, proof_tokens


def run_eval(*, seed: Optional[int] = None, cases: int = 24) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    failures = []
    case_type_counts: dict[str, int] = {}
    backend_counts = {"claim": 0, "record": 0}
    proof_tokens = 0
    record_correct = 0
    claim_correct = 0
    for case in generated:
        case_type_counts[case.case_type] = case_type_counts.get(case.case_type, 0) + 1
        case_ok = True
        backend_details = {}
        for backend in ("record", "claim"):
            ok, detail, tokens = _run_once(case, backend=backend)
            proof_tokens += tokens
            if ok:
                backend_counts[backend] += 1
                if backend == "record":
                    record_correct += 1
                else:
                    claim_correct += 1
            else:
                case_ok = False
            backend_details[backend] = detail
        if not case_ok:
            failures.append({
                "case_id": case.case_id,
                "case_type": case.case_type,
                "question": case.question,
                "expected": case.expected,
                "forbidden_in_proof": case.forbidden_in_proof,
                "backends": backend_details,
            })
    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        "checks": cases * 2,
        "correct": cases - len(failures),
        "record_backend_correct": record_correct,
        "claim_backend_correct": claim_correct,
        "failures": failures,
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": round(proof_tokens / max(1, cases * 2), 2),
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
