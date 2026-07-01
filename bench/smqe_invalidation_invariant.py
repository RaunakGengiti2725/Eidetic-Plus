"""Rotating invalidation invariant eval for SMQE.

Corrected or expired memories must stop supporting answers after their invalidation time. This
sidecar loads old records with `invalid_at` plus corrected records in the same scope, then asks the
same question before and after invalidation. Both record-only and claim-backed paths must drop the
invalidated proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from bench.seed_policy import resolve_seed
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from bench.smqe_synthetic_invariant import _Retriever, _proof_excludes_terms
from eidetic.models import MemoryRecord, Scope
from eidetic.smqe import structured_answer
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.store import RecordStore


@dataclass
class InvalidationRow:
    text: str
    valid_at: float
    invalid_at: Optional[float] = None
    expired_at: Optional[float] = None


@dataclass
class InvalidationCase:
    case_id: str
    op: str
    question: str
    rows: list[InvalidationRow]
    before_at: float
    after_at: float
    before_expected: str
    after_expected: str
    before_forbidden_in_proof: list[str] = field(default_factory=list)
    after_forbidden_in_proof: list[str] = field(default_factory=list)


_NAMES = ["Ari", "Nila", "Mika", "Sana", "Theo", "Rowan", "Lina", "Owen"]
_OBJECTS = ["backup badge", "kiln token", "garden permit", "field notebook", "studio key", "travel charger"]
_LOCATIONS = [
    "Quartz Loft", "North Pier Studio", "Cedar Annex", "Blue Finch Lab", "Orchid Room",
    "Harbor Desk", "Maple Archive", "River Gate", "Juniper Shelf",
]
_TARGETS = ["ceramic studios", "tea shops", "library workshops", "bike routes", "garden plots"]
_TABLE_VALUES = ["7 AM", "late", "north desk", "2 PM", "midday", "west desk"]
_PREFS = ["mint tea", "fantasy novels", "graphite pens", "berry salad", "quiet playlists", "linen maps"]
_PROJECTS = ["mural ledger", "orchid catalog", "harbor map", "kiln checklist", "field guide"]


def _pick(rng: random.Random, values: list[str], suffix: str = "") -> str:
    return rng.choice(values) + suffix


def _record(row: InvalidationRow, *, scope: Scope) -> MemoryRecord:
    digest = hashlib.sha256(
        f"{scope.namespace}\0{row.text}\0{row.valid_at}\0{row.invalid_at}\0{row.expired_at}".encode("utf-8")
    ).hexdigest()
    return MemoryRecord(
        text=row.text,
        source="user",
        scope=scope,
        valid_at=row.valid_at,
        invalid_at=row.invalid_at,
        expired_at=row.expired_at,
        content_hash=f"h-{digest}",
        raw_uri="mem://synthetic-smqe-invalidation",
    )


def _latest_case(rng: random.Random, idx: int) -> InvalidationCase:
    name = _pick(rng, _NAMES)
    obj = _pick(rng, _OBJECTS, f" {idx}")
    old_loc, new_loc = rng.sample(_LOCATIONS, k=2)
    t = 1_706_000_000 + idx * 1_000
    invalid_at = t + 400
    return InvalidationCase(
        case_id=f"latest-invalid-{idx}",
        op="latest_value",
        question=f"Where does {name} keep the {obj}?",
        rows=[
            InvalidationRow(f"{name}: I keep the {obj} at {old_loc}.", t, invalid_at),
            InvalidationRow(f"{name}: I keep the {obj} at {new_loc}.", t + 500),
        ],
        before_at=t + 100,
        after_at=t + 900,
        before_expected=old_loc,
        after_expected=new_loc,
        before_forbidden_in_proof=[new_loc],
        after_forbidden_in_proof=[old_loc],
    )


def _expired_source_case(rng: random.Random, idx: int) -> InvalidationCase:
    name = _pick(rng, _NAMES)
    obj = _pick(rng, _OBJECTS, f" {idx}")
    active_loc, expired_loc = rng.sample(_LOCATIONS, k=2)
    t = 1_706_050_000 + idx * 1_000
    expired_at = t + 400
    return InvalidationCase(
        case_id=f"latest-expired-source-{idx}",
        op="latest_value",
        question=f"Where does {name} keep the {obj}?",
        rows=[
            InvalidationRow(f"{name}: I keep the {obj} at {active_loc}.", t),
            InvalidationRow(f"{name}: I keep the {obj} at {expired_loc}.", t + 100, expired_at=expired_at),
        ],
        before_at=t + 200,
        after_at=t + 900,
        before_expected=expired_loc,
        after_expected=active_loc,
        before_forbidden_in_proof=[active_loc],
        after_forbidden_in_proof=[expired_loc],
    )


def _count_case(rng: random.Random, idx: int) -> InvalidationCase:
    target = _pick(rng, _TARGETS)
    singular = target[:-1] if target.endswith("s") else target
    old_n = rng.randint(2, 4)
    new_n = rng.randint(5, 7)
    old_labels = [f"Old {label}" for label in rng.sample(_LOCATIONS, k=old_n)]
    new_labels = [f"New {label}" for label in rng.sample(_LOCATIONS, k=new_n)]
    t = 1_706_100_000 + idx * 1_000
    invalid_at = t + 400
    rows = [
        InvalidationRow(f"User: I visited the {old_labels[j]} {singular} this month.", t + j, invalid_at)
        for j in range(old_n)
    ]
    rows.extend(
        InvalidationRow(f"User: I visited the {new_labels[j]} {singular} this month.", t + 500 + j)
        for j in range(new_n)
    )
    return InvalidationCase(
        case_id=f"count-invalid-{idx}",
        op="count_aggregate",
        question=f"How many {target} did I visit this month?",
        rows=rows,
        before_at=t + 100,
        after_at=t + 900,
        before_expected=str(old_n),
        after_expected=str(new_n),
        before_forbidden_in_proof=new_labels,
        after_forbidden_in_proof=old_labels,
    )


def _relative_case(rng: random.Random, idx: int) -> InvalidationCase:
    item = _pick(rng, _OBJECTS, f" {idx}")
    old_ref = datetime(2024, rng.randint(1, 4), rng.randint(10, 18), 12, 0)
    new_ref = old_ref + timedelta(days=rng.randint(20, 40))
    invalid_at = (old_ref + timedelta(days=5)).timestamp()
    old_expected = (old_ref - timedelta(days=1)).date().isoformat()
    new_expected = (new_ref - timedelta(days=1)).date().isoformat()
    return InvalidationCase(
        case_id=f"relative-invalid-{idx}",
        op="relative_temporal",
        question=f"When did I pick up the {item}?",
        rows=[
            InvalidationRow(f"User: Yesterday I picked up the {item}.", old_ref.timestamp(), invalid_at),
            InvalidationRow(f"User: Yesterday I picked up the {item}.", new_ref.timestamp()),
        ],
        before_at=(old_ref + timedelta(days=1)).timestamp(),
        after_at=(new_ref + timedelta(days=1)).timestamp(),
        before_expected=old_expected,
        after_expected=new_expected,
        before_forbidden_in_proof=[new_expected],
        after_forbidden_in_proof=[old_expected],
    )


def _table_case(rng: random.Random, idx: int) -> InvalidationCase:
    person = _pick(rng, _NAMES)
    day = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"])
    old_value, new_value = rng.sample(_TABLE_VALUES, k=2)
    t = 1_706_200_000 + idx * 1_000
    invalid_at = t + 400
    return InvalidationCase(
        case_id=f"table-invalid-{idx}",
        op="table_lookup",
        question=f"What shift does {person} have on {day} in the schedule?",
        rows=[
            InvalidationRow(f"| Name | {day} |\n| {person} | {old_value} |", t, invalid_at),
            InvalidationRow(f"| Name | {day} |\n| {person} | {new_value} |", t + 500),
        ],
        before_at=t + 100,
        after_at=t + 900,
        before_expected=old_value,
        after_expected=new_value,
        before_forbidden_in_proof=[new_value],
        after_forbidden_in_proof=[old_value],
    )


def _preference_case(rng: random.Random, idx: int) -> InvalidationCase:
    old_good, new_good = rng.sample(_PREFS, k=2)
    old = f"{old_good} {idx}"
    new = f"{new_good} {idx}"
    t = 1_706_300_000 + idx * 1_000
    invalid_at = t + 400
    return InvalidationCase(
        case_id=f"preference-invalid-{idx}",
        op="preference_synth",
        question=f"Would I prefer {old} or {new}?",
        rows=[
            InvalidationRow(f"User: I enjoy {old} after work.", t, invalid_at),
            InvalidationRow(f"User: I avoid {new} before meetings.", t + 1, invalid_at),
            InvalidationRow(f"User: I avoid {old} before meetings now.", t + 500),
            InvalidationRow(f"User: I enjoy {new} after work now.", t + 501),
        ],
        before_at=t + 100,
        after_at=t + 900,
        before_expected=old,
        after_expected=new,
        before_forbidden_in_proof=[f"enjoy {new}", f"avoid {old}"],
        after_forbidden_in_proof=[f"enjoy {old}", f"avoid {new}"],
    )


def _speaker_case(rng: random.Random, idx: int) -> InvalidationCase:
    speaker = _pick(rng, _NAMES)
    topic = _pick(rng, _OBJECTS, f" {idx}")
    old_loc, new_loc = [value.lower() for value in rng.sample(_LOCATIONS, k=2)]
    t = 1_706_400_000 + idx * 1_000
    invalid_at = t + 400
    old_expected = f"the {topic} stays in the {old_loc}"
    new_expected = f"the {topic} stays in the {new_loc}"
    return InvalidationCase(
        case_id=f"speaker-invalid-{idx}",
        op="speaker_fact",
        question=f"What did {speaker} say about the {topic}?",
        rows=[
            InvalidationRow(f"{speaker}: I said the {topic} stays in the {old_loc}.", t, invalid_at),
            InvalidationRow(f"{speaker}: I said the {topic} stays in the {new_loc}.", t + 500),
        ],
        before_at=t + 100,
        after_at=t + 900,
        before_expected=old_expected,
        after_expected=new_expected,
        before_forbidden_in_proof=[new_loc],
        after_forbidden_in_proof=[old_loc],
    )


def _temporal_delta_case(rng: random.Random, idx: int) -> InvalidationCase:
    start_item = _pick(rng, _OBJECTS, f" {idx}")
    finish_item = _pick(rng, [x for x in _OBJECTS if x not in start_item], f" {idx}")
    old_days = rng.randint(3, 7)
    new_days = rng.randint(8, 13)
    start = datetime(2024, rng.randint(1, 7), rng.randint(2, 10), 12, 0)
    future_start = start + timedelta(days=20)
    invalid_at = (future_start - timedelta(days=2)).timestamp()
    return InvalidationCase(
        case_id=f"delta-invalid-{idx}",
        op="temporal_delta",
        question=(
            f"How many days passed between the day I started calibrating the {start_item} "
            f"and the day I finished installing the {finish_item}?"
        ),
        rows=[
            InvalidationRow(f"User: I started calibrating the {start_item} today.", start.timestamp(), invalid_at),
            InvalidationRow(f"User: I finished installing the {finish_item} today.", (start + timedelta(days=old_days)).timestamp(), invalid_at),
            InvalidationRow(f"User: I started calibrating the {start_item} today.", future_start.timestamp()),
            InvalidationRow(f"User: I finished installing the {finish_item} today.", (future_start + timedelta(days=new_days)).timestamp()),
        ],
        before_at=(start + timedelta(days=old_days + 1)).timestamp(),
        after_at=(future_start + timedelta(days=new_days + 1)).timestamp(),
        before_expected=f"{old_days} days",
        after_expected=f"{new_days} days",
        before_forbidden_in_proof=[f"{new_days} days"],
        after_forbidden_in_proof=[f"{old_days} days"],
    )


def _sum_case(rng: random.Random, idx: int) -> InvalidationCase:
    project = _pick(rng, _PROJECTS, f" {idx}")
    old_a, old_b = rng.randint(1, 3), rng.randint(2, 4)
    new_a, new_b = rng.randint(5, 7), rng.randint(6, 8)
    old_total = old_a + old_b
    new_total = new_a + new_b
    t = 1_706_500_000 + idx * 1_000
    invalid_at = t + 400
    return InvalidationCase(
        case_id=f"sum-invalid-{idx}",
        op="multi_session_sum",
        question=f"How many total hours did I spend on the {project}?",
        rows=[
            InvalidationRow(f"User: I spent {old_a} hours on the {project}.", t, invalid_at),
            InvalidationRow(f"User: I spent {old_b} hours on the {project}.", t + 1, invalid_at),
            InvalidationRow(f"User: I spent {new_a} hours on the {project}.", t + 500),
            InvalidationRow(f"User: I spent {new_b} hours on the {project}.", t + 501),
        ],
        before_at=t + 100,
        after_at=t + 900,
        before_expected=f"{old_total} hours",
        after_expected=f"{new_total} hours",
        before_forbidden_in_proof=[f"{new_a} hours", f"{new_b} hours"],
        after_forbidden_in_proof=[f"{old_a} hours", f"{old_b} hours"],
    )


_GENERATORS: list[Callable[[random.Random, int], InvalidationCase]] = [
    _latest_case,
    _expired_source_case,
    _count_case,
    _relative_case,
    _table_case,
    _preference_case,
    _speaker_case,
    _temporal_delta_case,
    _sum_case,
]


def _case_type(case: InvalidationCase) -> str:
    if case.case_id.startswith("latest-expired-source-"):
        return "source_expiration"
    if case.case_id.startswith("latest-invalid-"):
        return "latest_supersession"
    if case.case_id.startswith("count-invalid-"):
        return "count_supersession"
    if case.case_id.startswith("relative-invalid-"):
        return "relative_temporal_supersession"
    if case.case_id.startswith("table-invalid-"):
        return "table_lookup_supersession"
    if case.case_id.startswith("preference-invalid-"):
        return "preference_supersession"
    if case.case_id.startswith("speaker-invalid-"):
        return "speaker_fact_supersession"
    if case.case_id.startswith("delta-invalid-"):
        return "temporal_delta_supersession"
    if case.case_id.startswith("sum-invalid-"):
        return "multi_session_sum_supersession"
    return case.op


def generate_cases(seed: int, cases: int) -> list[InvalidationCase]:
    rng = random.Random(seed)
    out = [_GENERATORS[idx % len(_GENERATORS)](rng, idx) for idx in range(cases)]
    rng.shuffle(out)
    return out


def _load_case(store: RecordStore, scope: Scope, case: InvalidationCase, *, add_claims: bool) -> None:
    for row in case.rows:
        rec = _record(row, scope=scope)
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))


def _answer_ok(answer: str, expected: str) -> bool:
    return (expected or "").lower() in (answer or "").lower()


def _run_case(case: InvalidationCase, *, claims_present: bool) -> tuple[int, int, list[dict], int]:
    with tempfile.TemporaryDirectory(prefix="smqe-invalid-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        scope = Scope(namespace=f"smqe-invalid-{case.case_id}-{'claim' if claims_present else 'record'}")
        _load_case(store, scope, case, add_claims=claims_present)
        checks = [
            ("before", case.before_at, case.before_expected, case.before_forbidden_in_proof),
            ("after", case.after_at, case.after_expected, case.after_forbidden_in_proof),
        ]
        correct = 0
        proof_tokens = 0
        failures: list[dict] = []
        for stage, at, expected, forbidden in checks:
            ans = structured_answer(retriever, case.question, at=at, verify=True, scope=scope)
            proof = " ".join(c.snippet for c in (ans.citations if ans else []))
            proof_tokens += sum(max(0, len(c.snippet or "") // 4) for c in (ans.citations if ans else []))
            ok = (
                ans is not None
                and ans.verified
                and _answer_ok(ans.answer, expected)
                and _proof_excludes_terms(proof, forbidden)
            )
            if ok:
                correct += 1
                continue
            failures.append({
                "stage": stage,
                "at": at,
                "expected": expected,
                "actual": ans.answer if ans else "",
                "note": ans.note if ans else "",
                "verified": bool(ans and ans.verified),
                "proof": proof[:500],
                "forbidden_in_proof": forbidden,
            })
        return correct, len(checks), failures, proof_tokens


def run_eval(*, seed: Optional[int] = None, cases: int = 24) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    failures: list[dict] = []
    operator_counts: dict[str, int] = {}
    case_type_counts: dict[str, int] = {}
    backend_counts = {"record": 0, "claim": 0}
    preference_supersession_cases = 0
    preference_supersession_checks = 0
    preference_supersession_correct = 0
    preference_supersession_record_correct = 0
    preference_supersession_claim_correct = 0
    total_checks = 0
    correct = 0
    proof_tokens = 0
    for case in generated:
        operator_counts[case.op] = operator_counts.get(case.op, 0) + 1
        case_type = _case_type(case)
        case_type_counts[case_type] = case_type_counts.get(case_type, 0) + 1
        is_preference_supersession = case_type == "preference_supersession"
        if is_preference_supersession:
            preference_supersession_cases += 1
        for claims_present in (False, True):
            backend = "claim" if claims_present else "record"
            got, checks, local_failures, tokens = _run_case(case, claims_present=claims_present)
            total_checks += checks
            correct += got
            proof_tokens += tokens
            if is_preference_supersession:
                preference_supersession_checks += checks
                preference_supersession_correct += got
                if claims_present:
                    preference_supersession_claim_correct += got
                else:
                    preference_supersession_record_correct += got
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
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "backend_counts": {k: v for k, v in sorted(backend_counts.items())},
        "preference_supersession_pass": (
            preference_supersession_cases > 0
            and preference_supersession_checks == preference_supersession_cases * 4
            and preference_supersession_correct == preference_supersession_checks
            and preference_supersession_record_correct == preference_supersession_cases * 2
            and preference_supersession_claim_correct == preference_supersession_cases * 2
        ),
        "preference_supersession_cases": preference_supersession_cases,
        "preference_supersession_checks": preference_supersession_checks,
        "preference_supersession_correct": preference_supersession_correct,
        "preference_supersession_record_correct": preference_supersession_record_correct,
        "preference_supersession_claim_correct": preference_supersession_claim_correct,
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
