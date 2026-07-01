"""Rotating temporal-window invariant for SMQE aggregates.

This sidecar checks that rolling query windows such as "recently", "past week",
"past N days", "past few months", and "fortnight" constrain aggregate evidence
before the count/sum is computed. It runs every case through both raw-record and
source-backed claim modes and rejects proofs that cite out-of-window distractors.
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

from bench.smqe_synthetic_invariant import _Retriever, _answer_matches, _proof_excludes_terms
from eidetic.models import MemoryRecord, Scope
from eidetic.smqe import structured_answer
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.store import RecordStore


@dataclass
class TemporalWindowCase:
    case_id: str
    case_type: str
    question: str
    expected: str
    expected_op: str
    query_at: float
    rows: list[tuple[str, float]]
    forbidden_in_proof: list[str] = field(default_factory=list)


_TARGETS = [
    ("tea shop", "tea shops"),
    ("record store", "record stores"),
    ("garden center", "garden centers"),
    ("bookstore", "bookstores"),
    ("studio room", "studio rooms"),
    ("market stall", "market stalls"),
]
_LABELS = [
    "Cedar", "Harbor", "Juniper", "Linen", "Maple", "Orchid", "Quartz", "River",
    "Saffron", "Tide", "Umber", "Violet", "Willow", "Yarrow",
]
_AUDIT_TOPICS = ["archive", "studio", "kitchen", "garden", "gallery", "workshop"]
_SOURCE_ITEMS = [
    "copper notebook", "linen bookmark", "orchid journal", "cedar sketchpad",
    "harbor pencil case", "violet travel mug",
]
_SOURCE_PLACES = [
    "Cedar Market", "Harbor Archive", "Juniper Supply", "Linen Depot",
    "Orchid Exchange", "Quartz Studio", "River Bazaar", "Willow Books",
]
_SOURCE_ACTIONS = [
    ("ordered", "picked it up at", "Where did I pick up the {item} recently?"),
    ("needed", "bought it at", "Where did I buy the {item} recently?"),
    ("found", "purchased it from", "Where did I purchase the {item} recently?"),
]


def _record(text: str, *, scope: Scope, valid_at: float) -> MemoryRecord:
    digest = hashlib.sha256(f"{scope.namespace}\0{text}\0{valid_at}".encode("utf-8")).hexdigest()
    return MemoryRecord(
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        content_hash=f"h-{digest}",
        raw_uri="mem://synthetic-smqe-temporal-window",
    )


def _ref(idx: int) -> datetime:
    return datetime(2025, 8 + (idx % 3), 18 + (idx % 7), 12, 0)


def _labels(rng: random.Random, idx: int, count: int) -> list[str]:
    return rng.sample(_LABELS, count)


def _count_case(
    *,
    rng: random.Random,
    idx: int,
    case_type: str,
    question_suffix: str,
    recent_offsets: list[int],
    old_offset: int,
) -> TemporalWindowCase:
    singular, plural = rng.choice(_TARGETS)
    n = rng.randint(2, min(3, len(recent_offsets)))
    labels = _labels(rng, idx, n + 1)
    ref = _ref(idx)
    rows = [
        (f"User: I visited the {labels[pos]} {singular}.", (ref - timedelta(days=offset)).timestamp())
        for pos, offset in enumerate(recent_offsets[:n])
    ]
    old_label = labels[-1]
    rows.append((f"User: I visited the {old_label} {singular}.", (ref - timedelta(days=old_offset)).timestamp()))
    return TemporalWindowCase(
        case_id=f"{case_type}-{idx}",
        case_type=case_type,
        question=f"How many {plural} did I visit {question_suffix}?",
        expected=str(n),
        expected_op="count_aggregate",
        query_at=ref.timestamp(),
        rows=rows,
        forbidden_in_proof=[old_label],
    )


def _recent_count_case(rng: random.Random, idx: int) -> TemporalWindowCase:
    return _count_case(
        rng=rng,
        idx=idx,
        case_type="recent_count",
        question_suffix="recently",
        recent_offsets=[1, 4, 6],
        old_offset=18,
    )


def _past_week_count_case(rng: random.Random, idx: int) -> TemporalWindowCase:
    return _count_case(
        rng=rng,
        idx=idx,
        case_type="past_week_count",
        question_suffix="in the past week",
        recent_offsets=[1, 3, 6],
        old_offset=16,
    )


def _past_days_count_case(rng: random.Random, idx: int) -> TemporalWindowCase:
    days = rng.randint(8, 12)
    return _count_case(
        rng=rng,
        idx=idx,
        case_type="past_days_count",
        question_suffix=f"in the past {days} days",
        recent_offsets=[2, max(3, days // 2), days - 1],
        old_offset=days + 8,
    )


def _fortnight_count_case(rng: random.Random, idx: int) -> TemporalWindowCase:
    return _count_case(
        rng=rng,
        idx=idx,
        case_type="fortnight_count",
        question_suffix="in the past fortnight",
        recent_offsets=[2, 9, 13],
        old_offset=24,
    )


def _past_few_months_count_case(rng: random.Random, idx: int) -> TemporalWindowCase:
    return _count_case(
        rng=rng,
        idx=idx,
        case_type="past_few_months_count",
        question_suffix="in the past few months",
        recent_offsets=[12, 46, 82],
        old_offset=142,
    )


def _recent_hours_sum_case(rng: random.Random, idx: int) -> TemporalWindowCase:
    topic = rng.choice(_AUDIT_TOPICS)
    labels = _labels(rng, idx, 3)
    ref = _ref(idx)
    hours = [rng.randint(1, 4), rng.randint(2, 5)]
    rows = [
        (
            f"User: I spent {hours[0]} hours on the {labels[0]} {topic} audit.",
            (ref - timedelta(days=2)).timestamp(),
        ),
        (
            f"User: I spent {hours[1]} hours on the {labels[1]} {topic} audit.",
            (ref - timedelta(days=5)).timestamp(),
        ),
        (
            f"User: I spent {rng.randint(6, 10)} hours on the {labels[2]} {topic} audit.",
            (ref - timedelta(days=27)).timestamp(),
        ),
    ]
    return TemporalWindowCase(
        case_id=f"recent-hours-sum-{idx}",
        case_type="recent_hours_sum",
        question=f"What is the total number of hours I spent on {topic} audits recently?",
        expected=f"{sum(hours)} hours",
        expected_op="multi_session_sum",
        query_at=ref.timestamp(),
        rows=rows,
        forbidden_in_proof=[labels[2]],
    )


def _list_case(
    *,
    rng: random.Random,
    idx: int,
    case_type: str,
    question_suffix: str,
    recent_offsets: list[int],
    old_offset: int,
) -> TemporalWindowCase:
    singular, plural = rng.choice(_TARGETS)
    labels = _labels(rng, idx, 3)
    ref = _ref(idx)
    rows = [
        (f"User: I visited the {labels[pos]} {singular}.", (ref - timedelta(days=offset)).timestamp())
        for pos, offset in enumerate(recent_offsets[:2])
    ]
    rows.append((f"User: I visited the {labels[2]} {singular}.", (ref - timedelta(days=old_offset)).timestamp()))
    return TemporalWindowCase(
        case_id=f"{case_type}-{idx}",
        case_type=case_type,
        question=f"Which {plural} did I visit {question_suffix}?",
        expected=f"{labels[0]} {singular} and {labels[1]} {singular}",
        expected_op="latest_value",
        query_at=ref.timestamp(),
        rows=rows,
        forbidden_in_proof=[labels[2]],
    )


def _recent_list_case(rng: random.Random, idx: int) -> TemporalWindowCase:
    return _list_case(
        rng=rng,
        idx=idx,
        case_type="recent_list",
        question_suffix="recently",
        recent_offsets=[1, 5],
        old_offset=18,
    )


def _past_week_list_case(rng: random.Random, idx: int) -> TemporalWindowCase:
    return _list_case(
        rng=rng,
        idx=idx,
        case_type="past_week_list",
        question_suffix="in the past week",
        recent_offsets=[2, 6],
        old_offset=19,
    )


def _most_recent_latest_case(rng: random.Random, idx: int) -> TemporalWindowCase:
    singular, _plural = rng.choice(_TARGETS)
    labels = _labels(rng, idx, 3)
    ref = _ref(idx)
    rows = [
        (f"User: I visited the {labels[0]} {singular}.", (ref - timedelta(days=2)).timestamp()),
        (f"User: I visited the {labels[1]} {singular}.", (ref - timedelta(days=5)).timestamp()),
        (f"User: I visited the {labels[2]} {singular}.", (ref - timedelta(days=28)).timestamp()),
    ]
    return TemporalWindowCase(
        case_id=f"most-recent-latest-{idx}",
        case_type="most_recent_latest",
        question=f"What {singular} did I visit most recently?",
        expected=f"{labels[0]} {singular}",
        expected_op="latest_value",
        query_at=ref.timestamp(),
        rows=rows,
        forbidden_in_proof=[labels[1], labels[2]],
    )


def _source_location_window_case(rng: random.Random, idx: int) -> TemporalWindowCase:
    item = rng.choice(_SOURCE_ITEMS)
    recent_place, old_place = rng.sample(_SOURCE_PLACES, 2)
    ref = _ref(idx)
    rows = [
        (
            f"User: I bought a {item} for travel notes. I got it from {recent_place}.",
            (ref - timedelta(days=2)).timestamp(),
        ),
        (
            f"User: I bought a {item} for travel notes. "
            f"I got it from {old_place} after asking where to get the {item}.",
            (ref - timedelta(days=24)).timestamp(),
        ),
    ]
    return TemporalWindowCase(
        case_id=f"source-location-window-{idx}",
        case_type="source_location_window",
        question=f"Where did I get the {item} recently?",
        expected=recent_place,
        expected_op="latest_value",
        query_at=ref.timestamp(),
        rows=rows,
        forbidden_in_proof=[old_place],
    )


def _source_action_variant_window_case(rng: random.Random, idx: int) -> TemporalWindowCase:
    item = rng.choice(_SOURCE_ITEMS)
    setup_action, source_phrase, question_template = rng.choice(_SOURCE_ACTIONS)
    recent_place, old_place = rng.sample(_SOURCE_PLACES, 2)
    ref = _ref(idx)
    rows = [
        (
            f"User: I {setup_action} a {item} for travel notes. I {source_phrase} {recent_place}.",
            (ref - timedelta(days=2)).timestamp(),
        ),
        (
            f"User: I {setup_action} a {item} for travel notes. "
            f"I {source_phrase} {old_place} after asking where to get the {item}.",
            (ref - timedelta(days=24)).timestamp(),
        ),
    ]
    return TemporalWindowCase(
        case_id=f"source-action-variant-window-{idx}",
        case_type="source_action_variant_window",
        question=question_template.format(item=item),
        expected=recent_place,
        expected_op="latest_value",
        query_at=ref.timestamp(),
        rows=rows,
        forbidden_in_proof=[old_place],
    )


_GENERATORS: list[Callable[[random.Random, int], TemporalWindowCase]] = [
    _recent_count_case,
    _past_week_count_case,
    _past_days_count_case,
    _fortnight_count_case,
    _past_few_months_count_case,
    _recent_hours_sum_case,
    _recent_list_case,
    _past_week_list_case,
    _most_recent_latest_case,
    _source_location_window_case,
    _source_action_variant_window_case,
]


def generate_cases(seed: int, cases: int) -> list[TemporalWindowCase]:
    rng = random.Random(seed)
    out = [_GENERATORS[idx % len(_GENERATORS)](rng, idx) for idx in range(cases)]
    rng.shuffle(out)
    return out


def _load_case(store: RecordStore, scope: Scope, case: TemporalWindowCase, *, add_claims: bool) -> None:
    for text, valid_at in case.rows:
        rec = _record(text, scope=scope, valid_at=valid_at)
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))


def _run_once(case: TemporalWindowCase, *, backend: str) -> tuple[bool, dict, int]:
    with tempfile.TemporaryDirectory(prefix=f"smqe-temporal-window-{backend}-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        scope = Scope(namespace=f"smqe-temporal-window-{backend}-{case.case_id}")
        _load_case(store, scope, case, add_claims=(backend == "claim"))
        ans = structured_answer(retriever, case.question, at=case.query_at, verify=True, scope=scope)
        note = ans.note if ans else ""
        parts = note.split(":") if note.startswith("smqe:") else []
        actual_op = parts[1] if len(parts) >= 3 else ""
        actual_backend = parts[-1] if len(parts) >= 3 else ""
        proof = " ".join(c.snippet for c in (ans.citations if ans else []))
        ok = (
            ans is not None
            and ans.verified
            and actual_op == case.expected_op
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
                "expected_op": case.expected_op,
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
