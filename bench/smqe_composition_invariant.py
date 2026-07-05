"""Rotating multi-record composition invariant for SMQE.

This sidecar targets questions that need a small join over memories rather than a single slot:
shared values across named people, event ordering, and a clock-time lookup relative to another event.
It is dataset-neutral and runs against both record-only and source-backed claim modes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from bench.seed_policy import resolve_seed
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from bench.smqe_synthetic_invariant import _Retriever, _answer_matches, _proof_excludes_terms
from eidetic.models import MemoryRecord, Scope
from eidetic.smqe import structured_answer
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.store import RecordStore


@dataclass
class CompositionCase:
    case_id: str
    case_type: str
    question: str
    expected: str
    rows: list[tuple[str, float]]
    forbidden_in_proof: list[str] = field(default_factory=list)


_NAMES = ["Ari", "Nila", "Mika", "Sana", "Theo", "Rowan", "Lina", "Owen", "Tessa", "Ira"]
_ACTIVITIES = ["sketching maps", "folding paper cranes", "restoring old radios", "pressing wildflowers"]
_TOPICS = ["marine robotics", "ceramic chemistry", "urban beekeeping", "archival audio repair"]
_PLACES = ["Cedar Clinic", "Harbor Pantry", "Juniper Library", "Quartz Community Garden"]
_SKILLS = ["sign painting", "bike maintenance", "linen bookbinding", "field sketching"]
_EVENTS = ["lantern workshop", "river archive tour", "kiln safety class", "orchid catalog session"]
_APPOINTMENTS = ["dentist appointment", "permit meeting", "clinic checkup", "studio review"]


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
        raw_uri="mem://synthetic-smqe-composition",
    )


def _shared_unwind_case(rng: random.Random, idx: int) -> CompositionCase:
    left, right, other = rng.sample(_NAMES, k=3)
    expected, distractor = rng.sample(_ACTIVITIES, k=2)
    t = 1_705_000_000 + idx * 100
    return CompositionCase(
        case_id=f"shared-unwind-{idx}",
        case_type="shared_value",
        question=f"What activity do {left} and {right} both use to unwind?",
        expected=expected,
        rows=[
            (f"User: {left} unwinds by {expected} after work.", t),
            (f"User: {right} unwinds by {expected} after work.", t + 1),
            (f"User: {other} unwinds by {distractor} after work.", t + 2),
        ],
        forbidden_in_proof=[distractor],
    )


def _shared_research_case(rng: random.Random, idx: int) -> CompositionCase:
    left, right, other = rng.sample(_NAMES, k=3)
    expected, distractor = rng.sample(_TOPICS, k=2)
    t = 1_705_100_000 + idx * 100
    return CompositionCase(
        case_id=f"shared-research-{idx}",
        case_type="shared_value",
        question=f"What topic are {left} and {right} both researching?",
        expected=expected,
        rows=[
            (f"User: {left} is researching {expected} for the seminar.", t),
            (f"User: {right} is researching {expected} for the seminar.", t + 1),
            (f"User: {other} is researching {distractor} for the seminar.", t + 2),
        ],
        forbidden_in_proof=[distractor],
    )


def _shared_volunteer_case(rng: random.Random, idx: int) -> CompositionCase:
    left, right, other = rng.sample(_NAMES, k=3)
    expected, distractor = rng.sample(_PLACES, k=2)
    t = 1_705_200_000 + idx * 100
    return CompositionCase(
        case_id=f"shared-volunteer-{idx}",
        case_type="shared_value",
        question=f"Where do {left} and {right} both volunteer?",
        expected=expected,
        rows=[
            (f"User: {left} volunteers at {expected} on Mondays.", t),
            (f"User: {right} volunteers at {expected} on Thursdays.", t + 1),
            (f"User: {other} volunteers at {distractor} on Saturdays.", t + 2),
        ],
        forbidden_in_proof=[distractor],
    )


def _shared_learning_case(rng: random.Random, idx: int) -> CompositionCase:
    left, right, other = rng.sample(_NAMES, k=3)
    expected, distractor = rng.sample(_SKILLS, k=2)
    t = 1_705_300_000 + idx * 100
    return CompositionCase(
        case_id=f"shared-learning-{idx}",
        case_type="shared_value",
        question=f"What skill are {left} and {right} both learning?",
        expected=expected,
        rows=[
            (f"User: {left} is learning {expected} with a weekend group.", t),
            (f"User: {right} is learning {expected} with a weekend group.", t + 1),
            (f"User: {other} is learning {distractor} with a weekend group.", t + 2),
        ],
        forbidden_in_proof=[distractor],
    )


def _event_order_case(rng: random.Random, idx: int) -> CompositionCase:
    first, second = rng.sample(_EVENTS, k=2)
    start_day = rng.randint(3, 15)
    later_day = start_day + rng.randint(2, 8)
    t = 1_705_400_000 + idx * 100
    return CompositionCase(
        case_id=f"event-order-{idx}",
        case_type="event_order",
        question=f"Which happened first, attending the {first} or attending the {second}?",
        expected=f"the {first}",
        rows=[
            (f"User: On 2024-03-{start_day:02d} I attended the {first}.", t),
            (f"User: On 2024-03-{later_day:02d} I attended the {second}.", t + 1),
        ],
        forbidden_in_proof=[second],
    )


def _before_time_case(rng: random.Random, idx: int) -> CompositionCase:
    appointment = _pick(rng, _APPOINTMENTS)
    hour = rng.randint(9, 11)
    minute = rng.choice([5, 10, 15, 25, 40, 50])
    decoy_hour = rng.randint(6, 8)
    day = rng.randint(10, 20)
    expected = f"{hour}:{minute:02d} PM"
    distractor = f"{decoy_hour}:30 AM"
    t = 1_705_500_000 + idx * 100
    return CompositionCase(
        case_id=f"before-time-{idx}",
        case_type="relative_event_time",
        question=f"What time did I go to bed the day before my {appointment}?",
        expected=expected,
        rows=[
            (f"User: On 2024-04-{day - 1:02d} I went to bed at {expected}.", t),
            (f"User: On 2024-04-{day:02d} I had a {appointment} at {distractor}.", t + 1),
            (f"User: On 2024-04-{day + 1:02d} I woke up at {decoy_hour}:00 AM.", t + 2),
        ],
        forbidden_in_proof=[f"{decoy_hour}:00 AM"],
    )


_GENERATORS: list[Callable[[random.Random, int], CompositionCase]] = [
    _shared_unwind_case,
    _shared_research_case,
    _shared_volunteer_case,
    _shared_learning_case,
    _event_order_case,
    _before_time_case,
]


def generate_cases(seed: int, cases: int) -> list[CompositionCase]:
    rng = random.Random(seed)
    out = [_GENERATORS[idx % len(_GENERATORS)](rng, idx) for idx in range(cases)]
    rng.shuffle(out)
    return out


def _load_case(store: RecordStore, scope: Scope, case: CompositionCase, *, add_claims: bool) -> None:
    for text, valid_at in case.rows:
        rec = _record(text, scope=scope, valid_at=valid_at)
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))


def _run_once(case: CompositionCase, *, backend: str) -> tuple[bool, dict, int]:
    with tempfile.TemporaryDirectory(prefix=f"smqe-composition-{backend}-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        scope = Scope(namespace=f"smqe-composition-{backend}-{case.case_id}")
        _load_case(store, scope, case, add_claims=(backend == "claim"))
        ans = structured_answer(retriever, case.question, at=1_900_000_000, verify=True, scope=scope)
        note = ans.note if ans else ""
        actual_backend = (note.split(":") + ["", "", ""])[2] if note.startswith("smqe:") else ""
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
