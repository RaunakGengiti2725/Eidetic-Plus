"""Rotating temporal-conflict invariant for SMQE.

This sidecar checks that latest/current questions survive changed memories. Each case has an old
value, an active newer value, and a future not-yet-active value. A case only passes when both the
record backend and the claim backend return the active latest value with verified proof that excludes
the stale and future values.
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
class ConflictCase:
    case_id: str
    value_type: str
    question: str
    expected: str
    rows: list[tuple[str, float]]
    forbidden_in_proof: list[str] = field(default_factory=list)


_NAMES = ["Ari", "Nila", "Mika", "Sana", "Theo", "Rowan", "Lina", "Owen"]
_OBJECTS = ["backup badge", "kiln token", "garden permit", "field notebook", "studio key", "travel charger"]
_LOCATIONS = ["Quartz Loft", "North Pier Studio", "Cedar Annex", "Blue Finch Lab", "Orchid Room", "Harbor Desk"]
_STATUSES = ["pending", "approved", "paused", "ready", "archived", "active"]
_BUDGETS = ["repair budget", "travel budget", "studio fund", "permit budget", "equipment budget"]


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
        raw_uri="mem://synthetic-smqe-conflict",
    )


def _location_case(rng: random.Random, idx: int) -> ConflictCase:
    name = _pick(rng, _NAMES)
    obj = _pick(rng, _OBJECTS, f" {idx}")
    old, new, future = rng.sample(_LOCATIONS, k=3)
    t = 1_702_000_000 + idx * 100
    return ConflictCase(
        case_id=f"location-conflict-{idx}",
        value_type="location",
        question=f"Where does {name} keep the {obj} now?",
        expected=new,
        rows=[
            (f"{name}: I keep the {obj} at {old}.", t),
            (f"{name}: I moved the {obj} to {new}.", t + 10),
            (f"{name}: I will move the {obj} to {future}.", t + 10_000_000),
        ],
        forbidden_in_proof=[old, future],
    )


def _status_case(rng: random.Random, idx: int) -> ConflictCase:
    item = _pick(rng, _OBJECTS, f" {idx}")
    old, new, future = rng.sample(_STATUSES, k=3)
    t = 1_702_100_000 + idx * 100
    return ConflictCase(
        case_id=f"status-conflict-{idx}",
        value_type="status",
        question=f"What is my current {item} status?",
        expected=new,
        rows=[
            (f"User: My {item} status is {old}.", t),
            (f"User: My {item} status is {new}.", t + 10),
            (f"User: My {item} status will be {future}.", t + 10_000_000),
        ],
        forbidden_in_proof=[old, future],
    )


def _amount_case(rng: random.Random, idx: int) -> ConflictCase:
    budget = _pick(rng, _BUDGETS, f" {idx}")
    old = rng.randrange(80, 180, 5)
    new = rng.randrange(220, 420, 5)
    future = rng.randrange(500, 800, 5)
    t = 1_702_200_000 + idx * 100
    currency_word = rng.choice(["dollars", "usd", "bucks"])
    use_words = rng.choice([False, True])

    def fmt(value: int) -> str:
        return f"{value} {currency_word}" if use_words else f"${value}"

    return ConflictCase(
        case_id=f"amount-conflict-{idx}",
        value_type="amount",
        question=f"What is my current {budget}?",
        expected=fmt(new),
        rows=[
            (f"User: My current {budget} is {fmt(old)}.", t),
            (f"User: My current {budget} is {fmt(new)}.", t + 10),
            (f"User: My current {budget} will be {fmt(future)}.", t + 10_000_000),
        ],
        forbidden_in_proof=[fmt(old), fmt(future)],
    )


_GENERATORS: list[Callable[[random.Random, int], ConflictCase]] = [
    _location_case,
    _status_case,
    _amount_case,
]


def generate_cases(seed: int, cases: int) -> list[ConflictCase]:
    rng = random.Random(seed)
    out = [_GENERATORS[idx % len(_GENERATORS)](rng, idx) for idx in range(cases)]
    rng.shuffle(out)
    return out


def _load_case(store: RecordStore, scope: Scope, case: ConflictCase, *, add_claims: bool) -> None:
    for text, valid_at in case.rows:
        rec = _record(text, scope=scope, valid_at=valid_at)
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))


def _run_once(case: ConflictCase, *, backend: str) -> tuple[bool, dict, int]:
    with tempfile.TemporaryDirectory(prefix=f"smqe-conflict-{backend}-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        scope = Scope(namespace=f"smqe-conflict-{backend}-{case.case_id}")
        _load_case(store, scope, case, add_claims=(backend == "claim"))
        ans = structured_answer(retriever, case.question, at=1_800_000_000, verify=True, scope=scope)
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
    value_type_counts: dict[str, int] = {}
    backend_counts = {"claim": 0, "record": 0}
    proof_tokens = 0
    record_correct = 0
    claim_correct = 0
    for case in generated:
        value_type_counts[case.value_type] = value_type_counts.get(case.value_type, 0) + 1
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
                "value_type": case.value_type,
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
        "value_type_counts": dict(sorted(value_type_counts.items())),
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
