"""Rotating scope-isolation invariant eval for SMQE.

The same names, objects, and questions can appear in different conversations. This sidecar loads
conflicting memories into two scopes and requires the identical question to resolve only against the
requested scope, with both record-only evidence and source-backed claims present.
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
class ScopeSide:
    expected: str
    rows: list[tuple[str, float]]
    forbidden_in_proof: list[str] = field(default_factory=list)


@dataclass
class ScopeCase:
    case_id: str
    op: str
    question: str
    left: ScopeSide
    right: ScopeSide
    # P0 fail-closed (2026-07-09): a DERIVED count/sum abstains instead of shipping a verified
    # aggregate (eidetic/smqe/verify.py). Such cases assert abstention on both scope sides.
    expect_abstain: bool = False


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


def _record(text: str, *, scope: Scope, valid_at: float) -> MemoryRecord:
    digest = hashlib.sha256(f"{scope.namespace}\0{text}\0{valid_at}".encode("utf-8")).hexdigest()
    return MemoryRecord(
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        content_hash=f"h-{digest}",
        raw_uri="mem://synthetic-smqe-scope",
    )


def _latest_case(rng: random.Random, idx: int) -> ScopeCase:
    name = _pick(rng, _NAMES)
    obj = _pick(rng, _OBJECTS, f" {idx}")
    left_loc, right_loc = rng.sample(_LOCATIONS, k=2)
    t = 1_704_000_000 + idx * 100
    return ScopeCase(
        case_id=f"latest-scope-{idx}",
        op="latest_value",
        question=f"Where does {name} keep the {obj}?",
        left=ScopeSide(left_loc, [(f"{name}: I keep the {obj} at {left_loc}.", t)], [right_loc]),
        right=ScopeSide(right_loc, [(f"{name}: I keep the {obj} at {right_loc}.", t + 1)], [left_loc]),
    )


def _count_case(rng: random.Random, idx: int) -> ScopeCase:
    target = _pick(rng, _TARGETS)
    singular = target[:-1] if target.endswith("s") else target
    left_n = rng.randint(2, 4)
    right_n = rng.randint(5, 7)
    labels = rng.sample(_LOCATIONS, k=right_n)
    t = 1_704_100_000 + idx * 100
    left_rows = [
        (f"User: I visited the {labels[j]} {singular} this month.", t + j)
        for j in range(left_n)
    ]
    right_rows = [
        (f"User: I visited the {labels[j]} {singular} this month.", t + 20 + j)
        for j in range(right_n)
    ]
    return ScopeCase(
        case_id=f"count-scope-{idx}",
        op="count_aggregate",
        question=f"How many {target} did I visit this month?",
        left=ScopeSide(str(left_n), left_rows, [str(right_n)]),
        right=ScopeSide(str(right_n), right_rows, [str(left_n)]),
        expect_abstain=True,
    )


def _relative_case(rng: random.Random, idx: int) -> ScopeCase:
    item = _pick(rng, _OBJECTS, f" {idx}")
    left_ref = datetime(2024, rng.randint(2, 5), rng.randint(10, 18), 12, 0)
    right_ref = left_ref + timedelta(days=rng.randint(10, 30))
    left_expected = (left_ref - timedelta(days=1)).date().isoformat()
    right_expected = (right_ref - timedelta(days=1)).date().isoformat()
    return ScopeCase(
        case_id=f"relative-scope-{idx}",
        op="relative_temporal",
        question=f"When did I pick up the {item}?",
        left=ScopeSide(left_expected, [(f"User: Yesterday I picked up the {item}.", left_ref.timestamp())], [right_expected]),
        right=ScopeSide(right_expected, [(f"User: Yesterday I picked up the {item}.", right_ref.timestamp())], [left_expected]),
    )


def _table_case(rng: random.Random, idx: int) -> ScopeCase:
    person = _pick(rng, _NAMES)
    day = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"])
    left_value, right_value = rng.sample(_TABLE_VALUES, k=2)
    t = 1_704_200_000 + idx * 100
    return ScopeCase(
        case_id=f"table-scope-{idx}",
        op="table_lookup",
        question=f"What shift does {person} have on {day} in the schedule?",
        left=ScopeSide(
            left_value,
            [(f"| Name | {day} |\n| {person} | {left_value} |", t)],
            [right_value],
        ),
        right=ScopeSide(
            right_value,
            [(f"| Name | {day} |\n| {person} | {right_value} |", t + 1)],
            [left_value],
        ),
    )


def _preference_case(rng: random.Random, idx: int) -> ScopeCase:
    left_good, right_good = rng.sample(_PREFS, k=2)
    left = f"{left_good} {idx}"
    right = f"{right_good} {idx}"
    t = 1_704_300_000 + idx * 100
    return ScopeCase(
        case_id=f"preference-scope-{idx}",
        op="preference_synth",
        question=f"Would I prefer {left} or {right}?",
        left=ScopeSide(
            left,
            [(f"User: I enjoy {left} after work.", t), (f"User: I avoid {right} before meetings.", t + 1)],
            [right],
        ),
        right=ScopeSide(
            right,
            [(f"User: I avoid {left} before meetings.", t + 2), (f"User: I enjoy {right} after work.", t + 3)],
            [left],
        ),
    )


def _speaker_case(rng: random.Random, idx: int) -> ScopeCase:
    speaker = _pick(rng, _NAMES)
    topic = _pick(rng, _OBJECTS, f" {idx}")
    left_loc, right_loc = [value.lower() for value in rng.sample(_LOCATIONS, k=2)]
    t = 1_704_400_000 + idx * 100
    left_expected = f"the {topic} stays in the {left_loc}"
    right_expected = f"the {topic} stays in the {right_loc}"
    return ScopeCase(
        case_id=f"speaker-scope-{idx}",
        op="speaker_fact",
        question=f"What did {speaker} say about the {topic}?",
        left=ScopeSide(left_expected, [(f"{speaker}: I said the {topic} stays in the {left_loc}.", t)], [right_loc]),
        right=ScopeSide(right_expected, [(f"{speaker}: I said the {topic} stays in the {right_loc}.", t + 1)], [left_loc]),
    )


def _temporal_delta_case(rng: random.Random, idx: int) -> ScopeCase:
    start_item = _pick(rng, _OBJECTS, f" {idx}")
    finish_item = _pick(rng, [x for x in _OBJECTS if x not in start_item], f" {idx}")
    left_days = rng.randint(3, 7)
    right_days = rng.randint(8, 13)
    start = datetime(2024, rng.randint(1, 8), rng.randint(2, 10), 12, 0)
    t0 = start.timestamp()
    return ScopeCase(
        case_id=f"delta-scope-{idx}",
        op="temporal_delta",
        question=(
            f"How many days passed between the day I started calibrating the {start_item} "
            f"and the day I finished installing the {finish_item}?"
        ),
        left=ScopeSide(
            f"{left_days} days",
            [
                (f"User: I started calibrating the {start_item} today.", t0),
                (f"User: I finished installing the {finish_item} today.", (start + timedelta(days=left_days)).timestamp()),
            ],
            [f"{right_days} days"],
        ),
        right=ScopeSide(
            f"{right_days} days",
            [
                (f"User: I started calibrating the {start_item} today.", t0 + 100),
                (f"User: I finished installing the {finish_item} today.", (start + timedelta(days=right_days)).timestamp() + 100),
            ],
            [f"{left_days} days"],
        ),
    )


def _sum_case(rng: random.Random, idx: int) -> ScopeCase:
    project = _pick(rng, _PROJECTS, f" {idx}")
    left_a, left_b = rng.randint(1, 3), rng.randint(2, 4)
    right_a, right_b = rng.randint(4, 6), rng.randint(3, 5)
    left_total = left_a + left_b
    right_total = right_a + right_b
    t = 1_704_500_000 + idx * 100
    return ScopeCase(
        case_id=f"sum-scope-{idx}",
        op="multi_session_sum",
        question=f"How many total hours did I spend on the {project}?",
        left=ScopeSide(
            f"{left_total} hours",
            [(f"User: I spent {left_a} hours on the {project}.", t), (f"User: I spent {left_b} hours on the {project}.", t + 1)],
        ),
        right=ScopeSide(
            f"{right_total} hours",
            [(f"User: I spent {right_a} hours on the {project}.", t + 2), (f"User: I spent {right_b} hours on the {project}.", t + 3)],
        ),
        expect_abstain=True,
    )


_GENERATORS: list[Callable[[random.Random, int], ScopeCase]] = [
    _latest_case,
    _count_case,
    _relative_case,
    _table_case,
    _preference_case,
    _speaker_case,
    _temporal_delta_case,
    _sum_case,
]


def generate_cases(seed: int, cases: int) -> list[ScopeCase]:
    rng = random.Random(seed)
    out = [_GENERATORS[idx % len(_GENERATORS)](rng, idx) for idx in range(cases)]
    rng.shuffle(out)
    return out


def _load_side(store: RecordStore, scope: Scope, side: ScopeSide, *, add_claims: bool) -> None:
    for text, valid_at in side.rows:
        rec = _record(text, scope=scope, valid_at=valid_at)
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))


def _answer_ok(answer: str, expected: str) -> bool:
    return (expected or "").lower() in (answer or "").lower()


def _run_case(case: ScopeCase, *, claims_present: bool) -> tuple[int, int, list[dict], int]:
    with tempfile.TemporaryDirectory(prefix="smqe-scope-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        left_scope = Scope(namespace=f"smqe-scope-left-{case.case_id}")
        right_scope = Scope(namespace=f"smqe-scope-right-{case.case_id}")
        _load_side(store, left_scope, case.left, add_claims=claims_present)
        _load_side(store, right_scope, case.right, add_claims=claims_present)
        correct = 0
        proof_tokens = 0
        failures: list[dict] = []
        for side_name, scope, side in (("left", left_scope, case.left), ("right", right_scope, case.right)):
            ans = structured_answer(retriever, case.question, at=1_900_000_000, verify=True, scope=scope)
            proof = " ".join(c.snippet for c in (ans.citations if ans else []))
            proof_tokens += sum(max(0, len(c.snippet or "") // 4) for c in (ans.citations if ans else []))
            if case.expect_abstain:
                ok = ans is None  # derived aggregate fails closed on both scope sides
            else:
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
