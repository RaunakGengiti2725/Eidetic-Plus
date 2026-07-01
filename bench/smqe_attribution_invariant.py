"""Rotating actor-attribution invariant for SMQE.

This sidecar checks "who recommended/gave/told/shared X" style questions on fresh
synthetic memories. The answer must be the actor attached to the positive evidence,
with verified proof, and negated same-target distractors must not be cited.
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
class AttributionCase:
    case_id: str
    case_type: str
    question: str
    expected: str
    rows: list[str]
    forbidden_in_proof: list[str] = field(default_factory=list)


_PEOPLE = [
    "Ari", "Mira", "Nolan", "Tessa", "Omar", "Nila", "Rhea", "Soren",
    "Vera", "Theo", "Iris", "Jonah",
]
_PLACES = [
    "Cedar Cafe", "Harbor Books", "Juniper Studio", "Linen Market",
    "Orchid Bakery", "Quartz Gallery", "River Supply", "Willow Records",
]
_ITEMS = [
    "brass compass", "copper notebook", "linen sketchbook", "orchid journal",
    "violet pencil case", "cedar travel mug", "harbor map", "quartz field guide",
]


def _record(text: str, *, scope: Scope, valid_at: float) -> MemoryRecord:
    digest = hashlib.sha256(f"{scope.namespace}\0{text}\0{valid_at}".encode("utf-8")).hexdigest()
    return MemoryRecord(
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        content_hash=f"h-{digest}",
        raw_uri="mem://synthetic-smqe-attribution",
    )


def _people(rng: random.Random) -> tuple[str, str]:
    left, right = rng.sample(_PEOPLE, 2)
    return left, right


def _recommend_case(rng: random.Random, idx: int) -> AttributionCase:
    actor, distractor = _people(rng)
    place = rng.choice(_PLACES)
    return AttributionCase(
        case_id=f"recommend-{idx}",
        case_type="recommend_actor",
        question=f"Who recommended {place}?",
        expected=actor,
        rows=[
            f"{actor}: I recommend {place} for brunch.",
            f"{distractor}: {place} is near the station, but I did not recommend it.",
        ],
        forbidden_in_proof=[distractor],
    )


def _gave_case(rng: random.Random, idx: int) -> AttributionCase:
    actor, distractor = _people(rng)
    item = rng.choice(_ITEMS)
    return AttributionCase(
        case_id=f"gave-{idx}",
        case_type="gave_actor",
        question=f"Who gave me the {item}?",
        expected=actor,
        rows=[
            f"{actor}: I gave you the {item} after the hike.",
            f"{distractor}: I borrowed the {item}, but I did not give it to you.",
        ],
        forbidden_in_proof=[distractor],
    )


def _told_case(rng: random.Random, idx: int) -> AttributionCase:
    actor, distractor = _people(rng)
    place = rng.choice(_PLACES)
    return AttributionCase(
        case_id=f"told-{idx}",
        case_type="told_actor",
        question=f"Who told me about {place}?",
        expected=actor,
        rows=[
            f"{actor}: I told you about {place} yesterday.",
            f"{distractor}: {place} has a sale, but I never told you about it.",
        ],
        forbidden_in_proof=[distractor],
    )


def _shared_case(rng: random.Random, idx: int) -> AttributionCase:
    actor, distractor = _people(rng)
    item = rng.choice(_ITEMS)
    return AttributionCase(
        case_id=f"shared-{idx}",
        case_type="shared_actor",
        question=f"Who shared the {item} with me?",
        expected=actor,
        rows=[
            f"{actor}: I shared the {item} with you before the workshop.",
            f"{distractor}: I saw the {item}, but I did not share it with you.",
        ],
        forbidden_in_proof=[distractor],
    )


_GENERATORS: list[Callable[[random.Random, int], AttributionCase]] = [
    _recommend_case,
    _gave_case,
    _told_case,
    _shared_case,
]


def generate_cases(seed: int, cases: int) -> list[AttributionCase]:
    rng = random.Random(seed)
    out = [_GENERATORS[idx % len(_GENERATORS)](rng, idx) for idx in range(cases)]
    rng.shuffle(out)
    return out


def _load_case(store: RecordStore, scope: Scope, case: AttributionCase, *, add_claims: bool) -> None:
    for idx, text in enumerate(case.rows):
        rec = _record(text, scope=scope, valid_at=1_700_000_000 + idx)
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))


def _run_once(case: AttributionCase, *, backend: str) -> tuple[bool, dict, int]:
    with tempfile.TemporaryDirectory(prefix=f"smqe-attribution-{backend}-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        scope = Scope(namespace=f"smqe-attribution-{backend}-{case.case_id}")
        _load_case(store, scope, case, add_claims=(backend == "claim"))
        ans = structured_answer(retriever, case.question, at=1_800_000_000, verify=True, scope=scope)
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
