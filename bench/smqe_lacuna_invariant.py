"""Rotating lacuna/antimemory eval for SMQE proposition confirmation.

Yes/no recall has four honest outcomes: a stated proposition answers "Yes - <premise>"; a stored
NEGATED assertion (antimemory) answers "No - <premise>"; when both polarities were asserted the
LATEST assertion wins (retraction / re-assertion); and an absent proposition must never produce a
structured yes/no at all (closed-world judgment stays with the reader). This sidecar proves all
four contracts with invented conversations and rotating entities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bench.seed_policy import resolve_seed
from bench.smqe_synthetic_invariant import _Retriever
from eidetic.models import MemoryRecord, Scope
from eidetic.smqe import structured_answer
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.store import RecordStore


_NAMES = ["Ari", "Nila", "Mika", "Sana", "Theo", "Rowan", "Lina", "Owen", "Tessa", "Ira"]
_ACTIVITIES = [
    ("weekly {obj} lessons", "taking", ["signed up for weekly {obj} lessons at the studio",
                                        "restarted my weekly {obj} lessons at the studio"],
     "dropped the {obj} lessons, not doing them anymore"),
]
_OBJECTS = ["pottery", "violin", "archery", "calligraphy", "fencing", "weaving"]
_PLACES = ["jazz festival", "night market", "sculpture garden", "book fair", "planetarium",
           "botanical garden"]
_TOOLS = ["meal planning app", "habit tracker", "carpool roster", "seed catalog",
          "route planner", "shift calendar"]


@dataclass
class LacunaCase:
    case_id: str
    case_type: str
    question: str
    rows: list[tuple[str, float]]
    expect_prefix: str = ""            # "yes" / "no" / "" (no structured yes/no expected)
    expect_in_proof: str = ""
    forbid_in_proof: str = ""
    expect_answer: bool = True
    notes: dict = field(default_factory=dict)


def _record(text: str, *, scope: Scope, valid_at: float) -> MemoryRecord:
    digest = hashlib.sha256(f"{scope.namespace}\0{text}\0{valid_at}".encode("utf-8")).hexdigest()
    return MemoryRecord(
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        content_hash=f"h-{digest}",
        raw_uri="mem://synthetic-smqe-lacuna",
    )


def _positive_confirmation(rng: random.Random, idx: int) -> LacunaCase:
    name = rng.choice(_NAMES)
    tool = rng.choice(_TOOLS)
    t = 1_706_000_000 + idx * 1_000
    convo = (
        f"{name}: By the way, my aunt is actually using the same {tool} as me now, "
        f"so we can share notes."
    )
    return LacunaCase(
        case_id=f"lacuna-positive-{idx}",
        case_type="positive_confirmation",
        question=f"Is {name}'s aunt using the same {tool} as {name}?",
        rows=[(convo, t)],
        expect_prefix="yes",
        expect_in_proof=tool,
    )


def _negative_assertion(rng: random.Random, idx: int) -> LacunaCase:
    name = rng.choice(_NAMES)
    place = rng.choice(_PLACES)
    t = 1_706_500_000 + idx * 1_000
    convo = f"{name}: I've never been to a {place}, honestly."
    return LacunaCase(
        case_id=f"lacuna-negative-{idx}",
        case_type="negative_assertion",
        question=f"Has {name} ever been to a {place}?",
        rows=[(convo, t)],
        expect_prefix="no",
        expect_in_proof=place,
    )


def _retraction_order(rng: random.Random, idx: int) -> LacunaCase:
    name = rng.choice(_NAMES)
    obj = rng.choice(_OBJECTS)
    t = 1_707_000_000 + idx * 1_000
    positive = f"{name}: I signed up for weekly {obj} lessons at the studio."
    negative = f"{name}: I dropped the {obj} lessons, not doing them anymore."
    restart = f"{name}: Good news - I restarted my weekly {obj} lessons at the studio."
    if idx % 2 == 0:
        rows = [(positive, t), (negative, t + 5_000)]
        expect_prefix = "no"
        expect_in_proof = "dropped"
        forbid_in_proof = "signed up"
    else:
        rows = [(negative, t), (restart, t + 5_000)]
        expect_prefix = "yes"
        expect_in_proof = "restarted"
        forbid_in_proof = "dropped"
    return LacunaCase(
        case_id=f"lacuna-retraction-{idx}",
        case_type="retraction_order",
        question=f"Is {name} taking {obj} lessons?",
        rows=rows,
        expect_prefix=expect_prefix,
        expect_in_proof=expect_in_proof,
        forbid_in_proof=forbid_in_proof,
    )


def _absent_proposition(rng: random.Random, idx: int) -> LacunaCase:
    name = rng.choice(_NAMES)
    place = rng.choice(_PLACES)
    other_place = rng.choice([p for p in _PLACES if p != place])
    t = 1_707_500_000 + idx * 1_000
    # Memory covers the person and unrelated outings, but never asserts the proposition in
    # either polarity: the executor must not fabricate a closed-world yes/no.
    convo = (
        f"{name}: The {other_place} trip last month was lovely.\n"
        f"{name}: I've been busy with work most weekends."
    )
    return LacunaCase(
        case_id=f"lacuna-absent-{idx}",
        case_type="absent_proposition",
        question=f"Has {name} ever been to a {place}?",
        rows=[(convo, t)],
        expect_answer=False,
        forbid_in_proof=other_place,
    )


_BUILDERS = (_positive_confirmation, _negative_assertion, _retraction_order, _absent_proposition)


def generate_cases(seed: int, cases: int) -> list[LacunaCase]:
    rng = random.Random(seed)
    out: list[LacunaCase] = []
    for idx in range(cases):
        out.append(_BUILDERS[idx % len(_BUILDERS)](rng, idx))
    rng.shuffle(out)
    return out


def _run_case(case: LacunaCase, tmp: Path) -> tuple[bool, dict]:
    store = RecordStore(tmp / f"{case.case_id}.sqlite")
    scope = Scope(namespace=f"smqe-lacuna-{case.case_id}")
    for text, valid_at in case.rows:
        rec = _record(text, scope=scope, valid_at=valid_at)
        store.upsert_record(rec)
        store.add_claims(claims_for_record(rec))
    ans = structured_answer(_Retriever(store), case.question, at=1_800_000_000, scope=scope)
    detail: dict = {
        "answer": None if ans is None else ans.answer,
        "verified": None if ans is None else ans.verified,
        "note": None if ans is None else ans.note,
    }
    proof = "" if ans is None else " ".join(c.snippet for c in ans.citations)
    answer_low = "" if ans is None else (ans.answer or "").lower()
    if case.expect_answer:
        if ans is None:
            return False, {**detail, "why": "expected a structured answer"}
        if case.expect_prefix and not answer_low.startswith(case.expect_prefix):
            return False, {**detail, "why": f"answer must start with {case.expect_prefix!r}"}
        if not ans.verified:
            return False, {**detail, "why": "answer must be verified"}
        if case.expect_in_proof and case.expect_in_proof.lower() not in proof.lower():
            return False, {**detail, "why": f"proof must contain {case.expect_in_proof!r}"}
    else:
        # A structured non-polarity answer (e.g. a slot value) is tolerated; fabricating a
        # closed-world yes/no from absence is the failure.
        if answer_low.startswith("yes") or answer_low.startswith("no"):
            return False, {**detail, "why": "absent proposition must not produce a yes/no"}
    if case.forbid_in_proof and case.forbid_in_proof.lower() in proof.lower():
        return False, {**detail, "why": f"forbidden content in proof: {case.forbid_in_proof!r}"}
    return True, detail


def run_eval(*, seed: Optional[int] = None, cases: int = 24) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    failures = []
    type_counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="smqe-lacuna-") as tmp_str:
        tmp = Path(tmp_str)
        for case in generated:
            type_counts[case.case_type] = type_counts.get(case.case_type, 0) + 1
            ok, detail = _run_case(case, tmp)
            if not ok:
                failures.append({
                    "case_id": case.case_id,
                    "case_type": case.case_type,
                    "question": case.question,
                    **detail,
                })
    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        "correct": cases - len(failures),
        "failures": failures,
        "case_type_counts": dict(sorted(type_counts.items())),
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
