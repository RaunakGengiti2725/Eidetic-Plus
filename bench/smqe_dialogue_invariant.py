"""Rotating dialogue Q->A crystal eval for SMQE.

Conversations answer questions in-line; the consolidation layer crystallizes the answering
sentence with the RECORDED question as claim filters. This sidecar proves the paraphrase-stable
contract with invented conversations: an equivalent later query answers from the crystal with a
verified citation; entity attribution never bleeds across speakers; advice requests are NOT
served by replaying one past reply; and an incidental one-word overlap with an unrelated recorded
question cannot bridge an answer.
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
_SEASONS = ["spring", "summer", "autumn", "winter"]
_PROJECTS = [
    "garden plots", "adoption agencies", "ceramic kilns", "trail maps", "violin lessons",
    "solar panels", "language classes", "archive scans",
]
_TIDY_TOPICS = ["kitchen counters", "garage shelves", "desk drawers", "studio benches"]
_TIDY_ITEMS = ["utensil holder", "label maker", "pegboard rack", "drawer divider"]


@dataclass
class DialogueCase:
    case_id: str
    case_type: str
    question: str
    rows: list[tuple[str, float]]
    expect_contains: str = ""
    forbid_in_answer: str = ""
    forbid_in_proof: str = ""
    expect_answer: bool = True
    require_verified: bool = True
    notes: dict = field(default_factory=dict)


def _record(text: str, *, scope: Scope, valid_at: float) -> MemoryRecord:
    digest = hashlib.sha256(f"{scope.namespace}\0{text}\0{valid_at}".encode("utf-8")).hexdigest()
    return MemoryRecord(
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        content_hash=f"h-{digest}",
        raw_uri="mem://synthetic-smqe-dialogue",
    )


def _paraphrase_slot(rng: random.Random, idx: int) -> DialogueCase:
    name = rng.choice(_NAMES)
    other = rng.choice([n for n in _NAMES if n != name])
    season = rng.choice(_SEASONS)
    project = rng.choice(_PROJECTS) + f" {idx}"
    t = 1_704_000_000 + idx * 1_000
    convo = (
        f"{other}: Any fun plans for the {season}?\n"
        f"{name}: Researching {project} - it's been a dream of mine for years."
    )
    return DialogueCase(
        case_id=f"dialogue-paraphrase-{idx}",
        case_type="paraphrase_slot",
        question=f"What are {name}'s plans for the {season}?",
        rows=[(convo, t)],
        expect_contains=f"Researching {project}",
    )


def _entity_guard(rng: random.Random, idx: int) -> DialogueCase:
    name = rng.choice(_NAMES)
    other = rng.choice([n for n in _NAMES if n != name])
    asker = rng.choice([n for n in _NAMES if n not in (name, other)])
    season = rng.choice(_SEASONS)
    project = rng.choice(_PROJECTS) + f" {idx}"
    decoy = rng.choice([p for p in _PROJECTS if p not in project]) + f" {idx}"
    t = 1_704_500_000 + idx * 1_000
    row_a = (
        f"{asker}: Any fun plans for the {season}?\n"
        f"{name}: Researching {project} - I've waited years for this."
    )
    row_b = (
        f"{asker}: Any fun plans for the {season}?\n"
        f"{other}: Researching {decoy} - finally taking the leap."
    )
    return DialogueCase(
        case_id=f"dialogue-entity-{idx}",
        case_type="entity_guard",
        question=f"What are {name}'s plans for the {season}?",
        rows=[(row_a, t), (row_b, t + 10)],
        expect_contains=f"Researching {project}",
        forbid_in_answer=decoy,
        forbid_in_proof=decoy,
    )


def _advice_deferral(rng: random.Random, idx: int) -> DialogueCase:
    name = rng.choice(_NAMES)
    other = rng.choice([n for n in _NAMES if n != name])
    topic = rng.choice(_TIDY_TOPICS)
    item = rng.choice(_TIDY_ITEMS) + f" {idx}"
    t = 1_705_000_000 + idx * 1_000
    convo = (
        f"{other}: Any tips for keeping the {topic} tidy?\n"
        f"{name}: I bought a new {item} and it keeps everything in place."
    )
    return DialogueCase(
        case_id=f"dialogue-advice-{idx}",
        case_type="advice_deferral",
        question=f"Any tips for keeping my {topic} tidy?",
        rows=[(convo, t)],
        forbid_in_answer=item,
        expect_answer=False,
        require_verified=False,
    )


def _unrelated_guard(rng: random.Random, idx: int) -> DialogueCase:
    name = rng.choice(_NAMES)
    other = rng.choice([n for n in _NAMES if n != name])
    season = rng.choice(_SEASONS)
    project = rng.choice(_PROJECTS) + f" {idx}"
    t = 1_705_500_000 + idx * 1_000
    # The recorded question shares exactly one incidental content word ("plans") with the query,
    # but asks about a different subject matter entirely.
    convo = (
        f"{other}: Do the blueprint plans include a service elevator?\n"
        f"{name}: Researching {project} - the annex drawings are done."
    )
    return DialogueCase(
        case_id=f"dialogue-unrelated-{idx}",
        case_type="unrelated_guard",
        question=f"What are {name}'s plans for the {season}?",
        rows=[(convo, t)],
        forbid_in_proof="blueprint",
        expect_answer=False,
        require_verified=False,
        notes={"allow_answer": True},
    )


_BUILDERS = (_paraphrase_slot, _entity_guard, _advice_deferral, _unrelated_guard)


def generate_cases(seed: int, cases: int) -> list[DialogueCase]:
    rng = random.Random(seed)
    out: list[DialogueCase] = []
    for idx in range(cases):
        out.append(_BUILDERS[idx % len(_BUILDERS)](rng, idx))
    rng.shuffle(out)
    return out


def _run_case(case: DialogueCase, tmp: Path) -> tuple[bool, dict]:
    store = RecordStore(tmp / f"{case.case_id}.sqlite")
    scope = Scope(namespace=f"smqe-dialogue-{case.case_id}")
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
    if case.expect_answer:
        if ans is None:
            return False, {**detail, "why": "expected an answer"}
        if case.expect_contains and case.expect_contains.lower() not in ans.answer.lower():
            return False, {**detail, "why": f"answer must contain {case.expect_contains!r}"}
        if case.require_verified and not ans.verified:
            return False, {**detail, "why": "answer must be verified"}
    else:
        allow_answer = bool(case.notes.get("allow_answer"))
        if ans is not None and not allow_answer and case.forbid_in_answer and \
                case.forbid_in_answer.lower() in (ans.answer or "").lower():
            return False, {**detail, "why": f"answer must not replay {case.forbid_in_answer!r}"}
    if case.forbid_in_answer and ans is not None and \
            case.forbid_in_answer.lower() in (ans.answer or "").lower():
        return False, {**detail, "why": f"forbidden content in answer: {case.forbid_in_answer!r}"}
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
    with tempfile.TemporaryDirectory(prefix="smqe-dialogue-") as tmp_str:
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
