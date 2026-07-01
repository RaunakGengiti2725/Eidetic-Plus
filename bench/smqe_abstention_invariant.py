"""Rotating unsupported-question abstention eval for SMQE.

Positive recall is not enough: a memory agent must also refuse unsupported structured questions. This
sidecar invents decoy memories that are lexically close to the question but do not entail the answer.
Each case must abstain both with record-only evidence and with source-backed claims present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from bench.seed_policy import resolve_seed
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from bench.smqe_synthetic_invariant import _Retriever
from eidetic.models import MemoryRecord, Scope
from eidetic.smqe import structured_answer
from eidetic.smqe.claim_extraction import claims_for_record
from eidetic.store import RecordStore


@dataclass
class AbstentionCase:
    case_id: str
    case_type: str
    question: str
    rows: list[tuple[str, float]]


_NAMES = ["Ari", "Nila", "Mika", "Sana", "Theo", "Rowan", "Lina", "Owen"]
_OBJECTS = ["backup badge", "kiln token", "garden permit", "field notebook", "studio key", "travel charger"]
_LOCATIONS = ["Quartz Loft", "North Pier Studio", "Cedar Annex", "Blue Finch Lab", "Orchid Room"]
_TARGETS = ["ceramic studios", "tea shops", "library workshops", "bike routes", "garden plots"]
_DECOYS = ["museum exhibits", "recipe cards", "train stops", "repair notes", "hiking trails"]
_GOOD = ["mint tea", "fantasy novels", "graphite pens", "berry salad", "quiet playlists"]
_BAD = ["cedar tea", "tax manuals", "brass pens", "kale salad", "alarm playlists"]


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
        raw_uri="mem://synthetic-smqe-abstention",
    )


def _latest_missing_subject(rng: random.Random, idx: int) -> AbstentionCase:
    wanted = _pick(rng, _NAMES)
    other = _pick(rng, [x for x in _NAMES if x != wanted])
    obj = _pick(rng, _OBJECTS, f" {idx}")
    loc = _pick(rng, _LOCATIONS)
    t = 1_703_000_000 + idx * 100
    return AbstentionCase(
        case_id=f"latest-missing-subject-{idx}",
        case_type="latest_missing_subject",
        question=f"Where does {wanted} keep the {obj}?",
        rows=[(f"{other}: I keep the {obj} at {loc}.", t)],
    )


def _latest_future_only(rng: random.Random, idx: int) -> AbstentionCase:
    name = _pick(rng, _NAMES)
    obj = _pick(rng, _OBJECTS, f" {idx}")
    loc = _pick(rng, _LOCATIONS)
    t = 1_703_100_000 + idx * 100
    return AbstentionCase(
        case_id=f"latest-future-only-{idx}",
        case_type="latest_future_only",
        question=f"Where does {name} keep the {obj} now?",
        rows=[(f"{name}: I will move the {obj} to {loc}.", t)],
    )


def _count_target_mismatch(rng: random.Random, idx: int) -> AbstentionCase:
    target = _pick(rng, _TARGETS)
    decoy = _pick(rng, _DECOYS, f" {idx}")
    n = rng.randint(3, 9)
    t = 1_703_200_000 + idx * 100
    return AbstentionCase(
        case_id=f"count-target-mismatch-{idx}",
        case_type="count_target_mismatch",
        question=f"How many {target} did I sample this month?",
        rows=[(f"User: I saved a list of {target}. I sampled {n} new {decoy} this month.", t)],
    )


def _count_neutral_quantity(rng: random.Random, idx: int) -> AbstentionCase:
    target = _pick(rng, _TARGETS)
    n = rng.randint(3, 9)
    t = 1_703_300_000 + idx * 100
    return AbstentionCase(
        case_id=f"count-neutral-quantity-{idx}",
        case_type="count_neutral_quantity",
        question=f"How many {target} did I visit this month?",
        rows=[(f"User: The directory lists {n} different {target}, but I have not visited them.", t)],
    )


def _table_missing_row(rng: random.Random, idx: int) -> AbstentionCase:
    person = _pick(rng, _NAMES)
    other = _pick(rng, [x for x in _NAMES if x != person])
    day = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"])
    t = 1_703_400_000 + idx * 100
    return AbstentionCase(
        case_id=f"table-missing-row-{idx}",
        case_type="table_missing_row",
        question=f"What shift does {person} have on {day} in the schedule?",
        rows=[(f"| Name | {day} |\n| {other} | 7 AM |", t)],
    )


def _preference_no_positive(rng: random.Random, idx: int) -> AbstentionCase:
    neutral = _pick(rng, _GOOD, f" {idx}")
    negative = _pick(rng, _BAD, f" {idx}")
    t = 1_703_500_000 + idx * 100
    return AbstentionCase(
        case_id=f"preference-no-positive-{idx}",
        case_type="preference_no_positive",
        question=f"Would I prefer {neutral} or {negative}?",
        rows=[
            (f"User: The archive mentions {neutral} as a label.", t),
            (f"User: I avoid {negative} before meetings.", t + 1),
        ],
    )


def _temporal_missing_anchor(rng: random.Random, idx: int) -> AbstentionCase:
    start_item = _pick(rng, _OBJECTS, f" {idx}")
    finish_item = _pick(rng, [x for x in _OBJECTS if x not in start_item], f" {idx}")
    start = datetime(2024, rng.randint(1, 9), rng.randint(2, 12), 12, 0)
    return AbstentionCase(
        case_id=f"temporal-missing-anchor-{idx}",
        case_type="temporal_missing_anchor",
        question=f"How many days passed between opening the {start_item} and closing the {finish_item}?",
        rows=[(f"User: I opened the {start_item} today.", start.timestamp())],
    )


def _speaker_crossed_support(rng: random.Random, idx: int) -> AbstentionCase:
    speaker = _pick(rng, _NAMES)
    other = _pick(rng, [x for x in _NAMES if x != speaker])
    topic = _pick(rng, _OBJECTS, f" {idx}")
    other_topic = _pick(rng, [x for x in _OBJECTS if x not in topic], f" {idx}")
    loc = _pick(rng, _LOCATIONS).lower()
    other_loc = _pick(rng, [x for x in _LOCATIONS if x.lower() != loc]).lower()
    t = 1_703_600_000 + idx * 100
    return AbstentionCase(
        case_id=f"speaker-crossed-support-{idx}",
        case_type="speaker_crossed_support",
        question=f"What did {speaker} say about the {topic}?",
        rows=[
            (f"{speaker}: I said the {other_topic} stays in the {loc}.", t),
            (f"{other}: I said the {topic} stays in the {other_loc}.", t + 1),
        ],
    )


_GENERATORS: list[Callable[[random.Random, int], AbstentionCase]] = [
    _latest_missing_subject,
    _latest_future_only,
    _count_target_mismatch,
    _count_neutral_quantity,
    _table_missing_row,
    _preference_no_positive,
    _temporal_missing_anchor,
    _speaker_crossed_support,
]


def generate_cases(seed: int, cases: int) -> list[AbstentionCase]:
    rng = random.Random(seed)
    out = [_GENERATORS[idx % len(_GENERATORS)](rng, idx) for idx in range(cases)]
    rng.shuffle(out)
    return out


def _load_case(store: RecordStore, scope: Scope, case: AbstentionCase, *, add_claims: bool) -> None:
    for text, valid_at in case.rows:
        rec = _record(text, scope=scope, valid_at=valid_at)
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))


def _run_once(case: AbstentionCase, *, claims_present: bool) -> tuple[bool, dict]:
    with tempfile.TemporaryDirectory(prefix="smqe-abstain-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        mode = "claims" if claims_present else "record_only"
        scope = Scope(namespace=f"smqe-abstain-{mode}-{case.case_id}")
        _load_case(store, scope, case, add_claims=claims_present)
        ans = structured_answer(retriever, case.question, at=1_800_000_000, verify=True, scope=scope)
        ok = ans is None
        return ok, {
            "actual": ans.answer if ans else "",
            "note": ans.note if ans else "",
            "verified": bool(ans and ans.verified),
            "proof": " ".join(c.snippet for c in (ans.citations if ans else []))[:500],
        }


def run_eval(*, seed: Optional[int] = None, cases: int = 24) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    failures = []
    type_counts: dict[str, int] = {}
    record_only_abstained = 0
    claims_present_abstained = 0
    for case in generated:
        type_counts[case.case_type] = type_counts.get(case.case_type, 0) + 1
        case_ok = True
        details = {}
        for claims_present in (False, True):
            key = "claims_present" if claims_present else "record_only"
            ok, detail = _run_once(case, claims_present=claims_present)
            if ok:
                if claims_present:
                    claims_present_abstained += 1
                else:
                    record_only_abstained += 1
            else:
                case_ok = False
            details[key] = detail
        if not case_ok:
            failures.append({
                "case_id": case.case_id,
                "case_type": case.case_type,
                "question": case.question,
                "modes": details,
            })
    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        "checks": cases * 2,
        "abstained": cases - len(failures),
        "record_only_abstained": record_only_abstained,
        "claims_present_abstained": claims_present_abstained,
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
