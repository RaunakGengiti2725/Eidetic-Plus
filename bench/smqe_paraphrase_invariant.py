"""Rotating paraphrase robustness eval for SMQE.

This sidecar mutates question and evidence wording for the same structured memory operators used by
``smqe_synthetic_invariant``. A case only counts as correct when it passes twice: once with no claims
so the record backend must work, and once with source-backed extracted claims so tier-1 coverage must
work too.
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
class ParaphraseCase:
    case_id: str
    op: str
    question: str
    expected: str
    rows: list[tuple[str, float]]
    forbidden_in_proof: list[str] = field(default_factory=list)
    # P0 fail-closed (2026-07-09): a DERIVED count/sum abstains instead of shipping a verified
    # aggregate (eidetic/smqe/verify.py). Such cases assert abstention under BOTH backends.
    expect_abstain: bool = False


_NAMES = ["Ari", "Nila", "Mika", "Sana", "Theo", "Rowan", "Lina", "Owen"]
_OBJECTS = ["backup badge", "kiln token", "garden permit", "field notebook", "studio key", "travel charger"]
_LOCATIONS = ["Quartz Loft", "North Pier Studio", "Cedar Annex", "Blue Finch Lab", "Orchid Room"]
_COUNT_TARGETS = ["ceramic studios", "tea shops", "library workshops", "bike routes", "garden plots"]
_COUNT_ACTIONS = ["checked out", "sampled", "toured", "explored"]
_PROJECTS = ["mural ledger", "orchid catalog", "harbor map", "kiln checklist", "field guide"]
_SUM_ACTIONS = ["logged", "recorded", "tracked"]
_GOOD = ["mint tea", "fantasy novels", "graphite pens", "berry salad", "quiet playlists"]
_BAD = ["cedar tea", "tax manuals", "brass pens", "kale salad", "alarm playlists"]
_TABLE_VALUES = ["7 AM", "late", "north desk", "2 PM", "midday", "west desk"]


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
        raw_uri="mem://synthetic-smqe-paraphrase",
    )


def _latest_case(rng: random.Random, idx: int) -> ParaphraseCase:
    name = _pick(rng, _NAMES)
    other = _pick(rng, [x for x in _NAMES if x != name])
    obj = _pick(rng, _OBJECTS, f" {idx}")
    loc = _pick(rng, _LOCATIONS)
    decoy = _pick(rng, [x for x in _LOCATIONS if x != loc])
    t = 1_701_000_000 + idx * 100
    return ParaphraseCase(
        case_id=f"latest-para-{idx}",
        op="latest_value",
        question=f"Where did {name} leave the {obj}?",
        expected=loc,
        rows=[
            (f"{other}: I left the {obj} inside {decoy}.", t),
            (f"{name}: I left the {obj} inside {loc}.", t + 1),
        ],
        forbidden_in_proof=[other, decoy],
    )


def _count_case(rng: random.Random, idx: int) -> ParaphraseCase:
    target = _pick(rng, _COUNT_TARGETS)
    singular = target[:-1] if target.endswith("s") else target
    action = _pick(rng, _COUNT_ACTIONS)
    n = rng.randint(2, 5)
    labels = rng.sample(_LOCATIONS, k=n)
    decoy = _pick(rng, [x for x in _COUNT_TARGETS if x != target])
    t = 1_701_100_000 + idx * 100
    rows = [
        (f"User: I {action} the {label} {singular} this month.", t + j)
        for j, label in enumerate(labels)
    ]
    rows.extend([
        (f"User: I bookmarked a directory of {target}.", t + 30),
        (f"User: I {action} {n + 4} different {decoy} this month.", t + 31),
    ])
    return ParaphraseCase(
        case_id=f"count-para-{idx}",
        op="count_aggregate",
        question=f"What is the number of {target} I {action} this month?",
        expected=str(n),
        rows=rows,
        forbidden_in_proof=["directory", decoy, f"{n + 4} different"],
        expect_abstain=True,
    )


def _relative_case(rng: random.Random, idx: int) -> ParaphraseCase:
    item = _pick(rng, _OBJECTS, f" {idx}")
    decoy = _pick(rng, [x for x in _OBJECTS if x not in item], f" {idx}")
    action = _pick(rng, ["collected", "mailed", "filed"])
    ref = datetime(2024, rng.randint(2, 10), rng.randint(10, 24), 12, 0)
    t = ref.timestamp()
    return ParaphraseCase(
        case_id=f"relative-para-{idx}",
        op="relative_temporal",
        question=f"What date did I {action} the {item}?",
        expected=(ref - timedelta(days=1)).date().isoformat(),
        rows=[
            (f"User: Yesterday I {action} the {item}.", t),
            (f"User: Yesterday I {action} the {decoy}.", t + 1),
        ],
        forbidden_in_proof=[decoy],
    )


def _table_case(rng: random.Random, idx: int) -> ParaphraseCase:
    person = _pick(rng, _NAMES)
    other = _pick(rng, [x for x in _NAMES if x != person])
    day = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"])
    other_day = rng.choice([d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"] if d != day])
    value = _pick(rng, _TABLE_VALUES)
    decoy = _pick(rng, [x for x in _TABLE_VALUES if x != value])
    t = 1_701_200_000 + idx * 100
    return ParaphraseCase(
        case_id=f"table-para-{idx}",
        op="table_lookup",
        question=f"In the schedule, what is listed for {person} on {day}?",
        expected=value,
        rows=[
            (
                f"| Name | {day} | {other_day} |\n"
                f"| {other} | {decoy} | off |\n"
                f"| {person} | {value} | standby |",
                t,
            )
        ],
        forbidden_in_proof=[other, decoy],
    )


def _preference_case(rng: random.Random, idx: int) -> ParaphraseCase:
    good = _pick(rng, _GOOD, f" {idx}")
    bad = _pick(rng, _BAD, f" {idx}")
    t = 1_701_300_000 + idx * 100
    return ParaphraseCase(
        case_id=f"preference-para-{idx}",
        op="preference_synth",
        question=f"Which would I rather pick, {bad} or {good}?",
        expected=good,
        rows=[
            (f"User: The archive mentions {bad} as a label.", t),
            (f"User: I avoid {bad} before meetings.", t + 1),
            (f"User: I like {good} during long work sessions.", t + 2),
        ],
        forbidden_in_proof=[bad],
    )


def _open_case(rng: random.Random, idx: int) -> ParaphraseCase:
    good = _pick(rng, _GOOD, f" {idx}")
    bad = _pick(rng, _BAD, f" {idx}")
    t = 1_701_350_000 + idx * 100
    return ParaphraseCase(
        case_id=f"open-para-{idx}",
        op="open_inference",
        question=f"Which would probably be better for me between {bad} or {good}?",
        expected=good,
        rows=[
            (f"User: {bad} makes me uncomfortable when I need to focus.", t),
            (f"User: I usually enjoy {good} when I need to focus.", t + 1),
        ],
        forbidden_in_proof=[bad],
    )


def _speaker_case(rng: random.Random, idx: int) -> ParaphraseCase:
    speaker = _pick(rng, _NAMES)
    other = _pick(rng, [x for x in _NAMES if x != speaker])
    topic = _pick(rng, _OBJECTS, f" {idx}")
    other_topic = _pick(rng, [x for x in _OBJECTS if x not in topic], f" {idx}")
    wanted = _pick(rng, _LOCATIONS).lower()
    distractor = _pick(rng, [x for x in _LOCATIONS if x.lower() != wanted]).lower()
    t = 1_701_400_000 + idx * 100
    expected = f"that the {topic} belongs in the {wanted}"
    return ParaphraseCase(
        case_id=f"speaker-para-{idx}",
        op="speaker_fact",
        question=f"What did {speaker} mention regarding the {topic}?",
        expected=expected,
        rows=[
            (f"{speaker}: I mentioned that the {other_topic} belongs in the {wanted}.", t),
            (f"{other}: I mentioned that the {topic} belongs in the {distractor}.", t + 1),
            (f"{speaker}: I mentioned that the {topic} belongs in the {wanted}.", t + 2),
        ],
        forbidden_in_proof=[other, distractor, other_topic],
    )


def _delta_case(rng: random.Random, idx: int) -> ParaphraseCase:
    start_item = _pick(rng, _OBJECTS, f" {idx}")
    finish_item = _pick(rng, [x for x in _OBJECTS if x not in start_item], f" {idx}")
    decoy = _pick(rng, [x for x in _OBJECTS if x not in start_item and x not in finish_item], f" {idx}")
    start = datetime(2024, rng.randint(1, 9), rng.randint(2, 12), 12, 0)
    days = rng.randint(3, 11)
    finish = start + timedelta(days=days)
    return ParaphraseCase(
        case_id=f"delta-para-{idx}",
        op="temporal_delta",
        question=f"How many days passed between opening the {start_item} and closing the {finish_item}?",
        expected=f"{days} days",
        rows=[
            (f"User: I opened the {start_item} today.", start.timestamp()),
            (f"User: I opened the {decoy} today.", (start + timedelta(days=1)).timestamp()),
            (f"User: I closed the {finish_item} today.", finish.timestamp()),
        ],
        forbidden_in_proof=[decoy],
    )


def _sum_case(rng: random.Random, idx: int) -> ParaphraseCase:
    project = _pick(rng, _PROJECTS, f" {idx}")
    decoy = _pick(rng, [x for x in _PROJECTS if x not in project], f" {idx}")
    action = _pick(rng, _SUM_ACTIONS)
    first = rng.randint(1, 4)
    second = rng.randint(2, 5)
    total = first + second
    t = 1_701_500_000 + idx * 100
    return ParaphraseCase(
        case_id=f"sum-para-{idx}",
        op="multi_session_sum",
        question=f"What is the combined number of hours I {action} for the {project}?",
        expected=f"{total} hours",
        rows=[
            (f"User: I {action} {first} hours for the {project}.", t),
            (f"User: I {action} {second} hours for the {project}.", t + 1),
            (f"User: I {action} {total + 5} hours for the {decoy}.", t + 2),
        ],
        forbidden_in_proof=[decoy, f"{total + 5} hours"],
        expect_abstain=True,
    )


_GENERATORS: list[Callable[[random.Random, int], ParaphraseCase]] = [
    _latest_case,
    _count_case,
    _relative_case,
    _table_case,
    _preference_case,
    _open_case,
    _speaker_case,
    _delta_case,
    _sum_case,
]


def generate_cases(seed: int, cases: int) -> list[ParaphraseCase]:
    rng = random.Random(seed)
    out = [_GENERATORS[idx % len(_GENERATORS)](rng, idx) for idx in range(cases)]
    rng.shuffle(out)
    return out


def _load_case(store: RecordStore, scope: Scope, case: ParaphraseCase, *, add_claims: bool) -> None:
    for text, valid_at in case.rows:
        rec = _record(text, scope=scope, valid_at=valid_at)
        store.upsert_record(rec)
        if add_claims:
            store.add_claims(claims_for_record(rec))


def _run_once(case: ParaphraseCase, *, backend: str) -> tuple[bool, dict, int]:
    with tempfile.TemporaryDirectory(prefix=f"smqe-para-{backend}-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        scope = Scope(namespace=f"smqe-para-{backend}-{case.case_id}")
        _load_case(store, scope, case, add_claims=(backend == "claim"))
        ans = structured_answer(retriever, case.question, at=1_800_000_000, verify=True, scope=scope)
        note = ans.note if ans else ""
        actual_backend = (note.split(":") + ["", "", ""])[2] if note.startswith("smqe:") else ""
        proof = " ".join(c.snippet for c in (ans.citations if ans else []))
        if case.expect_abstain:
            ok = ans is None  # derived aggregate fails closed under both backends
        else:
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
    op_counts: dict[str, int] = {}
    backend_counts = {"claim": 0, "record": 0}
    proof_tokens = 0
    record_correct = 0
    claim_correct = 0
    for case in generated:
        op_counts[case.op] = op_counts.get(case.op, 0) + 1
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
                "op": case.op,
                "question": case.question,
                "expected": case.expected,
                "backends": backend_details,
            })
    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        "checks": cases * 2,
        "correct": cases - len(failures),
        # P0 fail-closed: expect_abstain cases pass by ABSTAINING under both backends; they are
        # counted in record/claim_backend_correct as contract-passes, not as backend answers.
        # Published so a gate reader can scope the answered contract to cases - this count.
        "expected_abstain_cases": sum(
            1 for c in generated if getattr(c, "expect_abstain", False)),
        "record_backend_correct": record_correct,
        "claim_backend_correct": claim_correct,
        "failures": failures,
        "operator_counts": dict(sorted(op_counts.items())),
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
