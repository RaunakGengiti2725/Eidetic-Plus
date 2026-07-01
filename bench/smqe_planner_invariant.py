"""Rotating SMQE planner invariant.

The planner is the first gate in the low-latency memory path. This sidecar proves it is still a
generic query-shape classifier over invented domains, not a bag of benchmark question strings:

* each supported operator is selected from rotating synthetic wording;
* extracted terms/entities/slots come from the question itself;
* synthesis flags and temporal units are preserved where the executor needs them;
* no dataset rescue policy strings leak into the plan.

No retrieval, embeddings, reader, or model calls are used.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bench.seed_policy import resolve_seed
from eidetic.smqe.planner import plan_query


_FORBIDDEN_PLAN_TOKENS = {
    "longmemeval",
    "locomo",
    "source_scan",
    "source-scan",
    "direct-fact",
    "direct",
    "benchmark",
}
_FORBIDDEN_ENTITY_TOKENS = {
    "can", "could", "did", "do", "does", "how", "shall", "should", "what",
    "when", "where", "which", "who", "why", "will", "would",
}


@dataclass(frozen=True)
class PlannerCase:
    case_id: str
    query: str
    expected_op: str
    required_terms: tuple[str, ...]
    required_slot_terms: tuple[str, ...] = ()
    required_entities: tuple[str, ...] = ()
    expected_unit: str = ""
    expected_synthesis: Optional[bool] = None


_PEOPLE = [
    "Ari Vale", "Mina Cho", "Tessa Rowan", "Noor Patel", "Iris Chen",
    "Leah Stone", "Owen Grey", "Nico Hart",
]
_PROJECTS = [
    "cedar audit", "harbor mural", "lumen trial", "atlas kitchen",
    "river archive", "opal garden", "north lab", "saffron clinic",
]
_ITEMS = [
    "tea shops", "sensor boards", "permit packets", "garden beds", "therapy forms",
    "model kits", "recipe drafts", "studio keys",
]
_ACTIONS = ["visited", "tested", "filed", "prepared", "reviewed", "assembled"]
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _pick(rng: random.Random, values: list[str]) -> str:
    return rng.choice(values)


def _make_case(rng: random.Random, idx: int, op: str) -> PlannerCase:
    project = _pick(rng, _PROJECTS)
    item = _pick(rng, _ITEMS)
    action = _pick(rng, _ACTIONS)
    person = _pick(rng, _PEOPLE)
    weekday = _pick(rng, _WEEKDAYS)
    topic = _pick(rng, _ITEMS)
    other = _pick(rng, [p for p in _PROJECTS if p != project])

    if op == "count_aggregate":
        query = f"How many {item} did I {action} for the {project}?"
        return PlannerCase(
            f"planner-{idx}-count",
            query,
            op,
            required_terms=tuple((item + " " + action + " " + project).split()),
            required_slot_terms=tuple(item.split()[:1]),
        )
    if op == "latest_value":
        slot = rng.choice(["permit status", "budget amount", "pickup location", "access code"])
        query = f"What is my current {slot} for the {project}?"
        return PlannerCase(
            f"planner-{idx}-latest",
            query,
            op,
            required_terms=tuple((slot + " " + project).split()),
            required_slot_terms=tuple(slot.split()),
        )
    if op == "table_lookup":
        query = f"What shift does {person} have on {weekday} in the {project} schedule?"
        return PlannerCase(
            f"planner-{idx}-table",
            query,
            op,
            required_terms=tuple((person + " " + weekday + " " + project + " schedule").lower().split()),
            required_entities=(person,),
        )
    if op == "temporal_delta":
        query = f"How many days passed since the {project} kickoff?"
        return PlannerCase(
            f"planner-{idx}-delta",
            query,
            op,
            required_terms=tuple((project + " kickoff").split()),
            expected_unit="days",
        )
    if op == "multi_session_sum":
        query = f"How much total time did I spend on {project} across the follow-up sessions?"
        return PlannerCase(
            f"planner-{idx}-sum",
            query,
            op,
            required_terms=tuple((project + " total time sessions").split()),
        )
    if op == "preference_synth":
        query = f"What should I choose for the {project} based on my preferences?"
        return PlannerCase(
            f"planner-{idx}-preference",
            query,
            op,
            required_terms=tuple((project + " choose preferences").split()),
            expected_synthesis=True,
        )
    if op == "speaker_fact":
        query = f"What did {person} say about the {topic}?"
        return PlannerCase(
            f"planner-{idx}-speaker",
            query,
            op,
            required_terms=tuple((person + " say " + topic).lower().split()),
            required_entities=(person,),
        )
    if op == "relative_temporal":
        query = f"When will I schedule the {project} checklist?"
        return PlannerCase(
            f"planner-{idx}-relative",
            query,
            op,
            required_terms=tuple((project + " checklist").split()),
        )
    if op == "event_order":
        query = f"Which happened first, starting the {project} or finishing the {other}?"
        return PlannerCase(
            f"planner-{idx}-order",
            query,
            op,
            required_terms=tuple((project + " " + other + " first").split()),
        )
    if op == "open_inference":
        query = f"Would the {project} probably be better for me or the {other}?"
        return PlannerCase(
            f"planner-{idx}-open",
            query,
            op,
            required_terms=tuple((project + " " + other + " probably better").split()),
            expected_synthesis=True,
        )
    raise ValueError(f"unsupported planner op: {op}")


def generate_cases(seed: int, cases: int) -> list[PlannerCase]:
    rng = random.Random(seed)
    ops = [
        "count_aggregate",
        "latest_value",
        "table_lookup",
        "temporal_delta",
        "multi_session_sum",
        "preference_synth",
        "speaker_fact",
        "relative_temporal",
        "event_order",
        "open_inference",
    ]
    out: list[PlannerCase] = []
    idx = 0
    while len(out) < cases:
        for op in ops:
            if len(out) >= cases:
                break
            out.append(_make_case(rng, idx, op))
            idx += 1
    rng.shuffle(out)
    return out


def _plan_blob(plan) -> str:
    try:
        return plan.model_dump_json().lower()
    except Exception:
        return json.dumps(getattr(plan, "__dict__", {}), sort_keys=True).lower()


def _run_case(case: PlannerCase, *, as_of: float) -> tuple[int, int, dict]:
    start = time.perf_counter()
    plan = plan_query(case.query, as_of)
    latency_ms = (time.perf_counter() - start) * 1000.0
    terms = {str(t).lower() for t in (plan.filters.get("terms") or [])}
    slot_terms = {str(t).lower() for t in plan.slot.split()}
    entity_blob = " ".join(plan.entities).lower()
    blob = _plan_blob(plan)

    checks = 0
    correct = 0

    def check(ok: bool) -> None:
        nonlocal checks, correct
        checks += 1
        if ok:
            correct += 1

    check(plan.op == case.expected_op)
    check(plan.as_of == as_of)
    check(plan.backend_preference == "auto")
    check(not any(token in blob for token in _FORBIDDEN_PLAN_TOKENS))
    check(not any(token in entity_blob.split() for token in _FORBIDDEN_ENTITY_TOKENS))
    check(all(term.lower() in terms for term in case.required_terms))
    if case.required_slot_terms:
        check(all(term.lower() in slot_terms for term in case.required_slot_terms))
    if case.required_entities:
        check(all(entity.lower() in entity_blob for entity in case.required_entities))
    if case.expected_unit:
        check(plan.unit == case.expected_unit)
    if case.expected_synthesis is not None:
        check(plan.requires_synthesis is case.expected_synthesis)

    return correct, checks, {
        "case_id": case.case_id,
        "query": case.query,
        "expected_op": case.expected_op,
        "actual_op": plan.op,
        "terms": sorted(terms),
        "slot": plan.slot,
        "entities": list(plan.entities),
        "requires_synthesis": plan.requires_synthesis,
        "unit": plan.unit,
        "latency_ms": round(latency_ms, 6),
    }


def run_eval(seed: int | str | None = None, *, cases: int = 24) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    resolved_seed, seed_mode = resolve_seed(seed)
    as_of = 1_900_000_000.0
    generated = generate_cases(resolved_seed, cases)
    total_checks = 0
    correct = 0
    failures: list[dict] = []
    details: list[dict] = []
    op_counts: Counter[str] = Counter()
    latency_ms: list[float] = []
    for case in generated:
        got, checks, detail = _run_case(case, as_of=as_of)
        total_checks += checks
        correct += got
        op_counts[detail["actual_op"]] += 1
        latency_ms.append(float(detail["latency_ms"]))
        details.append(detail)
        if got != checks:
            failures.append({"case_id": case.case_id, "expected": checks, "got": got, **detail})
    lat_sorted = sorted(latency_ms)
    p95_idx = min(len(lat_sorted) - 1, int(round(0.95 * (len(lat_sorted) - 1))))
    return {
        "pass": correct == total_checks and not failures,
        "seed": resolved_seed,
        "seed_mode": seed_mode,
        "cases": cases,
        "checks": total_checks,
        "correct": correct,
        "operator_counts": dict(sorted(op_counts.items())),
        "generic_term_checks": cases,
        "case_type_counts": {"smqe_planner_generic_shape": cases},
        "p95_latency_ms": round(lat_sorted[p95_idx], 6) if lat_sorted else 0.0,
        "max_latency_ms": round(max(latency_ms), 6) if latency_ms else 0.0,
        "failures": failures,
        "details": details[:50],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=None)
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    report = run_eval(seed=args.seed, cases=args.cases)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
