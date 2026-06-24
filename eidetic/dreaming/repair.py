"""MemMA evidence-grounded self-repair (PDF Theme 1, Stage-1 highest evidence).

The agent quizzes itself, finds what it cannot answer, and patches the hole. After/within the
dreaming pass, for the suspect memories (targeted by anomaly score), it: (1) generates probe
questions, (2) answers them against the current memory via the existing retriever + NLI, (3)
DIAGNOSES each failure (missing info vs hard-to-retrieve vs contradicted), and (4) routes a
SKIP / MERGE / INSERT repair proposal -- which maps almost 1:1 onto the existing conflict
resolver + bi-temporal invalidate-not-delete (graph.add_fact) machinery.

DETERMINISTIC + offline-tested: target selection, the diagnose discriminator, and the
SKIP/MERGE/INSERT router. LLM-GATED (real DashScope calls, fail-loud, flag DREAM_REPAIR off,
NOT run under the current quota block): probe generation + probe answering. The enabled sweep
is deliberately PROPOSAL-ONLY -- it returns the proposed repairs and never auto-mutates the
store, so applying them stays behind the EvolveMem guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Diagnosis(str, Enum):
    PASSED = "passed"
    MISSING = "missing"
    HARD_TO_RETRIEVE = "hard_to_retrieve"
    CONTRADICTED = "contradicted"


class RepairAction(str, Enum):
    SKIP = "skip"
    MERGE = "merge"
    INSERT = "insert"


# diagnosis -> repair action (MemMA SKIP/MERGE/INSERT).
_ROUTE = {
    Diagnosis.PASSED: RepairAction.SKIP,             # the probe was answered -> nothing to fix
    Diagnosis.MISSING: RepairAction.INSERT,          # info absent -> add it
    Diagnosis.HARD_TO_RETRIEVE: RepairAction.MERGE,  # present but fragmented -> consolidate/link
    Diagnosis.CONTRADICTED: RepairAction.MERGE,      # wrong value -> supersede via conflict path
}


def diagnose(probe_passed: bool, coverage: float, contradicted: bool,
             abstention_threshold: float) -> Diagnosis:
    """Classify a probe outcome. A passed probe needs no repair. Otherwise: a contradiction is
    'contradicted'; weak evidence (coverage below the abstention threshold) is 'missing'; present-
    but-unentailed evidence is 'hard_to_retrieve'. Reuses the existing coverage + NLI signals."""
    if probe_passed:
        return Diagnosis.PASSED
    if contradicted:
        return Diagnosis.CONTRADICTED
    if coverage < abstention_threshold:
        return Diagnosis.MISSING
    return Diagnosis.HARD_TO_RETRIEVE


def route_repair(diagnosis: Diagnosis) -> RepairAction:
    return _ROUTE[diagnosis]


def select_repair_targets(records, anomaly_by_id: dict, topk: int) -> list:
    """Deterministically pick the highest-anomaly memories to probe (falls back to the given
    order when no anomaly score is present). Aims the expensive LLM sweep at suspect memories."""
    ranked = sorted(records, key=lambda r: -float(anomaly_by_id.get(r.memory_id, 0.0)))
    return ranked[: max(0, topk)]


@dataclass
class RepairProposal:
    target_id: str
    probe: str
    answer: str
    diagnosis: Diagnosis
    action: RepairAction


def run_sweep(engine, scope=None) -> dict:
    """Gated execution. Off -> immediate no-op (dreaming unchanged). Enabled -> a real,
    PROPOSAL-ONLY MemMA sweep (fail-loud; needs a funded key): probe -> answer -> diagnose ->
    route, returning proposals WITHOUT mutating the store (apply stays behind the guard)."""
    settings = engine.settings
    if not getattr(settings, "dream_repair_enabled", False):
        return {"skipped": "disabled"}

    from ..models import Scope
    from .anomaly import edge_anomaly_scores
    scope = scope or Scope()
    records = [r for r in engine.store.all_records(scope) if (r.text or "").strip()]
    if not records:
        return {"proposals": [], "note": "no records in scope"}

    # Deterministic targeting: rank by per-memory anomaly (mean anomaly of its incident edges).
    anomaly_by_id: dict[str, float] = {}
    try:
        edges = engine.store.all_edges(scope)
        ent_ids = [r.memory_id for r in records]
        vmap = engine.index.get_vectors(ent_ids)
        ent_vectors = {e: v for e, v in zip(ent_ids, (vmap.get(i) for i in ent_ids)) if v is not None}
        if edges and ent_vectors:
            scores = edge_anomaly_scores(edges, ent_vectors)
            for e, a in zip(edges, scores):
                anomaly_by_id[e.source_memory_id] = max(anomaly_by_id.get(e.source_memory_id, 0.0), float(a))
    except Exception:
        pass  # targeting is best-effort; fall back to record order

    targets = select_repair_targets(records, anomaly_by_id, settings.dream_repair_topk)
    proposals: list[dict] = []
    for rec in targets:
        probes = engine.client.generate_probes(rec.text)        # real LLM call (fail-loud)
        for probe in probes:
            ans = engine.retriever.answer(probe, scope=scope, verify=True)  # real LLM call
            passed = ans.verified
            contradicted = any(c.nli_label.value == "contradiction" for c in ans.citations)
            coverage = max((c.nli_score for c in ans.citations), default=0.0)
            diag = diagnose(passed, coverage, contradicted, settings.abstention_threshold)
            action = route_repair(diag)
            if action is not RepairAction.SKIP:
                proposals.append(RepairProposal(rec.memory_id, probe, ans.answer, diag, action).__dict__)
    return {"proposals": proposals, "targets": len(targets),
            "note": "proposal-only; apply behind the EvolveMem guard"}
