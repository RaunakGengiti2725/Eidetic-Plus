"""The curiosity loop: the organism does not wait to be asked.

Every sleep/improve tick samples the epistemic frontier (contested first, then
unknown by expected information gain), invents a probe for each cell from a
DETERMINISTIC template (zero probe-generation tokens, fully replayable), and
answers it through the REAL prove path (`Engine.ask`, verify=True). Outcomes:

  PASSED            -> the cell becomes KNOWN, with the verified answer as proof
  MISSING           -> a ResearchTask (the gap needs repair/operator work)
  HARD_TO_RETRIEVE  -> a ResearchTask (retrieval/read research)
  CONTRADICTED      -> a ContestedResolutionProgram task

The diagnosis discriminator is the SAME MemMA rule already shipped in
`eidetic/dreaming/repair.py` -- curiosity is MemMA generalized from anomaly-targeted
records to the whole epistemic frontier.

Public answers are untouched: probes run under the identical verify-or-abstain
contract, and probe outcomes write to the map + agenda, never to a user surface.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..dreaming.repair import Diagnosis, diagnose
from ..models import AnswerStatus, Scope, now
from .cells import CellKind, EpistemicCell
from .map import KnowledgeMap

_COVERAGE_NOTE_RE = re.compile(r"coverage\s+([0-9.]+)")


def probe_for_cell(cell: EpistemicCell) -> str:
    """Deterministic probe template per cell kind. Same cell -> same probe, always."""
    kind = cell.kind
    if kind == CellKind.QUERY.value:
        return cell.subject                                  # replay the question verbatim
    if kind == CellKind.FACT.value:
        return f"What is {cell.subject}'s current {cell.relation}?"
    if kind == CellKind.EVENT_DATE.value:
        return f"When did this happen: {cell.subject}?"
    if kind == CellKind.TEMPORAL_HOLE.value:
        relation = cell.relation.split("@", 1)[0]
        return f"What was {cell.subject}'s {relation} during the gap between the known values?"
    if kind == CellKind.LAW_PREDICTION.value:
        rel, _, obj = cell.relation.partition("?")
        return f"Is it true that {cell.subject} {rel.replace('_', ' ')} {obj}?"
    if kind == CellKind.CONFLICT.value:
        if cell.relation == "nli_conflict":
            return cell.subject
        return f"What is {cell.subject}'s current {cell.relation}?"
    return f"What do we know about {cell.subject}?"


def _diagnose_answer(answer, abstention_threshold: float) -> Diagnosis:
    """Map a governed Answer onto the MemMA diagnosis rule, deterministically."""
    verified = answer is not None and answer.status == AnswerStatus.VERIFIED
    note = (answer.note or "") if answer is not None else ""
    contradicted = "contradict" in note
    m = _COVERAGE_NOTE_RE.search(note)
    if m:
        coverage = float(m.group(1))
    elif answer is not None and answer.retrieved_count > 0:
        coverage = abstention_threshold          # evidence retrieved but not proven
    else:
        coverage = 0.0                            # nothing retrieved at all
    return diagnose(verified, coverage, contradicted, abstention_threshold)


def run_curiosity(engine, scope: Scope, *, max_probes: int, agenda=None,
                  at: Optional[float] = None, probes_log: Optional[Path] = None) -> dict:
    """One curiosity wave. Returns a report; appends one jsonl row per probe when
    `probes_log` is given (the auditable overnight artifact)."""
    kmap: KnowledgeMap = engine.knowledge_map_store
    read_at = at if at is not None else now()
    cells = kmap.sample_frontier(scope, max(0, int(max_probes)))
    report = {"probed": 0, "passed": 0, "missing": 0, "hard_to_retrieve": 0,
              "contradicted": 0, "errors": 0, "cells": []}
    rows = []
    for cell in cells:
        probe = probe_for_cell(cell)
        try:
            answer = engine.ask(probe, scope=scope, as_of=read_at,
                                verify=True, use_cache=False)
        except Exception as e:                    # a failed probe must never crash sleep
            report["errors"] += 1
            rows.append({"ts": now(), "cell_id": cell.cell_id, "probe": probe,
                         "error": f"{type(e).__name__}: {str(e)[:160]}"})
            continue
        diag = _diagnose_answer(answer, engine.settings.abstention_threshold)
        kmap.on_probe_outcome(cell.cell_id, answer, diagnosis=diag.value)
        report["probed"] += 1
        report[diag.value if diag != Diagnosis.PASSED else "passed"] = report.get(
            diag.value if diag != Diagnosis.PASSED else "passed", 0) + 1
        report["cells"].append({"cell_id": cell.cell_id, "diagnosis": diag.value})
        if agenda is not None and diag != Diagnosis.PASSED:
            from ..autoresearch.types import ResearchTask, failure_class_for_diagnosis
            agenda.enqueue(ResearchTask(
                query=probe,
                namespace=scope.namespace,
                agent_id=scope.agent_id,
                project_id=scope.project_id,
                failure_class=failure_class_for_diagnosis(diag.value),
                origin=("contested_cell" if diag == Diagnosis.CONTRADICTED
                        else "unknown_cell"),
                cell_id=cell.cell_id,
                priority_hint=cell.info_gain,
            ))
        rows.append({
            "ts": now(), "cell_id": cell.cell_id, "kind": cell.kind, "probe": probe,
            "status": answer.status.value, "diagnosis": diag.value,
            "citations": len(answer.citations), "note": (answer.note or "")[:160],
        })
    if probes_log is not None and rows:
        probes_log = Path(probes_log)
        probes_log.parent.mkdir(parents=True, exist_ok=True)
        with open(probes_log, "a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    return report
