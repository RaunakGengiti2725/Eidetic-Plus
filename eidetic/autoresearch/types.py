"""Research types + the failure-class taxonomy (mined from real r18/r19 forensics).

`classify_failure` is deterministic over the governed Answer's note + SMQE note tags,
so a task's failure_class is auditable from the answer that spawned it.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..models import now


class FailureClass(str, Enum):
    ENTAIL_FAILURE = "entail_failure"            # 'no source entails' (11x r19 -- READ target)
    CONTRADICTION_VETO = "contradiction_veto"    # contradiction gate abstention
    SPAN_UNGROUNDED = "span_ungrounded"          # sentence-level claim not grounded
    LOW_COVERAGE = "low_coverage"                # insufficient evidence pre-reader
    FORM_FLOOR = "form_floor"                    # non-responsive answer form
    NUMERIC_FLOOR = "numeric_floor"              # aggregation number not stated in a source
    TEMPORAL_SELECTION = "temporal_selection"    # mention_selected relative_temporal
    LATEST_VALUE_SELECTION = "latest_value_selection"
    SUGGESTION_SYNTH = "suggestion_synth"
    FALSE_PREMISE = "false_premise"
    COVE_UNGROUNDED = "cove_ungrounded"
    VERIFIED_WRONG = "verified_wrong"            # bench-only signal (audit wall applies)
    MISSING_KNOWLEDGE = "missing_knowledge"      # curiosity: nothing retrieved for the gap
    HARD_TO_RETRIEVE = "hard_to_retrieve"        # curiosity: present but unprovable
    CONTESTED_CONFLICT = "contested_conflict"    # curiosity/ask: contradiction to resolve
    SURPRISE_INGEST = "surprise_ingest"
    REPAIR_PROPOSAL = "repair_proposal"
    KNOB_IMBALANCE = "knob_imbalance"
    UNKNOWN = "unknown"


# Ordered, first-match-wins note patterns (mined from every observed r18/r19 shape).
_NOTE_RULES: tuple[tuple[str, FailureClass], ...] = (
    (r"false-premise", FailureClass.FALSE_PREMISE),
    (r"contradicts the answer", FailureClass.CONTRADICTION_VETO),
    (r"CoVe verification question", FailureClass.COVE_UNGROUNDED),
    (r"sentence-level claim was not grounded", FailureClass.SPAN_UNGROUNDED),
    (r"answer form is non-responsive|verbatim echo|fragment answers nothing",
     FailureClass.FORM_FLOOR),
    (r"number is not stated in any", FailureClass.NUMERIC_FLOOR),
    (r"temporal_selection|mention_selected", FailureClass.TEMPORAL_SELECTION),
    (r"insufficient evidence \(coverage", FailureClass.LOW_COVERAGE),
    (r"confidence [0-9.]+ < tau", FailureClass.LOW_COVERAGE),
    (r"no source entails", FailureClass.ENTAIL_FAILURE),
    (r"empty-or-no-active-memory|no_active_records", FailureClass.MISSING_KNOWLEDGE),
)

_SMQE_NOTE_RULES: tuple[tuple[str, FailureClass], ...] = (
    (r"smqe:latest_value", FailureClass.LATEST_VALUE_SELECTION),
    (r"suggestion_synth", FailureClass.SUGGESTION_SYNTH),
    (r"relative_temporal.*mention_selected", FailureClass.TEMPORAL_SELECTION),
)


def classify_failure(note: str, extra: Optional[dict] = None) -> FailureClass:
    """Deterministic Answer.note -> FailureClass. Bench rows may pass `extra`
    (e.g. {'verified': True, 'correct': False}) to surface VERIFIED_WRONG."""
    note = note or ""
    extra = extra or {}
    if extra.get("verified") and extra.get("correct") is False:
        for pattern, cls in _SMQE_NOTE_RULES:
            if re.search(pattern, note, re.I):
                return cls
        return FailureClass.VERIFIED_WRONG
    for pattern, cls in _NOTE_RULES:
        if re.search(pattern, note, re.I):
            return cls
    for pattern, cls in _SMQE_NOTE_RULES:
        if re.search(pattern, note, re.I):
            return cls
    return FailureClass.UNKNOWN


def failure_class_for_diagnosis(diagnosis: str) -> FailureClass:
    """MemMA/curiosity diagnosis -> failure class (see dreaming/repair.py)."""
    return {
        "missing": FailureClass.MISSING_KNOWLEDGE,
        "hard_to_retrieve": FailureClass.HARD_TO_RETRIEVE,
        "contradicted": FailureClass.CONTESTED_CONFLICT,
    }.get(diagnosis, FailureClass.UNKNOWN)


# Frontier priority (addendum §1): CONTESTED > UNKNOWN×gain > live ask failures >
# repair proposals > surprise ingest > knob imbalance. Base scores; UNKNOWN cells add
# their info_gain on top via priority_hint.
_ORIGIN_BASE_PRIORITY = {
    "contested_cell": 100.0,
    "unknown_cell": 80.0,
    "ask_fail": 60.0,
    "verified_wrong": 65.0,        # ground the map in live failures, above plain abstains
    "repair": 40.0,
    "surprise": 30.0,
    "knob_imbalance": 10.0,
}


@dataclass
class ResearchTask:
    query: str
    namespace: str = "default"
    agent_id: Optional[str] = None
    project_id: Optional[str] = None
    failure_class: FailureClass = FailureClass.UNKNOWN
    origin: str = "ask_fail"        # keys of _ORIGIN_BASE_PRIORITY
    cell_id: str = ""               # epistemic cell that spawned this (when frontier-born)
    candidate_ids: list[str] = field(default_factory=list)
    trace_id: str = ""
    priority_hint: float = 0.0      # e.g. cell info_gain / recurrence
    ts: float = field(default_factory=now)
    source: str = "live"            # live | dev_lab (integrity wall discriminator)

    @property
    def priority(self) -> float:
        return _ORIGIN_BASE_PRIORITY.get(self.origin, 20.0) + float(self.priority_hint)

    @property
    def dedup_key(self) -> str:
        raw = "\x1f".join([self.namespace, self.cell_id or
                           hashlib.sha256(self.query.lower().encode()).hexdigest()[:16],
                           self.failure_class.value])
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def to_json(self) -> str:
        d = dict(self.__dict__)
        d["failure_class"] = self.failure_class.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "ResearchTask":
        d = json.loads(s)
        d["failure_class"] = FailureClass(d.get("failure_class", "unknown"))
        return cls(**d)


@dataclass
class ResearchHypothesis:
    """ONE mind-layer change. Exactly one of knob/pipeline/law_id is set."""
    tier: str                        # "A" (knob) | "B" (operator pipeline) | "C" (law)
    failure_class: FailureClass
    rationale: str
    knob: str = ""                   # Tier A: env var name
    value: str = ""                  # Tier A: env var value
    pipeline: Optional[dict] = None  # Tier B: OperatorPipeline dict
    law_id: str = ""                 # Tier C: candidate law identifier
    origin_cell: str = ""
    origin_task_query: str = ""

    @property
    def key(self) -> str:
        """Identity for ResearchMemory dedup/blocking."""
        if self.tier == "A":
            payload = f"A:{self.knob}={self.value}"
        elif self.tier == "B":
            payload = "B:" + json.dumps(self.pipeline or {}, sort_keys=True)
        else:
            payload = f"C:{self.law_id}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def env_overlay(self) -> dict[str, str]:
        """The env this hypothesis applies for its challenger eval."""
        if self.tier == "A":
            try:                                  # COMPRESSION_RATIO expands to two vars
                from bench.sweep import stage_assignment
                return stage_assignment(self.knob, self.value)
            except ImportError:
                return {self.knob: self.value}
        if self.tier == "B":
            from .operators import compile_pipeline
            return compile_pipeline(self.pipeline or {})
        return {}                     # Tier C laws are store-side, not env-side

    def describe(self) -> dict:
        return {
            "tier": self.tier,
            "knob": self.knob or None,
            "value": self.value or None,
            "pipeline": self.pipeline,
            "law_id": self.law_id or None,
            "failure_class": self.failure_class.value,
            "origin_cell": self.origin_cell or None,
            "rationale": self.rationale,
        }


@dataclass
class ResearchTrial:
    trial_id: str
    hypothesis: ResearchHypothesis
    champion_id: str
    challenger_env: dict
    dev_score: float
    champion_score: float
    delta_pp: float
    mcnemar_p: Optional[float]
    paired_n: int
    decision: str                    # ACCEPT | REJECT
    reason: str
    artifact_dir: str
    map_before: dict = field(default_factory=dict)
    map_after: dict = field(default_factory=dict)
    duration_s: float = 0.0
    ts: float = field(default_factory=now)

    def to_row(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "ts": self.ts,
            "tier": self.hypothesis.tier,
            "hypothesis": self.hypothesis.describe(),
            "champion_id": self.champion_id,
            "challenger_env": self.challenger_env,
            "dev_score": self.dev_score,
            "champion_score": self.champion_score,
            "delta_pp": self.delta_pp,
            "mcnemar_p": self.mcnemar_p,
            "paired_n": self.paired_n,
            "decision": self.decision,
            "reason": self.reason,
            "map_before": self.map_before,
            "map_after": self.map_after,
            "artifact_dir": self.artifact_dir,
            "duration_s": round(self.duration_s, 2),
        }


TRIAL_ROW_REQUIRED_FIELDS = frozenset({
    "trial_id", "ts", "tier", "hypothesis", "champion_id", "challenger_env",
    "dev_score", "champion_score", "delta_pp", "mcnemar_p", "paired_n",
    "decision", "reason", "artifact_dir",
})


def validate_trial_row(row: dict) -> list[str]:
    """Schema check for a trials.jsonl row; returns problems (empty = valid)."""
    problems = [f"missing field: {f}" for f in TRIAL_ROW_REQUIRED_FIELDS if f not in row]
    if row.get("decision") not in ("ACCEPT", "REJECT"):
        problems.append(f"bad decision: {row.get('decision')!r}")
    if row.get("tier") not in ("A", "B", "C"):
        problems.append(f"bad tier: {row.get('tier')!r}")
    hyp = row.get("hypothesis")
    if not isinstance(hyp, dict) or "rationale" not in hyp:
        problems.append("hypothesis must be a dict with a rationale")
    return problems
