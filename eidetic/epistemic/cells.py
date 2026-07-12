"""Epistemic cells: the claim-shaped units the knowledge map tracks.

A cell is deterministic in identity: the same (scope, kind, subject, relation)
always hashes to the same cell_id, so enumerators re-running over the same store
converge instead of duplicating, and a probe outcome lands on the exact cell that
requested it.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..models import now


class CellState(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    CONTESTED = "CONTESTED"


class CellKind(str, Enum):
    FACT = "fact"                    # (subject, relation) current-value cell
    EVENT_DATE = "event_date"        # an event whose date is unresolved
    TEMPORAL_HOLE = "temporal_hole"  # validity-window gap inside a fact chain
    LAW_PREDICTION = "law_prediction"  # rule-predicted, unwitnessed fact
    QUERY = "query"                  # a concrete question (from ask/probe outcomes)
    CONFLICT = "conflict"            # a contested (subject, relation) or NLI conflict


_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip().lower())


def cell_id_for(namespace: str, agent_id: Optional[str], project_id: Optional[str],
                kind: str, subject: str, relation: str) -> str:
    key = "\x1f".join([
        _norm(namespace or "default"), _norm(agent_id or ""), _norm(project_id or ""),
        _norm(kind), _norm(subject), _norm(relation),
    ])
    return "cell_" + hashlib.sha256(key.encode()).hexdigest()[:20]


@dataclass
class EpistemicCell:
    namespace: str
    kind: str                       # CellKind value
    subject: str                    # entity / event fact / query text (kind-dependent)
    relation: str = ""              # relation / verb / "" for query cells
    state: str = CellState.UNKNOWN.value
    agent_id: Optional[str] = None
    project_id: Optional[str] = None
    reason: str = ""                # WHY the cell is in this state (human-auditable)
    evidence_ids: list[str] = field(default_factory=list)   # memory/edge/event ids
    proof: Optional[dict] = None    # KNOWN only: {answer, citations:[{memory_id, content_hash}], ts}
    origin: str = "enumerator"      # enumerator | ask | probe | contradiction | law | contested
    info_gain: float = 0.0          # priority signal (e.g. rule confidence, query recurrence)
    first_seen: float = field(default_factory=now)
    last_updated: float = field(default_factory=now)
    last_probed: Optional[float] = None
    probe_count: int = 0

    @property
    def cell_id(self) -> str:
        return cell_id_for(self.namespace, self.agent_id, self.project_id,
                           self.kind, self.subject, self.relation)

    def to_row(self) -> tuple:
        return (
            self.cell_id, self.namespace, self.agent_id, self.project_id,
            self.kind, self.subject, self.relation, self.state, self.reason,
            json.dumps(self.evidence_ids),
            json.dumps(self.proof) if self.proof is not None else None,
            self.origin, float(self.info_gain), float(self.first_seen),
            float(self.last_updated),
            float(self.last_probed) if self.last_probed is not None else None,
            int(self.probe_count),
        )

    @classmethod
    def from_row(cls, r) -> "EpistemicCell":
        cell = cls(
            namespace=r["namespace"], kind=r["kind"], subject=r["subject"],
            relation=r["relation"] or "", state=r["state"],
            agent_id=r["agent_id"], project_id=r["project_id"],
            reason=r["reason"] or "",
            evidence_ids=json.loads(r["evidence"] or "[]"),
            proof=json.loads(r["proof"]) if r["proof"] else None,
            origin=r["origin"] or "enumerator",
            info_gain=float(r["info_gain"] or 0.0),
            first_seen=float(r["first_seen"]),
            last_updated=float(r["last_updated"]),
            last_probed=float(r["last_probed"]) if r["last_probed"] is not None else None,
            probe_count=int(r["probe_count"] or 0),
        )
        return cell

    def brief(self) -> dict:
        """Public-surface projection: NO raw answer text beyond the cell naming --
        research surfaces must never leak unverified drafts."""
        return {
            "cell_id": self.cell_id,
            "kind": self.kind,
            "subject": self.subject[:160],
            "relation": self.relation[:80],
            "state": self.state,
            "reason": self.reason[:240],
            "info_gain": round(self.info_gain, 4),
            "probe_count": self.probe_count,
        }
