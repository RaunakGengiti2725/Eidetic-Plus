"""Shared data contracts for the whole engine. Everything imports from here.

Design note on (im)mutability:
  * Raw bytes live in the immutable content-addressed substrate (Component 1) and
    are NEVER mutated or deleted.
  * A MemoryRecord is INDEX/STATE (Component 2/5). It points at the raw bytes via
    `content_hash` / `raw_uri`. Forgetting mutates only its FSRS priority; a
    contradiction mutates only its bi-temporal `invalid_at` / `expired_at`. The
    raw record it points to is untouched. This is the lossless-store /
    forgetting-index decoupling that the whole project rests on.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def now() -> float:
    """Single source of wall-clock time (epoch seconds)."""
    return time.time()


def new_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    BINARY = "binary"  # un-embeddable: store raw, describe, embed the description


class NLILabel(str, Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


# --------------------------------------------------------------------------
# Scope (the universal-plugin isolation boundary).
#
# Memory crosses sessions and tools, so every read/write carries a Scope. The
# `namespace` is a HARD boundary -- a read in namespace A never sees namespace B,
# preventing cross-tool/cross-agent bleed. `agent_id`/`project_id` are optional
# finer filters within a namespace. The default namespace is the explicit string
# "default" -- never a global wildcard.
# --------------------------------------------------------------------------
class Scope(BaseModel):
    namespace: str = "default"
    agent_id: Optional[str] = None
    project_id: Optional[str] = None

    def visible_to(self, q: "Scope") -> bool:
        """Is a record in THIS scope visible to a query in scope `q`?"""
        if self.namespace != q.namespace:
            return False
        if q.agent_id is not None and self.agent_id != q.agent_id:
            return False
        if q.project_id is not None and self.project_id != q.project_id:
            return False
        return True

    def key(self) -> str:
        return f"{self.namespace}\x1f{self.agent_id or ''}\x1f{self.project_id or ''}"


# --------------------------------------------------------------------------
# FSRS / DSR forgetting state (Component 5). Sets index priority ONLY.
# --------------------------------------------------------------------------
class FSRSState(BaseModel):
    stability: float = 1.0       # days until retrievability decays to ~90%
    difficulty: float = 5.0      # 1..10
    retrievability: float = 1.0  # current recall probability (power-law)
    last_review: float = Field(default_factory=now)
    reps: int = 0
    lapses: int = 0

    def priority(self, at: Optional[float] = None) -> float:
        """Index-priority weight in [0,1]. Used for replay/consolidation scheduling
        and ambient surfacing ONLY -- deliberately kept OUT of the cued-retrieval
        ranking path so that recall@k stays age-independent (see retrieval.py)."""
        from .fsrs import current_retrievability  # local import avoids cycle

        return current_retrievability(self, at if at is not None else now())


# --------------------------------------------------------------------------
# The memory record (mutable index/state over an immutable raw blob).
# --------------------------------------------------------------------------
class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: new_id("mem"))
    content_hash: str = ""              # sha256 hex of the raw bytes
    modality: Modality = Modality.TEXT
    raw_uri: str = ""                   # locator into the immutable substrate
    raw_bytes_len: int = 0

    # Embeddable text: the original text, OR a transcription/OCR/description.
    text: str = ""
    is_described: bool = False          # True when `text` describes non-text raw
    summary: Optional[str] = None       # consolidation gist; NEVER replaces raw

    source: str = "user"                # provenance label
    scope: Scope = Field(default_factory=Scope)  # isolation boundary (plugin scoping)

    # Bi-temporal coordinates (Component 7).
    created_at: float = Field(default_factory=now)   # ingestion time (system knew)
    valid_at: float = Field(default_factory=now)     # event/world-valid time
    invalid_at: Optional[float] = None               # world fact stopped being true
    expired_at: Optional[float] = None               # system superseded this knowledge

    entities: list[str] = Field(default_factory=list)

    # Salience (Component 4).
    salience: float = 0.0
    surprise: float = 0.0
    importance: float = 0.0

    fsrs: FSRSState = Field(default_factory=FSRSState)
    consolidated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_active_at(self, t: Optional[float] = None) -> bool:
        """Bi-temporal validity: visible to retrieval at time `t`?"""
        t = now() if t is None else t
        if self.valid_at is not None and self.valid_at > t:
            return False
        if self.invalid_at is not None and self.invalid_at <= t:
            return False
        if self.expired_at is not None and self.expired_at <= t:
            return False
        return True

    def age_days(self, at: Optional[float] = None) -> float:
        at = now() if at is None else at
        return max(0.0, (at - self.valid_at) / 86400.0)


# --------------------------------------------------------------------------
# Bi-temporal knowledge graph (Components 2 & 7).
# --------------------------------------------------------------------------
class Entity(BaseModel):
    name: str
    type: str = "entity"
    first_seen: float = Field(default_factory=now)


# --------------------------------------------------------------------------
# Dreaming engine: the DERIVED layer (additive, reversible, provenance-tagged).
# Schema centroids, multi-resolution gist nodes, and inferred facts live here --
# NEVER written back into the observed lossless store. Content-addressed by `cid`.
# --------------------------------------------------------------------------
class DerivedRecord(BaseModel):
    cid: str = ""                        # content-address (sha1 of kind+namespace+text+members)
    kind: str = "gist"                   # "schema" | "gist" | "inferred_fact"
    namespace: str = "default"
    level: int = 1                       # 0 = raw memory; 1+ = gist/schema levels
    text: str = ""
    member_ids: list[str] = Field(default_factory=list)
    vector: list[float] = Field(default_factory=list)   # centroid embedding (derived only)
    confidence: float = 1.0
    provenance: str = ""
    created_at: float = Field(default_factory=now)


class Edge(BaseModel):
    edge_id: str = Field(default_factory=lambda: new_id("edge"))
    src: str
    dst: str
    relation: str
    fact: str = ""                       # human-readable fact text
    source_memory_id: str = ""
    supersedes: Optional[str] = None     # edge_id this fact superseded (knowledge-update chain)
    scope: Scope = Field(default_factory=Scope)  # graph is scoped too
    # Dreaming-engine derived layer: machine-INFERRED edges live here, NLI/confidence-gated,
    # excluded from observed reads by default and always flagged + provenance-tagged.
    inferred: bool = False
    confidence: float = 1.0
    provenance: str = ""                 # e.g. "transe", "rule:born&capital=>citizen"
    weight: float = 1.0                  # retrieval/replay weight (SHY renormalizes; never deletes)
    pruned: bool = False                 # SHY-pruned from the INDEX (reversible; store intact)
    # Bi-temporal: valid_* = world truth, created/expired = system knowledge.
    valid_at: float = Field(default_factory=now)
    invalid_at: Optional[float] = None
    created_at: float = Field(default_factory=now)
    expired_at: Optional[float] = None   # invalidation closes (never deletes) an edge

    def is_active_at(self, t: Optional[float] = None) -> bool:
        t = now() if t is None else t
        if self.valid_at > t:
            return False
        if self.invalid_at is not None and self.invalid_at <= t:
            return False
        if self.expired_at is not None and self.expired_at <= t:
            return False
        return True


# --------------------------------------------------------------------------
# Retrieval + provenance (Components 6 & 7).
# --------------------------------------------------------------------------
class RetrievalCandidate(BaseModel):
    record: MemoryRecord
    dense_score: float = 0.0
    graph_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class Citation(BaseModel):
    memory_id: str
    content_hash: str
    raw_uri: str
    source: str
    valid_at: float
    snippet: str = ""
    nli_label: NLILabel = NLILabel.NEUTRAL
    nli_score: float = 0.0


class Answer(BaseModel):
    question: str
    answer: str
    verified: bool = False
    confidence: float = 0.0
    citations: list[Citation] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    generated_by: str = ""
    retrieved_count: int = 0
    note: str = ""


# --------------------------------------------------------------------------
# Connected Brain Loop: the brain spine (observation-only contracts).
#
# These make retrieval explainable (RecallTrace), citations portable across
# answer/proof/repair/health (EvidencePacket), and improvement signals uniform
# (BrainEvent). They are a SIDE CHANNEL: nothing here participates in ranking,
# candidate selection, or answer generation -- every field is read off
# already-computed state, so enabling them is byte-identical to the baseline
# read/write path. See eidetic/brain.py for the builders + QualityGate logic.
# --------------------------------------------------------------------------
class RecallTrace(BaseModel):
    """Why a single retrieval found (or missed) what it did. Built behind RECALL_TRACE,
    it observes the candidate list without altering it. Powers proof recall-paths, the
    failure autopsy, and channel-win statistics."""
    query: str = ""
    scope: Scope = Field(default_factory=Scope)
    parsed_query: dict[str, Any] = Field(default_factory=dict)
    enabled_channels: list[str] = Field(default_factory=list)
    channel_results: dict[str, list[str]] = Field(default_factory=dict)   # channel -> ranked ids
    channel_weights: dict[str, float] = Field(default_factory=dict)
    fused_scores: dict[str, float] = Field(default_factory=dict)
    gist_provenance: dict[str, str] = Field(default_factory=dict)         # member_id -> gist cid
    selected_candidates: list[str] = Field(default_factory=list)
    dropped_candidates: list[str] = Field(default_factory=list)
    latency_by_stage: dict[str, float] = Field(default_factory=dict)      # stage -> milliseconds
    token_budget: int = 0
    abstention_reason: str = ""

    def paths_for(self, memory_id: str) -> list[str]:
        """The channels that surfaced `memory_id` -- its retrieval paths, in channel order."""
        return [ch for ch, ids in self.channel_results.items() if memory_id in ids]


class EvidencePacket(BaseModel):
    """A portable citation: the one evidence shape answer, proof, repair, and health all reason
    over (so no subsystem invents its own). Built from an Answer's Citation + its RecallTrace."""
    memory_id: str
    content_hash: str = ""
    raw_uri: str = ""
    raw_span: str = ""
    valid_at: float = 0.0
    invalid_at: Optional[float] = None
    retrieval_paths: list[str] = Field(default_factory=list)   # channels that surfaced it
    channel_scores: dict[str, float] = Field(default_factory=dict)
    nli_label: NLILabel = NLILabel.NEUTRAL
    nli_score: float = 0.0
    is_inferred: bool = False
    derived_from: str = ""                # gist cid when the memory was surfaced via a dream gist
    contradiction_edges: list[str] = Field(default_factory=list)


class BrainEventType(str, Enum):
    MEMORY_INGESTED = "memory_ingested"
    MEMORY_RECALLED = "memory_recalled"
    ANSWER_VERIFIED = "answer_verified"
    ANSWER_ABSTAINED = "answer_abstained"
    RETRIEVAL_MISSED = "retrieval_missed"
    CONTRADICTION_DETECTED = "contradiction_detected"
    DREAM_GIST_CREATED = "dream_gist_created"
    REPAIR_PROPOSED = "repair_proposed"
    REPAIR_APPLIED = "repair_applied"
    HEALTH_DEBT_DETECTED = "health_debt_detected"
    OPTIMIZER_PROMOTED = "optimizer_promoted"
    OPTIMIZER_REJECTED = "optimizer_rejected"


class BrainEvent(BaseModel):
    """One entry in the single improvement stream. Feedback, repair, health, and optimizer
    loops consume events instead of hidden side effects."""
    type: BrainEventType
    namespace: str = "default"
    at: float = Field(default_factory=now)
    memory_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class QualityGateResult(BaseModel):
    """The verdict of a QualityGate run: every new brain connection must pass before it ships
    by default. `passed` is the AND of all checks that were actually run (None = not checked)."""
    feature: str
    passed: bool = False
    checks: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
