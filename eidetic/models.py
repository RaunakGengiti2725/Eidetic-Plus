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
