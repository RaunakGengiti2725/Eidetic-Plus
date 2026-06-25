"""Reflex Recall, Component 2: the MemoryPacket contract.

A MemoryPacket is the LOCAL recall result -- what Eidetic remembers about a query before any
model runs. It is produced with no embedding, no NLI, and no answer generation (see
reflex_activation.build_memory_packet), so it is the sub-second "anti-RAG" surface that API/MCP
expose and that the reader path consumes as `precomputed` candidates.

Score contract: retrieval.answer() reads `dense_score` (its coverage gate, which drives cascade
escalation and abstention) and `rerank_score` (its confidence) off the candidates. So
`to_candidates()` populates BOTH from the reflex coverage proxy -- otherwise a confident reflex
hit would feed answer() zero-scored candidates and spuriously abstain.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr

from .models import RetrievalCandidate, Scope


class ReflexScore(BaseModel):
    """A candidate's activation, decomposed so a recall is explainable axis by axis. `aggregate`
    drives ranking; `match_strength` (content coverage in [0,1]) is the coverage proxy that
    becomes the candidate's dense_score/rerank_score."""
    entity: float = 0.0
    lexical: float = 0.0
    temporal: float = 0.0
    coactivation: float = 0.0
    hotset: float = 0.0
    aggregate: float = 0.0
    match_strength: float = 0.0


class PacketCandidate(BaseModel):
    """One ranked memory in a packet, carrying its proof-ready provenance and its score."""
    memory_id: str
    content_hash: str = ""
    raw_uri: str = ""
    snippet: str = ""
    valid_at: float = 0.0
    invalid_at: Optional[float] = None
    score: ReflexScore = Field(default_factory=ReflexScore)
    retrieval_paths: list[str] = Field(default_factory=list)   # which axes surfaced it


class MemoryPacket(BaseModel):
    """The local recall result. `candidates` are kept as full RetrievalCandidate objects (records
    included) for the reader path; `items` is the lightweight, serializable projection."""
    query: str = ""
    scope: Scope = Field(default_factory=Scope)
    as_of: Optional[float] = None
    source: str = "reflex"
    coverage: float = 0.0
    items: list[PacketCandidate] = Field(default_factory=list)
    scores: dict[str, ReflexScore] = Field(default_factory=dict)
    snippets: dict[str, str] = Field(default_factory=dict)
    content_hashes: dict[str, str] = Field(default_factory=dict)
    entity_matches: dict[str, list[str]] = Field(default_factory=dict)   # entity -> memory_ids
    temporal_match_ids: list[str] = Field(default_factory=list)
    coactivation_paths: dict[str, list[str]] = Field(default_factory=dict)  # seed -> linked ids
    supersession_chains: dict[str, list[str]] = Field(default_factory=dict)  # memory_id -> facts
    active_fact_edges: list[dict] = Field(default_factory=list)
    hot_ids: list[str] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)

    # The reader-path candidates (records attached). Private: excluded from serialization (heavy).
    _candidates: list[RetrievalCandidate] = PrivateAttr(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def candidate_ids(self) -> list[str]:
        return [c.memory_id for c in self.items]

    def to_candidates(self) -> list[RetrievalCandidate]:
        return list(self._candidates)

    def public_dict(self) -> dict:
        """API/MCP-facing projection: ids, snippets, hashes, score breakdown, paths, latency --
        never the full record bodies."""
        return {
            "query": self.query,
            "scope": self.scope.model_dump(),
            "as_of": self.as_of,
            "source": self.source,
            "coverage": self.coverage,
            "candidate_ids": self.candidate_ids(),
            "scores": {mid: s.model_dump() for mid, s in self.scores.items()},
            "snippets": dict(self.snippets),
            "content_hashes": dict(self.content_hashes),
            "entity_matches": {k: list(v) for k, v in self.entity_matches.items()},
            "temporal_match_ids": list(self.temporal_match_ids),
            "coactivation_paths": {k: list(v) for k, v in self.coactivation_paths.items()},
            "supersession_chains": {k: list(v) for k, v in self.supersession_chains.items()},
            "active_fact_edges": list(self.active_fact_edges),
            "hot_ids": list(self.hot_ids),
            "latency_ms": dict(self.latency_ms),
        }
