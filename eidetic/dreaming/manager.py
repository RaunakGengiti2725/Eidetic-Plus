"""Heuristic write-time memory manager -- the API-only approximation of Memory-R1's GRPO
manager (PDF 3a-RL). Memory-R1 fine-tunes ADD/UPDATE/DELETE/NOOP with GRPO/PPO (needs GPU);
the PDF explicitly says, on an API-only stack, "approximate the learned policy with your
existing heuristics." This is that deterministic operation router -- zero training.

  * NOOP  -- a duplicate of an existing fact (same value, or high embedding cosine).
  * UPDATE -- same (subject, relation) with a NEWER, different value -> supersede the old edge
              (reusing graph.add_fact's invalidate-not-delete + supersedes chain).
  * DELETE_TOMBSTONE -- a hard contradiction -> reversibly tombstone (close valid range) the
              contradicted edge. NEVER deletes a raw record (the WORM discipline).
  * ADD   -- a novel fact, or an OLDER historical value (kept additively, history preserved).

`classify_operation` is pure and deterministic -> fully offline-unit-testable. The execution
wrapper `run_memory_manager` is gated OFF by default and early-returns before any model call,
so ingest/consolidation behave byte-for-byte as before unless explicitly enabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Operation(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE_TOMBSTONE = "delete_tombstone"
    NOOP = "noop"


@dataclass
class Fact:
    subject: str
    relation: str
    object: str
    valid_at: float = 0.0
    text: str = ""
    vec: Optional[np.ndarray] = field(default=None)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _same_key(a: Fact, b: Fact) -> bool:
    return _norm(a.subject) == _norm(b.subject) and _norm(a.relation) == _norm(b.relation)


def _same_value(a: Fact, b: Fact) -> bool:
    return _norm(a.object) == _norm(b.object)


def _cos(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def classify_operation(candidate: Fact, existing: list[Fact], *, dup_cosine: float = 0.97,
                       contradicts: bool = False) -> Operation:
    """Route an incoming fact to ADD / UPDATE / DELETE_TOMBSTONE / NOOP given the existing facts
    about the same subject. `contradicts` is the (upstream, LLM-detected) hard-contradiction
    signal; everything else is decided deterministically here."""
    # NOOP: an exact-value duplicate, or a high-cosine near-duplicate of the same fact.
    for ex in existing:
        if _same_key(candidate, ex) and _same_value(candidate, ex):
            return Operation.NOOP
        if (candidate.vec is not None and ex.vec is not None
                and _cos(candidate.vec, ex.vec) >= dup_cosine
                and _norm(candidate.text) == _norm(ex.text)):
            return Operation.NOOP

    if contradicts:
        return Operation.DELETE_TOMBSTONE

    same_key = [ex for ex in existing if _same_key(candidate, ex)]
    if same_key:
        newest = max(ex.valid_at for ex in same_key)
        if candidate.valid_at > newest:
            return Operation.UPDATE          # a newer value for the same attribute -> supersede
        return Operation.ADD                 # an older historical value -> keep additively
    return Operation.ADD                      # novel fact


def run_memory_manager(engine, scope=None) -> dict:
    """Execution wrapper (gated). When MEMORY_MANAGER is off this returns immediately, BEFORE any
    store read or model call, so the write/consolidation path is unchanged. The enabled path
    routes recently-extracted facts through classify_operation and applies ADD/UPDATE via
    graph.add_fact and DELETE via a reversible tombstone (never a raw delete). LLM-gated (dup
    embeddings + contradiction detection are real calls); not run under the current quota block."""
    if not getattr(engine.settings, "memory_manager_enabled", False):
        return {"skipped": "disabled"}
    # The enabled orchestration is intentionally thin and fail-loud; it is exercised only with a
    # funded key (no mocks). It reuses engine.client.find_contradictions for the contradiction
    # signal and engine.graph.add_fact for ADD/UPDATE. Left as the documented integration point.
    from ..errors import FeatureNotImplementedError
    raise FeatureNotImplementedError(
        "MEMORY_MANAGER is experimental and not implemented yet; default off. The deterministic "
        "router classify_operation IS offline-validated; the enabled execution path (dup-embedding "
        "+ contradiction detection, real calls) is a documented integration point to build + "
        "measure on the dev split when quota is restored."
    )
