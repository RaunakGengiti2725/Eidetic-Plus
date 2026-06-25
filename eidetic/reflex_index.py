"""Reflex Recall, Component 1: the derived inverted lookup.

The ONE thing that is a full-corpus scan to recompute -- the entity/term -> memory_ids map --
is cached here, partitioned by `namespace` (the hard isolation boundary). Everything else a
reflex packet needs (co-activation neighbors, supersession chains, active fact edges) is read
LIVE from the graph/store for the handful of seeds, because those are already indexed SQLite
lookups and every extra incrementally-maintained surface is one more place to fall out of sync.

This index is a REBUILDABLE CACHE, never a source of truth. The activation burst loads every
seed back through the store, which re-applies the bi-temporal validity filter and the finer
agent/project scope filter. So a stale index can only LOSE a candidate (lower coverage -> safe
fallback to full retrieval); it can never surface an invalid-at-`as_of` or cross-scope record.
"""
from __future__ import annotations

import re
import threading
from typing import Optional

from .events import _QWORDS
from .models import MemoryRecord

# Additional high-frequency, low-signal terms beyond the question words in events._QWORDS.
_STOPWORDS = _QWORDS | {
    "with", "for", "from", "this", "that", "have", "has", "had", "not", "but", "all",
    "any", "can", "will", "would", "should", "could", "about", "into", "than", "then",
    "they", "them", "their", "there", "were", "been", "being", "its", "his", "her",
    "our", "out", "who", "whom", "via", "per",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Deterministic lexical tokens: lowercase alphanumeric runs, drop stopwords and runs
    shorter than 3 chars (they carry little retrieval signal and bloat the index)."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        if len(tok) >= 3 and tok not in _STOPWORDS:
            out.append(tok)
    return out


def _norm_entity(name: str) -> str:
    return (name or "").strip().lower()


class ReflexIndex:
    """Namespace -> {term|entity -> set(memory_id)}. Updated under the engine write lock on the
    same mutations that touch the vector index/store/graph, and fully rebuildable from the store."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._built = False
        self._built_count = -1            # store record count at last full rebuild (staleness probe)
        self._terms: dict[str, dict[str, set[str]]] = {}
        self._entities: dict[str, dict[str, set[str]]] = {}

    @property
    def built(self) -> bool:
        return self._built

    @property
    def built_count(self) -> int:
        return self._built_count

    def add_record(self, rec: MemoryRecord) -> None:
        with self._lock:
            self._add_locked(rec)

    def rebuild_from_store(self, store) -> int:
        """Full rebuild from the source of truth. Returns the record count indexed."""
        with self._lock:
            self._terms = {}
            self._entities = {}
            records = store.all_records(None)
            for rec in records:
                self._add_locked(rec)
            self._built = True
            self._built_count = len(records)
            return len(records)

    def ensure_built(self, store) -> None:
        with self._lock:
            if not self._built:
                self.rebuild_from_store(store)

    def seeds(self, namespace: str, entities: list[str], terms: list[str]) -> set[str]:
        """Candidate memory_ids in `namespace` matching ANY query entity OR term. Union, not
        intersection: recall over precision -- the activation burst and the NLI gate downstream
        prune; the index's job is to not MISS a relevant memory."""
        out: set[str] = set()
        with self._lock:
            term_map = self._terms.get(namespace, {})
            for t in terms:
                ids = term_map.get(t)
                if ids:
                    out |= ids
            ent_map = self._entities.get(namespace, {})
            for e in entities:
                ids = ent_map.get(_norm_entity(e))
                if ids:
                    out |= ids
        return out

    def stats(self) -> dict:
        with self._lock:
            return {
                "built": self._built,
                "namespaces": sorted(self._terms.keys() | self._entities.keys()),
                "terms": {ns: len(m) for ns, m in self._terms.items()},
                "entities": {ns: len(m) for ns, m in self._entities.items()},
            }

    # ---- internals --------------------------------------------------------
    def _add_locked(self, rec: MemoryRecord) -> None:
        ns = rec.scope.namespace
        terms = self._terms.setdefault(ns, {})
        for tok in set(tokenize(rec.text)):
            terms.setdefault(tok, set()).add(rec.memory_id)
        ents = self._entities.setdefault(ns, {})
        for ent in rec.entities:
            key = _norm_entity(ent)
            if key:
                ents.setdefault(key, set()).add(rec.memory_id)
