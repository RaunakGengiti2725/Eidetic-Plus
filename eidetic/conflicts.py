"""Deterministic current-value conflict resolution.

The LLM extracts every semantically matching candidate. Python chooses the freshest
timestamp. The model is never asked to decide which value is newest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .models import MemoryRecord, RetrievalCandidate

MatchExtractor = Callable[[str, list[dict]], list[dict]]

_CURRENT_HINTS = (
    "current", "currently", "now", "latest", "newest", "right now", "today",
    "still", "where do", "where does", "what is", "what's", "who is", "which is",
)
_NON_CURRENT_HINTS = (
    "previous", "previously", "before", "used to", "earlier", "historical",
    "history", "then", "at the time", "back then", "how many", "count",
    "list", "all ", "every", "activities", "books", "movies", "songs",
)
_YES_NO_RE = re.compile(r"\s*(is|are|was|were|do|does|did|has|have|had|can|could|should|would)\b", re.I)
_CURRENT_VALUE_RE = re.compile(
    r"\s*(where\s+(?:do|does|is|are)|what\s+(?:is|are|'s)|who\s+(?:is|are)|which\s+(?:is|are))\b",
    re.I,
)


@dataclass(frozen=True)
class CurrentValueResolution:
    answer: str
    records: list[MemoryRecord]
    matches: list[dict]
    note: str = "conflict-resolver"


def is_current_value_query(query: str) -> bool:
    """True for questions where the freshest matching value is the intended answer."""
    q = query.lower()
    if _YES_NO_RE.match(query):
        return False
    if any(h in q for h in _NON_CURRENT_HINTS):
        return False
    if any(h in q for h in _CURRENT_HINTS) and _CURRENT_VALUE_RE.match(query):
        return True
    return False


def decompose_current_value_query(query: str) -> list[str]:
    """Small Self-Ask style splitter for independent current-value subquestions."""
    parts = [
        p.strip(" ,;?")
        for p in re.split(r"\?\s+|;\s+|,\s+and\s+(?=(?:what|where|who|which)\b)", query, flags=re.I)
        if p.strip(" ,;?")
    ]
    return parts or [query.strip()]


def _candidate_payload(cands: Iterable[RetrievalCandidate]) -> tuple[list[dict], dict[str, MemoryRecord]]:
    payload: list[dict] = []
    records: dict[str, MemoryRecord] = {}
    for c in cands:
        rec = c.record
        records[rec.memory_id] = rec
        payload.append({
            "memory_id": rec.memory_id,
            "timestamp": rec.valid_at,
            "text": rec.text or rec.summary or "",
            "source": rec.source,
        })
    return payload, records


def _as_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_matches(matches: list[dict], records: dict[str, MemoryRecord]) -> list[dict]:
    out: list[dict] = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("memory_id", ""))
        rec = records.get(mid)
        if rec is None:
            continue
        ts = _as_float(rec.valid_at, 0.0)
        answer = str(m.get("answer") or m.get("quote") or rec.text or rec.summary or "").strip()
        if not answer:
            continue
        out.append({
            "memory_id": mid,
            "timestamp": ts,
            "answer": answer,
            "quote": str(m.get("quote") or "").strip(),
        })
    return out


def resolve_current_value(
    query: str,
    candidates: list[RetrievalCandidate],
    extractor: MatchExtractor,
) -> Optional[CurrentValueResolution]:
    """Resolve one current-value question by extract-all then max(timestamp)."""
    if not is_current_value_query(query) or not candidates:
        return None
    payload, records = _candidate_payload(candidates)
    matches = _normalize_matches(extractor(query, payload), records)
    if not matches:
        return None
    best = max(matches, key=lambda m: (m["timestamp"], m["memory_id"]))
    return CurrentValueResolution(
        answer=best["answer"],
        records=[records[best["memory_id"]]],
        matches=matches,
    )


def resolve_current_value_question(
    query: str,
    candidates: list[RetrievalCandidate],
    extractor: MatchExtractor,
) -> Optional[CurrentValueResolution]:
    """Resolve a current-value question, with a bounded multi-hop decomposition path."""
    if not is_current_value_query(query):
        return None
    hops = decompose_current_value_query(query)
    if len(hops) == 1:
        return resolve_current_value(query, candidates, extractor)

    answers: list[str] = []
    records: list[MemoryRecord] = []
    matches: list[dict] = []
    for hop in hops:
        res = resolve_current_value(hop, candidates, extractor)
        if res is None:
            return None
        answers.append(f"{hop}: {res.answer}")
        records.extend(res.records)
        matches.extend(res.matches)
    unique_records = list({r.memory_id: r for r in records}.values())
    return CurrentValueResolution(
        answer="; ".join(answers),
        records=unique_records,
        matches=matches,
        note="conflict-resolver-multihop",
    )
