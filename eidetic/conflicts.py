"""Deterministic current-value conflict resolution.

The LLM extracts every semantically matching candidate. Python chooses the freshest
timestamp. The model is never asked to decide which value is newest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .models import MemoryRecord, RetrievalCandidate, now

MatchExtractor = Callable[[str, list[dict]], list[dict]]

_CURRENT_HINTS = (
    "current", "currently", "now", "latest", "newest", "right now", "today",
    "still", "where do", "where does", "what is", "what's", "who is", "which is",
)
_NON_CURRENT_HINTS = (
    "previous", "previously", "before", "used to", "earlier", "historical",
    "history", "then", "at the time", "back then", "how many", "count",
    "list", "all ", "every", "activity", "activities", "books", "movies", "songs",
    "would", "wouldn't", "could", "couldn't", "might", "probably", "likely",
    "should", "recommend", "suggest", "good fit", "enjoy",
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
    abstained: bool = False
    superseded: list[str] = field(default_factory=list)   # memory_ids of older/closed candidates
    as_of: Optional[float] = None


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
    """Carry each candidate's BI-TEMPORAL coordinates from the (immutable) record -- not the LLM:
    valid_at (world-valid time), invalid_at (when it stopped being true), and created_at as the
    deterministic serial tiebreak. A missing valid_at FAILS LOUD (the resolver cannot order facts
    in time without it) rather than silently defaulting to 0."""
    out: list[dict] = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("memory_id", ""))
        rec = records.get(mid)
        if rec is None:
            continue
        if rec.valid_at is None:
            raise ValueError(
                f"conflict resolution requires a valid_at timestamp; memory {mid} has none")
        answer = str(m.get("answer") or m.get("quote") or rec.text or rec.summary or "").strip()
        if not answer:
            continue
        out.append({
            "memory_id": mid,
            "valid_at": float(rec.valid_at),
            "invalid_at": (float(rec.invalid_at) if rec.invalid_at is not None else None),
            "serial": _as_float(rec.created_at, float(rec.valid_at)),
            "answer": answer,
            "quote": str(m.get("quote") or "").strip(),
        })
    return out


def resolve_current_value(
    query: str,
    candidates: list[RetrievalCandidate],
    extractor: MatchExtractor,
    as_of: Optional[float] = None,
) -> Optional[CurrentValueResolution]:
    """Resolve one current-value question DETERMINISTICALLY. The LLM (extractor) only extracts
    semantically matching candidates; Python decides freshness:

        valid  = [c for c in matches if c.valid_at <= as_of
                                    and (c.invalid_at is None or as_of < c.invalid_at)]
        answer = argmax(valid, key=(valid_at, serial, memory_id))     # latest-valid wins
        if not valid: abstain("no fact valid as of the requested time")

    The model never compares timestamps. `as_of` defaults to now() (the 'current value' question);
    pass an explicit time for before/after/as-of time-travel."""
    if not is_current_value_query(query) or not candidates:
        return None
    as_of = now() if as_of is None else as_of
    payload, records = _candidate_payload(candidates)
    matches = _normalize_matches(extractor(query, payload), records)
    if not matches:
        return None
    valid = [m for m in matches
             if m["valid_at"] <= as_of and (m["invalid_at"] is None or as_of < m["invalid_at"])]
    if not valid:
        return CurrentValueResolution(
            answer="", records=[], matches=matches, abstained=True, as_of=as_of,
            note=f"no fact valid as of the requested time ({as_of:.0f})")
    best = max(valid, key=lambda m: (m["valid_at"], m["serial"], m["memory_id"]))
    superseded = [m["memory_id"] for m in matches if m["memory_id"] != best["memory_id"]]
    return CurrentValueResolution(
        answer=best["answer"],
        records=[records[best["memory_id"]]],
        matches=matches,
        superseded=superseded,
        as_of=as_of,
    )


def resolve_current_value_question(
    query: str,
    candidates: list[RetrievalCandidate],
    extractor: MatchExtractor,
    as_of: Optional[float] = None,
) -> Optional[CurrentValueResolution]:
    """Resolve a current-value question, with a bounded multi-hop decomposition path. `as_of`
    threads the bi-temporal time-travel point through every hop."""
    if not is_current_value_query(query):
        return None
    hops = decompose_current_value_query(query)
    if len(hops) == 1:
        return resolve_current_value(query, candidates, extractor, as_of)

    answers: list[str] = []
    records: list[MemoryRecord] = []
    matches: list[dict] = []
    superseded: list[str] = []
    for hop in hops:
        res = resolve_current_value(hop, candidates, extractor, as_of)
        if res is None or res.abstained:
            return res        # propagate an abstention rather than a half-answered multi-hop
        answers.append(f"{hop}: {res.answer}")
        records.extend(res.records)
        matches.extend(res.matches)
        superseded.extend(res.superseded)
    unique_records = list({r.memory_id: r for r in records}.values())
    return CurrentValueResolution(
        answer="; ".join(answers),
        records=unique_records,
        matches=matches,
        superseded=superseded,
        as_of=as_of,
        note="conflict-resolver-multihop",
    )
