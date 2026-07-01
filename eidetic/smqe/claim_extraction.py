"""Source-backed claim extraction helpers for SMQE consolidation."""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from eidetic import preferences
from eidetic.models import ClaimRecord, MemoryRecord


_CLAIM_TYPES = {"quantity", "state", "event", "interval", "table", "preference"}
_NUM_WORD_PATTERN = "eleven|twelve|three|seven|eight|zero|four|five|nine|one|two|six|ten|an|a"
_MONEY_AMOUNT_RE = re.compile(
    r"(?:[$€£]\s*\d+(?:,\d{3})*(?:\.\d+)?|"
    rf"\b(?:\d+(?:,\d{{3}})*(?:\.\d+)?|{_NUM_WORD_PATTERN})\s*(?:dollars?|usd|bucks|€|£)\b)",
    re.I,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _sentences(text: str, *, limit: int = 80) -> list[str]:
    pieces: list[str] = []
    for line in re.split(r"\n+", text or ""):
        line = line.strip()
        if not line:
            continue
        if "|" in line and re.search(r"\|.*\|", line):
            pieces.append(line)
            continue
        if (
            re.search(r"\b(?:suggest(?:ions?)?|recommend(?:ations?)?|options?|ideas?|tips?)\b", line, re.I)
            and re.search(r"\b\d+[.)]\s*[A-Z]", line)
        ):
            pieces.append(line)
            continue
        pieces.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    out = []
    seen = set()
    for piece in pieces:
        clean = re.sub(r"\s+", " ", piece).strip()
        if len(clean) < 8 or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        out.append(clean[:900])
        if len(out) >= limit:
            break
    return out


def _claim_type_for(atom: str) -> str:
    low = atom.lower()
    if "|" in atom and re.search(r"\|.*\|", atom):
        return "table"
    if preferences.is_preference(atom) or re.search(r"\b(?:prefer|favorite|favourite|like|love|hate|avoid)\b", low):
        return "preference"
    if _MONEY_AMOUNT_RE.search(atom) or re.search(
        r"\b\d+(?:\.\d+)?\s*(?:hours?|minutes?|days?|weeks?|months?|years?|kg|lbs?|miles?|%)\b",
        low,
    ):
        return "quantity"
    if re.search(r"\b(?:from|between|until|through)\b.+\b(?:to|and|until|through)\b", low):
        return "interval"
    if re.search(r"\b(?:went|visited|met|bought|attended|finished|started|left|arrived|scheduled|called|emailed)\b", low):
        return "event"
    return "state"


def _subject_for(atom: str, rec: MemoryRecord) -> str:
    m = re.match(r"\s*(user|assistant|system|human|ai)\s*:\s*", atom, re.I)
    if m:
        return m.group(1).lower()
    m = re.match(r"\s*([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\b", atom)
    if m:
        return m.group(1)
    return rec.source or "memory"


def _predicate_for(atom: str) -> str:
    low = atom.lower()
    for pat in (
        r"\b(?:is|was|are|were|am)\s+([a-z][a-z0-9_-]+)",
        r"\b(?:prefer|prefers|preferred|favorite|favourite|like|likes|liked|love|loves|avoid|avoids)\b",
        r"\b(?:went|visited|met|bought|attended|finished|started|left|arrived|scheduled|called|emailed)\b",
    ):
        m = re.search(pat, low)
        if m:
            return m.group(0).strip()
    terms = re.findall(r"[a-z0-9][a-z0-9_-]*", low)
    return " ".join(terms[:4])


def _object_for(atom: str) -> str:
    text = re.sub(r"^\s*(?:user|assistant|system|human|ai)\s*:\s*", "", atom, flags=re.I)
    m = re.search(r"\b(?:is|was|are|were|am|prefer|prefers|preferred|like|likes|liked|love|loves|visited|bought|attended|at|in|to|from)\s+([^.;!?]+)", text, re.I)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:180]
    return text[:180]


def _source_cutoff(rec: MemoryRecord) -> Optional[float]:
    cutoffs = [v for v in (rec.invalid_at, rec.expired_at) if v is not None]
    return min(cutoffs) if cutoffs else None


def _claim_from_atom(rec: MemoryRecord, atom: str, claim_type: Optional[str] = None,
                     *, predicate: str = "", value: Any = None) -> Optional[ClaimRecord]:
    atom = re.sub(r"\s+", " ", atom or "").strip()
    if not atom:
        return None
    # Keep claims source-backed. If an extractor emits a normalized proof atom, allow a lenient
    # whitespace-insensitive containment check.
    if _norm(atom) not in _norm(rec.text or ""):
        return None
    ctype = claim_type or _claim_type_for(atom)
    if ctype not in _CLAIM_TYPES:
        return None
    return ClaimRecord(
        claim_type=ctype, scope=rec.scope, subject=_subject_for(atom, rec),
        predicate=predicate or _predicate_for(atom), object=_object_for(atom),
        value=atom if value is None else value, valid_at=rec.valid_at,
        invalid_at=_source_cutoff(rec), source_memory_id=rec.memory_id, proof_atom=atom,
        confidence=1.0,
    )


def claims_from_triples(rec: MemoryRecord, triples: Iterable[dict[str, Any]]) -> list[ClaimRecord]:
    out: list[ClaimRecord] = []
    for t in triples or []:
        fact = str(t.get("fact") or "").strip()
        if not fact:
            src = str(t.get("src") or "").strip()
            rel = str(t.get("relation") or "").strip()
            dst = str(t.get("dst") or "").strip()
            fact = " ".join(v for v in (src, rel, dst) if v)
        claim = _claim_from_atom(rec, fact, predicate=str(t.get("relation") or ""))
        if claim is not None:
            claim.subject = str(t.get("src") or claim.subject)
            claim.object = str(t.get("dst") or claim.object)
            out.append(claim)
    return out


def validate_extracted_claims(rec: MemoryRecord, raw_claims: Iterable[dict[str, Any]]) -> list[ClaimRecord]:
    out: list[ClaimRecord] = []
    for raw in raw_claims or []:
        if not isinstance(raw, dict):
            continue
        atom = str(raw.get("proof_atom") or raw.get("fact") or raw.get("text") or "").strip()
        ctype = str(raw.get("claim_type") or raw.get("type") or "").strip().lower()
        claim = _claim_from_atom(
            rec,
            atom,
            ctype if ctype in _CLAIM_TYPES else None,
            predicate=str(raw.get("predicate") or ""),
            value=raw.get("value"),
        )
        if claim is None:
            continue
        claim.subject = str(raw.get("subject") or claim.subject)
        claim.object = str(raw.get("object") or claim.object)
        claim.unit = str(raw.get("unit") or "")
        filters = raw.get("filters")
        if isinstance(filters, dict):
            claim.filters = filters
        out.append(claim)
    return out


def heuristic_claims_from_text(rec: MemoryRecord) -> list[ClaimRecord]:
    claims = []
    for atom in _sentences(rec.text or ""):
        claim = _claim_from_atom(rec, atom)
        if claim is not None:
            claims.append(claim)
    return claims


def claims_for_record(
    rec: MemoryRecord,
    triples: Optional[Iterable[dict[str, Any]]] = None,
    extracted_claims: Optional[Iterable[dict[str, Any]]] = None,
) -> list[ClaimRecord]:
    claims = []
    claims.extend(claims_from_triples(rec, triples or []))
    claims.extend(validate_extracted_claims(rec, extracted_claims or []))
    claims.extend(heuristic_claims_from_text(rec))
    deduped: list[ClaimRecord] = []
    seen = set()
    for claim in claims:
        key = (claim.claim_type, _norm(claim.proof_atom), claim.source_memory_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped
