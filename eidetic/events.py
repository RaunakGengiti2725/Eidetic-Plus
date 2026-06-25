"""Chronos-style structured event calendar (the biggest lever for temporal + multi-session).

Two halves:
  * DETERMINISTIC (this file, no LLM, offline-testable): normalize REFERENCE-RELATIVE date
    expressions to explicit ISO-8601 RANGES ("yesterday", "3 days ago", "last week",
    "last May", "May 2023", "next Tuesday") against a turn/question timestamp; parse a
    query into {operation (filter/count/order), temporal ranges, entities, query type};
    select calendar events by INTERVAL OVERLAP + entity match.
  * LLM (wired in engine.consolidate_pending): extract subject-verb-object event tuples and
    paraphrase aliases. (Event-to-event chaining like "the week after X" is NOT handled in
    the deterministic path -- it needs the LLM or is genuinely hard; we don't promise it here.)

Crucial neutrality rule: the event calendar SELECTS and STRUCTURES events into context. The
SHARED reader still produces the final answer string (the count, the ordering, the fact) --
counting/ordering is never computed in code, so Eidetic gets no answerer advantage.
"""
from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from .models import new_id

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_WEEKDAYS = {d.lower(): i for i, d in enumerate(
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _day_range(d: datetime) -> tuple[str, str]:
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return _iso(start), _iso(start + timedelta(days=1) - timedelta(seconds=1))


def _month_range(year: int, month: int) -> tuple[str, str]:
    last = calendar.monthrange(year, month)[1]
    return (_iso(datetime(year, month, 1)),
            _iso(datetime(year, month, last, 23, 59, 59)))


def _year_range(year: int) -> tuple[str, str]:
    return _iso(datetime(year, 1, 1)), _iso(datetime(year, 12, 31, 23, 59, 59))


def _week_range(any_day: datetime) -> tuple[str, str]:
    monday = (any_day - timedelta(days=any_day.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return _iso(monday), _iso(monday + timedelta(days=7) - timedelta(seconds=1))


def normalize_dates(text: str, reference_time: Optional[float] = None) -> list[dict]:
    """Resolve reference-relative date expressions in `text` to ISO-8601 ranges.

    Returns [{'expr', 'start', 'end'}]. Ranges, not points; reference is `reference_time`
    (epoch) or now. Only reference-relative + absolute forms -- no event-to-event chaining."""
    ref = datetime.fromtimestamp(reference_time) if reference_time else datetime.now()
    low = text.lower()
    out: list[dict] = []

    def add(expr, rng):
        out.append({"expr": expr, "start": rng[0], "end": rng[1]})

    # Absolute ISO datetime / date.
    for m in re.finditer(r"\b(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b", text):
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
            add(m.group(0), _day_range(d))
        except ValueError:
            pass

    # "<Month> <Year>" and "<Month>" (relative to ref year) and "last <Month>".
    for m in re.finditer(r"\b(last\s+)?([A-Za-z]{3,9})\s+(\d{4})\b", text):
        mo = _MONTHS.get(m.group(2).lower())
        if mo:
            add(m.group(0), _month_range(int(m.group(3)), mo))
    for m in re.finditer(r"\b(last|this|next)\s+([A-Za-z]{3,9})\b", low):
        mo = _MONTHS.get(m.group(2))
        if mo:
            year = ref.year
            if m.group(1) == "last" and mo >= ref.month:
                year -= 1
            elif m.group(1) == "next" and mo <= ref.month:
                year += 1
            add(m.group(0), _month_range(year, mo))

    # Standalone year.
    for m in re.finditer(r"(?<![-\d])\b(19|20)\d{2}\b(?!-)", text):
        add(m.group(0), _year_range(int(m.group(0))))

    # today / yesterday / tomorrow.
    if "yesterday" in low:
        add("yesterday", _day_range(ref - timedelta(days=1)))
    if "tomorrow" in low:
        add("tomorrow", _day_range(ref + timedelta(days=1)))
    if re.search(r"\btoday\b", low):
        add("today", _day_range(ref))

    # "N day(s)/week(s)/month(s)/year(s) ago".
    for m in re.finditer(r"\b(\d+)\s+(day|week|month|year)s?\s+ago\b", low):
        n, unit = int(m.group(1)), m.group(2)
        target = ref - timedelta(days=n * _UNIT_DAYS[unit])
        rng = _day_range(target) if unit in ("day",) else (
            _week_range(target) if unit == "week" else (
                _month_range(target.year, target.month) if unit == "month" else _year_range(target.year)))
        add(m.group(0), rng)

    # last/this/next week|month|year.
    for m in re.finditer(r"\b(last|this|next)\s+(week|month|year)\b", low):
        rel, unit = m.group(1), m.group(2)
        delta = {"last": -1, "this": 0, "next": 1}[rel]
        if unit == "week":
            add(m.group(0), _week_range(ref + timedelta(weeks=delta)))
        elif unit == "month":
            y, mo = ref.year, ref.month + delta
            y, mo = (y + (mo - 1) // 12, (mo - 1) % 12 + 1)
            add(m.group(0), _month_range(y, mo))
        else:
            add(m.group(0), _year_range(ref.year + delta))

    # last/next <weekday>.
    for m in re.finditer(r"\b(last|next|this)\s+(" + "|".join(_WEEKDAYS) + r")\b", low):
        rel, wd = m.group(1), _WEEKDAYS[m.group(2)]
        diff = (wd - ref.weekday())
        if rel == "last":
            diff -= 7 if diff >= 0 else 0
        elif rel == "next":
            diff += 7 if diff <= 0 else 0
        add(m.group(0), _day_range(ref + timedelta(days=diff)))

    return out


_COUNT_KW = ("how many", "how often", "number of times", "count", "how much")
_ORDER_KW = ("first", "last", "earliest", "latest", "order", "before", "after", "when did", "what time")
_MULTIHOP_KW = (" and ", " both ", " after ", " before ", " then ", "related to", "connection")
_ID_RE = re.compile(r"\b([A-Z]{2,}-?\d{2,}|\d{4,}|#\w+)\b")
# Sentence-initial question words / determiners to exclude from entity extraction.
_QWORDS = {"how", "what", "when", "where", "why", "who", "which", "whom", "whose",
           "did", "does", "do", "is", "are", "was", "were", "the", "a", "an", "in",
           "on", "at", "to", "of", "and", "or", "i", "my", "you", "your", "he", "she"}


def parse_query(question: str, reference_time: Optional[float] = None) -> dict:
    """Parse a question into {operation, ranges, entities, is_namey, is_multihop}.
    Used to drive the structured calendar filter and query-adaptive RRF weights."""
    low = question.lower()
    if any(k in low for k in _COUNT_KW):
        operation = "count"
    elif any(k in low for k in _ORDER_KW):
        operation = "order"
    else:
        operation = "filter"
    ranges = normalize_dates(question, reference_time)
    # Entities: proper-noun-ish capitalized tokens + IDs/quoted spans (deterministic, no LLM).
    entities = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", question)
    entities += [m.group(0) for m in _ID_RE.finditer(question)]
    entities += re.findall(r'"([^"]+)"', question)
    entities = [e.strip() for e in entities if e.strip() and e.lower() not in _QWORDS]
    entities = list(dict.fromkeys(entities))
    is_namey = bool(_ID_RE.search(question)) or bool(ranges) or len(entities) >= 1
    is_multihop = any(k in low for k in _MULTIHOP_KW) or len(entities) >= 3
    return {"operation": operation, "ranges": ranges, "entities": entities,
            "is_namey": is_namey, "is_multihop": is_multihop}


class EventRecord(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    subject: str = ""
    verb: str = ""
    object: str = ""
    fact: str = ""
    aliases: list[str] = Field(default_factory=list)
    start: Optional[float] = None     # epoch seconds (range start)
    end: Optional[float] = None       # epoch seconds (range end)
    source_memory_id: str = ""
    namespace: str = "default"
    valid_at: float = 0.0

    def as_text(self) -> str:
        when = ""
        if self.start is not None:
            when = f" [{datetime.fromtimestamp(self.start).date()}"
            when += f"..{datetime.fromtimestamp(self.end).date()}]" if self.end else "]"
        return f"{self.fact or f'{self.subject} {self.verb} {self.object}'}{when}".strip()


def _overlaps(ev: EventRecord, start_e: float, end_e: float) -> bool:
    if ev.start is None:
        return False
    e_end = ev.end if ev.end is not None else ev.start
    return ev.start <= end_e and e_end >= start_e   # interval overlap, not equality


def select_for_query(events: list[EventRecord], parsed: dict,
                     reference_time: Optional[float] = None) -> list[EventRecord]:
    """Select calendar events matching the query's temporal ranges + entities (interval
    overlap). Returns events to place in context -- the reader still answers/counts."""
    ents = [e.lower() for e in parsed.get("entities", [])]
    ranges = parsed.get("ranges", [])
    range_epochs = []
    for r in ranges:
        try:
            range_epochs.append((datetime.strptime(r["start"], "%Y-%m-%dT%H:%M:%S").timestamp(),
                                 datetime.strptime(r["end"], "%Y-%m-%dT%H:%M:%S").timestamp()))
        except (ValueError, KeyError):
            continue

    def match_count(ev: EventRecord) -> int:
        if not ents:
            return 0
        hay = " ".join([ev.subject, ev.object, ev.fact] + ev.aliases).lower()
        return sum(1 for e in ents if e in hay)

    scored: list[tuple[int, EventRecord]] = []
    for ev in events:
        m = match_count(ev)
        if ents and m == 0:               # must match at least one query entity
            continue
        if range_epochs and not any(_overlaps(ev, s, e) for s, e in range_epochs):
            continue
        scored.append((m, ev))
    # Most-specific first (matches the MOST query entities), then chronological. This stops a
    # broad entity like "Caroline" from drowning the specific "LGBTQ support group" event.
    scored.sort(key=lambda t: (-t[0], t[1].start if t[1].start is not None else 0.0))
    return [ev for _, ev in scored]


def event_chain(events: list[EventRecord], parsed: dict,
                reference_time: Optional[float] = None, *, window: int = 6) -> list[EventRecord]:
    """Strengthened temporal indexing (Phase 5): for sequence / before-after / 'what changed after
    X' / order questions, return the matched events in STRICT CHRONOLOGICAL order (the chain), so
    the reader sees the progression. Deterministic -- selection + ordering only; the SHARED reader
    still produces the answer (no temporal advantage in code).

    Falls back to all time-stamped in-scope events (chronological) when the entity/range filter
    matches nothing but the query is clearly chronological -- a bare 'what happened, in order?'."""
    matched = select_for_query(events, parsed, reference_time)
    if not matched:
        matched = [e for e in events if e.start is not None]
    chrono = sorted((e for e in matched if e.start is not None),
                    key=lambda e: (e.start, e.end if e.end is not None else e.start))
    return chrono[:window]
