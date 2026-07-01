"""Chronos-style structured event calendar (the biggest lever for temporal + multi-session).

Two halves:
  * DETERMINISTIC (this file, no LLM, offline-testable): normalize REFERENCE-RELATIVE date
    expressions to explicit ISO-8601 RANGES ("yesterday", "3 days ago", "last week",
    "last May", "May 2023", "next Tuesday") against a turn/question timestamp; parse a
    query into {operation (filter/count/order), temporal ranges, entities, query type};
    select calendar events by INTERVAL OVERLAP + entity match.
  * LLM (wired in engine.consolidate_pending): extract subject-verb-object event tuples and
    paraphrase aliases. The deterministic path handles common fuzzy windows and can resolve
    simple event-relative phrases when callers pass the in-scope event list.

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
_COUNT_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "couple": 2,
    "few": 3,
    "several": 4,
}
_ANCHOR_STOPWORDS = {
    "the", "and", "for", "from", "with", "after", "before", "following", "previous",
    "prior", "next", "day", "week", "month", "event", "thing", "that", "this", "then",
}
_LEADING_ALIAS_STOPWORDS = {"the", "a", "an", "my", "your", "his", "her", "their", "our"}
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.I)


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


def _range(start: datetime, end: datetime) -> tuple[str, str]:
    return _iso(start.replace(microsecond=0)), _iso(end.replace(microsecond=0))


def _qualified_month_range(year: int, month: int, qual: str) -> tuple[str, str]:
    last = calendar.monthrange(year, month)[1]
    if qual == "early":
        start, end = datetime(year, month, 1), datetime(year, month, min(10, last), 23, 59, 59)
    elif qual == "mid":
        start, end = datetime(year, month, min(11, last)), datetime(year, month, min(20, last), 23, 59, 59)
    else:  # late
        start, end = datetime(year, month, min(21, last)), datetime(year, month, last, 23, 59, 59)
    return _range(start, end)


def _event_hay(ev) -> str:
    return " ".join([
        getattr(ev, "subject", "") or "",
        getattr(ev, "verb", "") or "",
        getattr(ev, "object", "") or "",
        getattr(ev, "fact", "") or "",
        " ".join(getattr(ev, "aliases", []) or []),
    ]).lower()


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _match_anchor_events(anchor: str, events: Optional[list]) -> list:
    if not anchor or not events:
        return []
    anchor_norm = " ".join(re.findall(r"[a-z0-9]+", anchor.lower()))
    terms = _tokens(anchor) - _ANCHOR_STOPWORDS
    if not terms:
        return []
    scored = []
    for ev in events:
        if getattr(ev, "start", None) is None:
            continue
        hay = _event_hay(ev)
        hay_tokens = _tokens(hay)
        hits = len(terms & hay_tokens)
        phrase_hit = bool(anchor_norm and anchor_norm in hay)
        if hits:
            scored.append((hits + (2 if phrase_hit else 0), float(ev.start), ev))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [ev for _, _, ev in scored]


def _count_word(value: str | None) -> int:
    if not value:
        return 1
    v = value.strip().lower()
    if v.isdigit():
        return max(1, int(v))
    return _COUNT_WORDS.get(v, 1)


def _event_relative_range(base: datetime, unit: str, rel: str, n: int = 1) -> tuple[str, str]:
    delta = timedelta(days=max(1, n) * _UNIT_DAYS[unit])
    target = base + (delta if rel == "after" else -delta)
    if unit == "day":
        return _day_range(target)
    if unit == "week":
        return _week_range(target)
    return _month_range(target.year, target.month)


def _absolute_date_anchor(anchor: str, ref: datetime) -> Optional[datetime]:
    s = (anchor or "").strip().rstrip("?.!,;")
    if not s:
        return None
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?", s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            return None
    m = re.fullmatch(r"([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?(?:\s+(\d{4}))?", s)
    if m:
        mo = _MONTHS.get(m.group(1).lower())
        if mo:
            try:
                return datetime(int(m.group(3)) if m.group(3) else ref.year, mo, int(m.group(2)))
            except ValueError:
                return None
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})(?:\s+(\d{4}))?", s)
    if m:
        mo = _MONTHS.get(m.group(2).lower())
        if mo:
            try:
                return datetime(int(m.group(3)) if m.group(3) else ref.year, mo, int(m.group(1)))
            except ValueError:
                return None
    return None


def _date_anchor_pattern() -> str:
    month_names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    return (
        rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|"
        rf"(?:{month_names})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?(?:\s+\d{{4}})?|"
        rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{month_names})(?:\s+\d{{4}})?)"
    )


def _range_seconds(item: dict) -> Optional[tuple[float, float]]:
    try:
        return (
            datetime.strptime(item["start"], "%Y-%m-%dT%H:%M:%S").timestamp(),
            datetime.strptime(item["end"], "%Y-%m-%dT%H:%M:%S").timestamp(),
        )
    except (KeyError, TypeError, ValueError):
        return None


def effective_date_ranges(ranges: list[dict]) -> list[dict]:
    """Return the ranges that should actually constrain event/source selection.

    Date normalization intentionally records every expression it sees for transparency, which means
    "the week before August 3, 2023" contains the anchor day and year as well as the intended
    relative week. Selection should use the intended narrow range: prefer event-relative/interval
    ranges, otherwise drop broad ranges that contain a more specific range.
    """
    valid = [r for r in (ranges or []) if _range_seconds(r) is not None]
    if not valid:
        return []
    preferred = [r for r in valid if r.get("anchored") or r.get("interval")]
    if preferred:
        return preferred
    seconds = [(r, _range_seconds(r)) for r in valid]
    out: list[dict] = []
    for item, rng in seconds:
        if rng is None:
            continue
        lo, hi = rng
        contains_more_specific = False
        for other, other_rng in seconds:
            if other is item or other_rng is None:
                continue
            olo, ohi = other_rng
            if lo <= olo and hi >= ohi and (lo < olo or hi > ohi):
                contains_more_specific = True
                break
        if not contains_more_specific:
            out.append(item)
    return out or valid


def _clean_alias(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip(" \t\r\n\"'`.,;:!?()[]{}")).strip()


def _add_alias(out: list[str], seen: set[str], value: str) -> None:
    alias = _clean_alias(value)
    if not alias or len(alias) > 140:
        return
    key = alias.lower()
    if key in seen:
        return
    seen.add(key)
    out.append(alias)


def _strip_leading_alias_stopwords(alias: str) -> str:
    parts = alias.split()
    while parts and parts[0].lower() in _LEADING_ALIAS_STOPWORDS:
        parts = parts[1:]
    return " ".join(parts)


def _contiguous_token_indexes(words: list[str], needle: list[str]) -> list[int]:
    if not needle or len(needle) > len(words):
        return []
    width = len(needle)
    return [i for i in range(len(words) - width + 1) if words[i:i + width] == needle]


def _ordered_alias_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "") if len(m.group(0)) > 2]


def event_aliases_from_text(text: str, triple: dict[str, str], *, max_aliases: int = 12) -> list[str]:
    """Build conservative source-derived aliases for an extracted event.

    Extraction often canonicalizes a rich mention ("annual robotics conference") into a generic
    object ("conference"). For event-relative queries, that can make two dated events tie on the
    same anchor token and recency can pick the wrong one. These aliases never add new facts: they
    only preserve nearby source wording around the extracted subject/relation/object.
    """
    src = str(triple.get("src", "") or "").strip()
    rel = str(triple.get("relation", "") or "").strip()
    dst = str(triple.get("dst", "") or "").strip()
    fact = str(triple.get("fact", "") or f"{src} {rel} {dst}").strip()
    out: list[str] = []
    seen: set[str] = set()

    for alias in (fact, f"{src} {dst}", f"{rel} {dst}", f"{src} {rel} {dst}", dst):
        _add_alias(out, seen, alias)

    object_terms = _ordered_alias_tokens(dst)
    relation_terms = _ordered_alias_tokens(rel)
    if not text or not object_terms:
        return out[:max_aliases]

    for quoted in re.findall(r"\"([^\"]{2,120})\"", text):
        if _tokens(quoted) & set(object_terms):
            _add_alias(out, seen, quoted)

    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        pieces = [(m.group(0), m.group(0).lower()) for m in _WORD_RE.finditer(sentence)]
        if not pieces:
            continue
        words = [p[1] for p in pieces]
        raw = [p[0] for p in pieces]
        object_indexes = _contiguous_token_indexes(words, object_terms)
        if not object_indexes:
            continue
        for idx in object_indexes:
            obj_end = idx + len(object_terms)
            for before in (2, 3, 4):
                start = max(0, idx - before)
                alias = " ".join(raw[start:obj_end])
                _add_alias(out, seen, alias)
                stripped = _strip_leading_alias_stopwords(alias)
                if stripped != alias:
                    _add_alias(out, seen, stripped)
            if relation_terms:
                for rel_idx in _contiguous_token_indexes(words[:idx], relation_terms):
                    if idx - rel_idx <= 8:
                        _add_alias(out, seen, " ".join(raw[rel_idx:obj_end]))
        if len(out) >= max_aliases:
            break
    return out[:max_aliases]


def normalize_dates(
    text: str,
    reference_time: Optional[float] = None,
    anchor_events: Optional[list] = None,
) -> list[dict]:
    """Resolve reference-relative date expressions in `text` to ISO-8601 ranges.

    Returns [{'expr', 'start', 'end'}]. Ranges, not points; reference is `reference_time`
    (epoch) or now. When `anchor_events` is provided, simple event-relative phrases such as
    "the week after the conference" resolve against the best matching calendar event."""
    ref = datetime.fromtimestamp(reference_time) if reference_time else datetime.now()
    low = text.lower()
    out: list[dict] = []

    def add(expr, rng):
        out.append({"expr": expr, "start": rng[0], "end": rng[1]})

    def add_anchored(expr, rng):
        item = {"expr": expr.rstrip("?.!,;"), "start": rng[0], "end": rng[1], "anchored": True}
        if item not in out:
            out.append(item)

    def add_interval(expr, rng):
        start, end = rng
        item = {"expr": expr.rstrip("?.!,;"), "start": start, "end": end, "interval": True}
        if item not in out:
            out.append(item)

    # Explicit date intervals: "between August 11 and August 15 2023" / "from May 4 to May 8".
    # If only the second endpoint carries a year, apply that same year to the first endpoint.
    date_anchor = _date_anchor_pattern()
    for m in re.finditer(
        rf"\b(?:between|from)\s+({date_anchor})\s+(?:and|to|-)\s+({date_anchor})\b",
        low,
        re.I,
    ):
        left_raw, right_raw = m.group(1), m.group(2)
        right = _absolute_date_anchor(right_raw, ref)
        left_ref = ref
        if right is not None and re.search(r"\b(?:19|20)\d{2}\b", right_raw) and not re.search(
            r"\b(?:19|20)\d{2}\b", left_raw
        ):
            try:
                left_ref = ref.replace(year=right.year)
            except ValueError:
                left_ref = datetime(right.year, ref.month, min(ref.day, 28), ref.hour, ref.minute, ref.second)
        left = _absolute_date_anchor(left_raw, left_ref)
        if left is None or right is None:
            continue
        start, end = (left, right) if left <= right else (right, left)
        add_interval(m.group(0), _range(
            start.replace(hour=0, minute=0, second=0, microsecond=0),
            end.replace(hour=23, minute=59, second=59, microsecond=0),
        ))

    # Absolute ISO datetime / date.
    for m in re.finditer(r"\b(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b", text):
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
            add(m.group(0), _day_range(d))
        except ValueError:
            pass
    for m in re.finditer(r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})\b", text):
        mo = _MONTHS.get(m.group(1).lower())
        if mo:
            try:
                add(m.group(0), _day_range(datetime(int(m.group(3)), mo, int(m.group(2)))))
            except ValueError:
                pass
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\s+(\d{4})\b", text):
        mo = _MONTHS.get(m.group(2).lower())
        if mo:
            try:
                add(m.group(0), _day_range(datetime(int(m.group(3)), mo, int(m.group(1)))))
            except ValueError:
                pass

    # fuzzy ranges: recently / past week / fortnight / last two weeks.
    if re.search(r"\b(recently|lately)\b", low):
        add("recently", _range(ref - timedelta(days=7), ref))
    if re.search(r"\b(?:past|last)\s+(?:fortnight|two\s+weeks?)\b|\bfortnight\b", low):
        add("fortnight", _range(ref - timedelta(days=14), ref))
    fuzzy_n = {"couple": 2, "few": 3, "several": 4}
    for m in re.finditer(
        r"\b(?:past|last|previous)\s+(a\s+)?(couple|few|several)\s+"
        r"(day|week|month|year)s?\b",
        low,
    ):
        n, unit = fuzzy_n[m.group(2)], m.group(3)
        add(m.group(0), _range(ref - timedelta(days=n * _UNIT_DAYS[unit]), ref))
    for m in re.finditer(r"\bpast\s+(day|week|month|year)s?\b", low):
        unit = m.group(1)
        add(m.group(0), _range(ref - timedelta(days=_UNIT_DAYS[unit]), ref))
    for m in re.finditer(r"\b(?:past|last)\s+(\d+)\s+(day|week|month|year)s?\b", low):
        n, unit = int(m.group(1)), m.group(2)
        add(m.group(0), _range(ref - timedelta(days=n * _UNIT_DAYS[unit]), ref))

    # "<Month> <Year>" and "<Month>" (relative to ref year) and "last <Month>".
    for m in re.finditer(r"\b(early|mid|late)\s+([A-Za-z]{3,9})(?:\s+(\d{4}))?\b", text):
        mo = _MONTHS.get(m.group(2).lower())
        if mo:
            year = int(m.group(3)) if m.group(3) else ref.year
            add(m.group(0), _qualified_month_range(year, mo, m.group(1).lower()))
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

    # Absolute-anchor relative phrases: "two days before July 4, 2023", "the week before
    # 9 June 2023". This is the deterministic counterpart to event-relative chaining below.
    count_words = "|".join(sorted(_COUNT_WORDS, key=len, reverse=True))
    date_anchor = _date_anchor_pattern()
    for m in re.finditer(
        rf"\b(?:the\s+)?(?:(\d+|{count_words})\s+)?(day|week|month)s?\s+"
        rf"(before|after)\s+({date_anchor})\b",
        low,
    ):
        n, unit, rel, anchor = _count_word(m.group(1)), m.group(2), m.group(3), m.group(4)
        base = _absolute_date_anchor(anchor, ref)
        if base is None:
            continue
        add_anchored(m.group(0), _event_relative_range(base, unit, rel, n))

    for m in re.finditer(
        rf"\b(?:the\s+)?(following|next|previous|prior)\s+(day|week|month)\s+"
        rf"(?:(?:after|before)\s+)?({date_anchor})\b",
        low,
    ):
        direction, unit, anchor = m.group(1), m.group(2), m.group(3)
        base = _absolute_date_anchor(anchor, ref)
        if base is None:
            continue
        rel = "after" if direction in ("following", "next") else "before"
        add_anchored(m.group(0), _event_relative_range(base, unit, rel, 1))

    # Event-relative phrases, when the caller supplies the in-scope calendar. This is intentionally
    # conservative: it only returns a range if an anchor event is already known and dated.
    for m in re.finditer(
        rf"\b(?:the\s+)?(?:(\d+|{count_words})\s+)?(day|week|month)s?\s+"
        r"(before|after)\s+(.+?)(?:[?.!,;]|$)",
        low,
    ):
        n, unit, rel, anchor = _count_word(m.group(1)), m.group(2), m.group(3), m.group(4).strip()
        matches = _match_anchor_events(anchor, anchor_events)
        if not matches:
            continue
        rng = _event_relative_range(datetime.fromtimestamp(float(matches[0].start)), unit, rel, n)
        add_anchored(m.group(0), rng)

    for m in re.finditer(
        r"\b(?:the\s+)?(following|next|previous|prior)\s+(day|week|month)\s+"
        r"(?:(?:after|before)\s+)?(.+?)(?:[?.!,;]|$)",
        low,
    ):
        direction, unit, anchor = m.group(1), m.group(2), m.group(3).strip()
        rel = "after" if direction in ("following", "next") else "before"
        matches = _match_anchor_events(anchor, anchor_events)
        if not matches:
            continue
        rng = _event_relative_range(datetime.fromtimestamp(float(matches[0].start)), unit, rel, 1)
        add_anchored(m.group(0), rng)

    for m in re.finditer(
        r"\b(?:the\s+)?(day|week|month)\s+(before|after)\s+(.+?)(?:[?.!,;]|$)",
        low,
    ):
        unit, rel, anchor = m.group(1), m.group(2), m.group(3).strip()
        matches = _match_anchor_events(anchor, anchor_events)
        if not matches:
            continue
        rng = _event_relative_range(datetime.fromtimestamp(float(matches[0].start)), unit, rel, 1)
        add_anchored(m.group(0), rng)

    return out


_COUNT_KW = ("how many", "how often", "number of times", "total number", "count of", "count", "how much")
_ORDER_KW = ("first", "last", "earliest", "latest", "newest", "order", "in order", "sequence",
             "before", "after", "when did", "what time")
_MULTIHOP_KW = (" and ", " both ", " after ", " before ", " then ", "related to", "connection")
_LATEST_KW = ("latest", "newest", "current", "currently", "now", "today", "still", "recent", "recently")
_EARLIEST_KW = ("first", "earliest", "initial", "oldest")
_ID_RE = re.compile(r"\b([A-Z]{2,}-?\d{2,}|\d{4,}|#\w+)\b")
# Sentence-initial question words / determiners to exclude from entity extraction.
_QWORDS = {"how", "what", "when", "where", "why", "who", "which", "whom", "whose",
           "did", "does", "do", "is", "are", "was", "were", "the", "a", "an", "in",
           "on", "at", "to", "of", "and", "or", "i", "my", "you", "your", "he", "she"}


def parse_query(
    question: str,
    reference_time: Optional[float] = None,
    anchor_events: Optional[list] = None,
) -> dict:
    """Parse a question into {operation, ranges, entities, is_namey, is_multihop}.
    Used to drive the structured calendar filter and query-adaptive RRF weights.

    `anchor_events` lets the normal parser resolve event-relative phrases ("the week after the
    conference") against already-indexed calendar events; callers that do not have events keep the
    previous purely text/reference-time behavior.
    """
    low = question.lower()
    if any(k in low for k in _COUNT_KW):
        operation = "count"
    elif any(k in low for k in _ORDER_KW):
        operation = "order"
    else:
        operation = "filter"
    ranges = normalize_dates(question, reference_time, anchor_events)
    interval_last = bool(re.search(r"\b(last|past|previous)\s+\w+", low))
    bare_last = bool(re.search(r"\blast\b", low)) and not interval_last
    if (any(k in low for k in _LATEST_KW) and not interval_last) or bare_last:
        temporal_order = "desc"
    elif any(k in low for k in _EARLIEST_KW):
        temporal_order = "asc"
    else:
        temporal_order = None
    # Entities: proper-noun-ish capitalized tokens + IDs/quoted spans (deterministic, no LLM).
    entities = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", question)
    entities += [m.group(0) for m in _ID_RE.finditer(question)]
    entities += re.findall(r'"([^"]+)"', question)
    entities = [e.strip() for e in entities if e.strip() and e.lower() not in _QWORDS]
    entities = list(dict.fromkeys(entities))
    is_namey = bool(_ID_RE.search(question)) or bool(ranges) or len(entities) >= 1
    is_multihop = any(k in low for k in _MULTIHOP_KW) or len(entities) >= 3
    return {"operation": operation, "ranges": ranges, "entities": entities,
            "temporal_order": temporal_order,
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
    ranges = effective_date_ranges(parsed.get("ranges", []))
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
    # Most-specific first (matches the MOST query entities). For "recent/latest/current" questions,
    # put the newest matching event first; ordinary ranges like "last week" stay chronological.
    reverse_time = parsed.get("temporal_order") == "desc"
    scored.sort(
        key=lambda t: (
            -t[0],
            -(t[1].start if t[1].start is not None else 0.0)
            if reverse_time else (t[1].start if t[1].start is not None else 0.0),
        )
    )
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
