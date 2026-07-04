"""Write-time event identity (P2): lemma + object-head tags on event claims.

Read-time date clustering was built and reverted three times -- date proximity cannot
distinguish 'same event retold' from 'different events sharing a noun'. Identity must be
decided where the information exists: at extraction, each event claim gets a canonical
action LEMMA and the OBJECT HEAD noun, plus the best event date the atom itself states
(with an explicit precision rank). A when-question then resolves by (lemma family match
to the question verb, object tie), and retellings collapse into one instance whose most
precise date wins.

Families here are deliberately the ONCE-ISH events (a shop opens once, an album releases
once, a couple marries once); repeatable actions (win, pick up, visit) stay out so the
repeated-event latest-wins semantics of the legacy path -- and the time-invariant
sidecar that encodes it -- are untouched. A 45-day span guard declines multi-instance
evidence instead of guessing.
"""
from __future__ import annotations

import re
from typing import Optional

INSTANCE_LEMMA_FAMILIES: dict[str, frozenset[str]] = {
    "release": frozenset({"release", "released", "releases", "drop", "dropped", "drops",
                          "debut", "debuted", "launch", "launched"}),
    "open": frozenset({"open", "opened", "opens", "start", "started", "starts", "begin",
                       "began", "begun", "founded", "launch", "launched"}),
    "marry": frozenset({"marry", "married", "wed", "engaged"}),
    "graduate": frozenset({"graduate", "graduated"}),
    "move": frozenset({"move", "moved", "relocate", "relocated"}),
    "adopt": frozenset({"adopt", "adopted", "rescue", "rescued"}),
    "team": frozenset({"team", "teamed", "collaborate", "collaborated", "partner",
                       "partnered"}),
}

_LEMMA_OF: dict[str, str] = {}
for _canon, _members in INSTANCE_LEMMA_FAMILIES.items():
    for _m in _members:
        _LEMMA_OF.setdefault(_m, _canon)

_HEAD_STOP = frozenset({
    "a", "an", "the", "my", "his", "her", "their", "our", "your", "own", "new", "old",
    "little", "big", "small", "first", "second", "third", "this", "that", "some",
})

PRECISION_EXPLICIT = 3
PRECISION_RELATIVE_DAY = 2
PRECISION_STATEMENT = 1
PRECISION_WINDOW = 0

INSTANCE_SPAN_GUARD_DAYS = 45


def canon_lemma(verb: str) -> str:
    return _LEMMA_OF.get((verb or "").lower(), "")


def question_lemma(query: str) -> str:
    q = (query or "").lower()
    if not q.strip().startswith("when"):
        return ""
    for w in re.findall(r"[a-z][\w'-]*", q):
        lemma = _LEMMA_OF.get(w, "")
        if lemma:
            return lemma
    return ""


def obj_head(obj_text: str) -> str:
    words = [w.lower() for w in re.findall(r"[A-Za-z][\w'-]*", obj_text or "")]
    for w in reversed(words):
        if w not in _HEAD_STOP:
            return w
    return ""


def format_answer(iso_date: str, precision: int) -> Optional[str]:
    """A statement-derived date is honest at MONTH granularity -- the report bounds the
    month, not the day; explicit and relative-day dates answer as the full day."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})$", iso_date or "")
    if not m:
        return None
    if precision >= PRECISION_RELATIVE_DAY:
        return iso_date
    import calendar
    return f"{calendar.month_name[int(m.group(2))]} {m.group(1)}"
