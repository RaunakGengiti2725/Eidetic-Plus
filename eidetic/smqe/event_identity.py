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

# Event-date claim family vocabulary (shared by write and read sides so "When was X's
# concert..." matches a claim written from "My concert ... was ..."). Deliberately
# SEPARATE from INSTANCE_LEMMA_FAMILIES: repeatable verbs never enter the once-ish
# contract (reverted 3x history); the family carries its own filter marker
# (event == "dated") and the read op declines on multi-instance ambiguity instead.
EVENT_NOUN_LEMMAS: dict[str, str] = {
    "concert": "concert", "wedding": "wedding", "graduation": "graduation",
    "ceremony": "ceremony", "interview": "interview", "flight": "flight",
    "trip": "trip", "recital": "recital", "performance": "concert",
    "show": "concert", "appointment": "appointment", "exam": "exam",
    "surgery": "surgery",
}

# Past-tense surface form -> canonical lemma for dated action verbs. Kept here (not in
# claim_extraction) so the read op imports ONE vocabulary and write/read never drift.
DATED_EVENT_VERB_LEMMAS: dict[str, str] = {
    "adopted": "adopt", "rescued": "rescue", "married": "marry",
    "graduated": "graduate", "moved": "move", "released": "release",
    "opened": "open", "started": "start", "joined": "join", "launched": "launch",
    "performed": "perform", "attended": "attend", "visited": "visit",
    "met": "meet", "traveled": "travel", "travelled": "travel", "flew": "fly",
    "went": "go", "reconnected": "reconnect", "held": "hold",
}

_DATED_SURFACE_OF: dict[str, str] = {}
for _surf, _lem in DATED_EVENT_VERB_LEMMAS.items():
    _DATED_SURFACE_OF[_surf] = _lem
    _DATED_SURFACE_OF.setdefault(_lem, _lem)

_HEAD_STOP = frozenset({
    "a", "an", "the", "my", "his", "her", "their", "our", "your", "own", "new", "old",
    "little", "big", "small", "first", "second", "third", "this", "that", "some",
    "and", "or", "but", "with", "for", "about", "into", "onto",
    "ago", "today", "yesterday", "tomorrow", "lately", "recently", "now", "soon",
    "time", "week", "weeks", "month", "months", "year", "years", "day", "days",
    "myself", "yourself", "himself", "herself", "themselves", "ourselves",
    "me", "him", "them", "us", "it", "journey", "feeling", "feelings",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "few", "couple", "several",
})

PRECISION_EXPLICIT = 3
PRECISION_RELATIVE_DAY = 2
PRECISION_STATEMENT = 1
PRECISION_WINDOW = 0

INSTANCE_SPAN_GUARD_DAYS = 45


def lemmas_compatible(a: str, b: str) -> bool:
    """Equal lemmas, or families sharing a surface verb: 'launch' lives in both release
    and open, so a launched-claim answers an open-question and vice versa."""
    if not a or not b:
        return False
    if a == b:
        return True
    fa, fb = INSTANCE_LEMMA_FAMILIES.get(a), INSTANCE_LEMMA_FAMILIES.get(b)
    return bool(fa and fb and fa & fb)


def canon_lemma(verb: str) -> str:
    return _LEMMA_OF.get((verb or "").lower(), "")


def dated_lemma_for(word: str) -> str:
    """Canonical dated-event lemma for a surface verb ('adopted' or 'adopt' -> 'adopt')."""
    return _DATED_SURFACE_OF.get((word or "").lower(), "")


def dated_lemmas_compatible(a: str, b: str) -> bool:
    """Exact lemma match, or both map into the same INSTANCE_LEMMA_FAMILIES family via the
    existing lemmas_compatible ('rescue' answers an adopt-question and vice versa)."""
    if not a or not b:
        return False
    if a == b:
        return True
    ca = canon_lemma(a) or a
    cb = canon_lemma(b) or b
    return ca == cb or lemmas_compatible(ca, cb)


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
