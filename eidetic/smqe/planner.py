"""Rules-first planner for the Structured Memory Query Engine.

The planner is intentionally generic: it classifies question shape and extracts only terms/entities
from the query. It must never branch on benchmark sample ids, fixed questions, or dataset entities.
"""
from __future__ import annotations

import re
from typing import Optional

from eidetic.events import effective_date_ranges, parse_query
from eidetic.models import ExecutionPlan


_ENTITY_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "our", "shall", "should", "the",
    "their", "there", "they", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
}


def _query_terms(text: str) -> list[str]:
    terms = []
    for raw in re.findall(r"[a-z0-9][a-z0-9_'-]*", (text or "").lower()):
        t = re.sub(r"'s$", "", raw)
        if len(t) > 1 and t not in _ENTITY_STOP:
            terms.append(t)
    return terms


def _slot(query: str) -> str:
    q = (query or "").lower()
    patterns = (
        r"\bwhat\s+(?:is|was|are|were)\s+(?:my|the|their|his|her|our)?\s*([^?.,;]+)",
        r"\bwhere\s+(?:is|was|are|were|did|do|does)\s+([^?.,;]+)",
        r"\bwhich\s+([^?.,;]+)",
        r"\bhow\s+(?:much|many|long|often)\s+([^?.,;]+)",
    )
    for pat in patterns:
        m = re.search(pat, q, re.I)
        if m:
            return " ".join(_query_terms(m.group(1)))[:120]
    terms = _query_terms(q)
    return " ".join(terms[:8])


def plan_query(query: str, at: Optional[float] = None) -> ExecutionPlan:
    q = (query or "").strip()
    low = q.lower()
    try:
        parsed = parse_query(q, at)
        entities = [
            str(e)
            for e in parsed.get("entities", [])
            if str(e).strip() and str(e).strip().lower() not in _ENTITY_STOP
        ]
        date_ranges = effective_date_ranges(parsed.get("ranges", []))
    except Exception:
        parsed = {}
        entities = []
        date_ranges = []
    if not entities:
        entities = []
        seen = set()
        for raw in re.findall(r"\b[A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,3}\b", q):
            key = raw.lower()
            if key not in seen and key not in _ENTITY_STOP:
                seen.add(key)
                entities.append(raw)

    op = "open_inference"
    reason = "default open inference"
    unit = ""
    requires_synthesis = False

    if re.search(r"\b(?:which|what)\b.+\bfirst\b.+\bor\b|\bhappened\s+first\b|\bwhich\s+event\b", low):
        op, reason = "event_order", "event ordering question"
    elif re.search(r"\b(?:better\s+for\s+me|probably)\b", low) and " or " in low:
        op, reason, requires_synthesis = "open_inference", "option inference question", True
    elif (
        re.search(r"\b(table|column|rotation|spreadsheet|which\s+(?:row|column))\b", low)
        or (
            re.search(r"\bschedule\b", low)
            and re.search(
                r"\b(?:shift|row|column|rotation|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                low,
            )
        )
    ) or (
        re.search(r"\brow\b", low) and "in a row" not in low and re.search(r"\b(?:table|schedule|spreadsheet|column)\b", low)
    ):
        op, reason = "table_lookup", "table-shaped lookup"
    elif re.search(r"\bhow\s+many\s+(days?|weeks?|months?|years?)\b", low) and re.search(
        r"\b(ago|since|between|after|before|passed|elapsed)\b", low
    ):
        op, reason = "temporal_delta", "elapsed-time question"
        m = re.search(r"\b(days?|weeks?|months?|years?)\b", low)
        unit = m.group(1) if m else ""
    elif re.search(r"\b(?:total|sum|combined|altogether|overall)\b", low) or (
        re.search(r"\bhow\s+much\b", low)
        and re.search(r"\b(?:total|combined|altogether|overall|spent|expenses?|related|since)\b", low)
    ) or (
        re.search(r"\bhow\s+long\b", low) and re.search(r"\b(?:total|sum|combined|altogether|spent)\b", low)
    ):
        op, reason = "multi_session_sum", "aggregate scalar question"
    elif re.search(r"\b(?:how\s+much|amount|money|cost|spent|pre[-\s]?approved)\b", low):
        op, reason = "latest_value", "scalar amount lookup"
    elif re.search(r"\b(?:when|what\s+(?:date|day|month|year)|which\s+(?:date|day|month|year))\b", low):
        op, reason = "relative_temporal", "date/time lookup"
    elif re.search(r"\bhow\s+many\s+hours?\b", low) and re.search(r"\b(?:spent|worked|put\s+in)\b", low) and not re.search(r"\b(?:total|sum|combined|altogether)\b", low):
        op, reason = "latest_value", "latest duration value"
    elif re.search(r"\b(?:how\s+many|count|number\s+of)\b", low):
        op, reason = "count_aggregate", "count question"
    elif re.search(
        r"\b(?:prefer|preference|favorite|favourite|like|suggest(?:ions?)?|recommend(?:ations?)?|advice|tips?|ideas?|should\s+i|what\s+should|what\s+can|how\s+should|rather|choose|pick)\b",
        low,
    ) and not re.search(r"\b(?:where|when|who|what|which)\s+(?:did|do|does)\b", low):
        op, reason, requires_synthesis = "preference_synth", "preference/suggestion question", True
    elif re.search(r"\b(?:what|where|when|which|who)\s+(?:did|do|does|was|were)\b", low) and re.search(
        r"\b(?:say|tell|mention|ask|answer|reply|discuss|talk)\b", low
    ):
        op, reason = "speaker_fact", "speaker-attributed recall"
    elif re.search(
        r"\b(?:degree|graduat|research(?:ed|ing)?|look(?:ed|ing)?\s+into|relationship status|marital status|identity|fields?|career|remind me|name of)\b",
        low,
    ):
        op, reason = "latest_value", "typed slot lookup"
    elif re.search(r"\b(?:current|currently|latest|last|recent|recently|now|pre[-\s]?approved|status)\b", low):
        op, reason = "latest_value", "latest/current value question"
    elif re.search(r"\b(?:where|who|which|what\s+(?:is|was|are|were)|how\s+much)\b", low):
        op, reason = "latest_value", "slot lookup question"

    return ExecutionPlan(
        op=op,
        entities=entities,
        slot=_slot(q),
        filters={"terms": _query_terms(q), "parsed_op": parsed.get("op"), "date_ranges": date_ranges},
        as_of=at,
        unit=unit,
        requires_synthesis=requires_synthesis,
        reason=reason,
    )
