"""Claim-tier QA operators: dialogue Q->A crystal matching, premise-affinity inference, and
experience-location extraction.

These operators are generic by construction: they classify question SHAPE and match against claim
metadata (recorded dialogue questions, affinity predicates, action stems). They must never branch
on benchmark sample ids, fixed questions, or dataset entities.
"""
from __future__ import annotations

import re

from eidetic.models import ClaimRecord


def _ro():
    from eidetic.smqe import record_ops
    return record_ops


def _verb_base_forms(verb: str) -> set[str]:
    """Candidate base forms of a possibly-inflected verb (camped -> camp, hiking -> hike,
    carries -> carry). Over-generation is safe: variants are only used as match sets."""
    bases = {verb}
    for suffix in ("ing", "ied", "ed", "es", "s", "d"):
        if verb.endswith(suffix) and len(verb) - len(suffix) >= 3:
            stem = verb[: -len(suffix)]
            if suffix == "ied":
                bases.add(stem + "y")
                continue
            bases.add(stem)
            bases.add(stem + "e")               # hoped -> hope, hiking -> hike
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                bases.add(stem[:-1])            # stopped -> stop, running -> run
    return {b for b in bases if len(b) >= 3}


def _action_location_phrase(atom: str, action_terms: set[str]) -> str:
    """Location stated with the queried action: 'camping at the beach' -> 'the beach'."""
    ro = _ro()
    variants = sorted({t for t in action_terms if len(t) > 2}, key=len, reverse=True)
    if not variants:
        return ""
    pat = "|".join(re.escape(v) for v in variants)
    m = re.search(
        rf"\b(?:{pat})\b(?:\s+\w+){{0,3}}?\s+(?:at|in)\s+(?:the\s+)?"
        rf"([a-z][a-z' -]{{2,40}}?)(?=[.,;!?]|\s+(?:last|this|next|two|a\s+few|yesterday|"
        rf"today|with|and|for|because|when|where)\b|$)",
        ro._strip_role(atom),
        re.I,
    )
    return ro._clean(m.group(1)) if m else ""


_ADVICE_REQUEST_RE = re.compile(
    r"\b(?:any\s+(?:tips?|ideas?|suggestions?|advice|recommendations?)|tips?\s+for|ideas?\s+for|"
    r"suggest|recommend|what\s+should\s+i|how\s+should\s+i|ways\s+to|help\s+me)\b",
    re.I,
)


def _dialogue_answer_match(query: str, atoms: list[tuple[float, object, str]]) -> tuple[str, list[tuple[float, object, str]]]:
    """A claim that was the literal in-conversation answer to an equivalent question answers the
    query directly: match query terms against the RECORDED question (paraphrase-stable), require
    subject/entity agreement, and return the crystallized answer sentence.

    Advice requests are excluded: they ask for fresh synthesis grounded in preferences, not the
    replay of one past reply."""
    if _ADVICE_REQUEST_RE.search(query or ""):
        return "", []
    ro = _ro()
    qkeys = {ro._count_term_key(t) for t in ro._query_terms(query)}
    if not qkeys:
        return "", []
    entity_terms = ro._subject_entity_terms(query)
    entity_keys = {ro._count_term_key(t) for t in entity_terms}
    best: tuple[int, float, object, str] | None = None
    for score, item, atom in atoms[:30]:
        if not isinstance(item, ClaimRecord) or item.filters.get("dialogue") != "answer":
            continue
        recorded_q = str(item.filters.get("question") or "")
        if not recorded_q:
            continue
        rq_keys = {ro._count_term_key(t) for t in ro._query_terms(recorded_q)} - entity_keys
        if not rq_keys:
            continue
        overlap = (qkeys - entity_keys) & rq_keys
        # Most of the recorded question's content must appear in the query, with >=2 shared
        # content terms, so "plans for the summer" matches "Any fun plans for the summer?" but
        # unrelated questions cannot bridge on one incidental word.
        if len(overlap) < 2 or 2 * len(overlap) < len(rq_keys):
            continue
        if entity_terms and ro._entity_hit_count(entity_terms, ro._item_match_text(item, atom)) == 0:
            continue
        if best is None or (len(overlap), score) > (best[0], best[1]):
            best = (len(overlap), score, item, atom)
    if best is None:
        return "", []
    _overlap, score, item, atom = best
    value = ro._answer_value(query, atom, item) or ro._clean(ro._strip_role(atom))
    if not value:
        return "", []
    return value, [(score, item, atom)]


_YESNO_HEAD_RE = re.compile(
    r"^\s*(?:is|are|was|were|am|do|does|did|has|have|had)\b(?!\s+you\b)", re.I)
_PROP_NEGATION_RE = re.compile(
    r"\b(?:not|no\s+longer|never|stopped|quit|isn't|aren't|wasn't|weren't|don't|doesn't|"
    r"didn't|hasn't|haven't|hadn't|anymore)\b",
    re.I,
)
_PROP_QUERY_STOP = {
    "actually", "anymore", "currently", "ever", "just", "now", "really", "same", "still",
}


def _proposition_confirmation_answer(query: str, atoms: list[tuple[float, object, str]]) -> tuple[str, list[tuple[float, object, str]]]:
    """A yes/no question whose proposition a memory literally asserts ('Is my mom using the same
    grocery list method as me?' <- 'my mom is actually using the same grocery list app as me now')
    answers 'Yes - <premise>' anchored on the asserting atom.

    Positive confirmations only: a negated or absent assertion produces no answer here (the
    reader keeps polarity judgment), and the embedded premise makes the strict query-aware
    verification hypothesis self-evident against the source record."""
    q = query or ""
    if not _YESNO_HEAD_RE.match(q) or " or " in q.lower():
        return "", []
    if _ADVICE_REQUEST_RE.search(q) or _PROP_NEGATION_RE.search(q):
        return "", []
    ro = _ro()
    prop_terms = {
        ro._count_term_key(t) for t in ro._query_terms(q)
        if t not in _PROP_QUERY_STOP and not t.isdigit()
    }
    if len(prop_terms) < 3:
        return "", []
    required = max(2, (len(prop_terms) + 1) // 2)
    best: tuple[int, float, object, str] | None = None
    for score, item, atom in atoms[:30]:
        text = ro._strip_role(atom)
        if _PROP_NEGATION_RE.search(text):
            continue
        atom_keys = {ro._count_term_key(t) for t in ro._expanded_terms(text)}
        hits = len(prop_terms & atom_keys)
        if hits < required:
            continue
        if best is None or (hits, score) > (best[0], best[1]):
            best = (hits, score, item, atom)
    if best is None:
        return "", []
    _hits, score, item, atom = best
    return f"Yes - {ro._clean(ro._strip_role(atom))}", [(score, item, atom)]


_REMIND_NAME_RE = re.compile(
    r"\b(?:remind\s+me|what\s+was\s+the\s+name\s+of|the\s+name\s+of\s+(?:that|the))\b", re.I)
_NAME_HEAD_RE = re.compile(
    r"^\s*(?:\d+[.)]\s*)?\*{0,2}((?:The\s+)?[A-Z][\w'&-]*(?:\s+[A-Z][\w'&-]*){0,4})\*{0,2}\s*[-:–—]")
_RECOMMEND_NAME_RE = re.compile(
    r"\b(?:recommend(?:ed)?|suggest(?:ed)?|try|check\s+out)\s+((?:The\s+)?[A-Z][\w'&-]*(?:\s+[A-Z][\w'&-]*){0,4})")
_LOCATED_AT_RE = re.compile(
    r"\blocated\s+(?:at|in|on)\s+((?:the\s+)?[A-Z][\w'&-]*(?:\s+[A-Z][\w'&-]*){0,3})")
_GENERIC_NAME_WORDS = {
    "note", "notes", "post", "posts", "tip", "tips", "step", "steps", "option", "options",
    "warning", "image", "video", "here", "first", "second", "third", "item", "items",
}


def _named_recommendation_answer(query: str, atoms: list[tuple[float, object, str]]) -> tuple[str, list[tuple[float, object, str]]]:
    """Remind-me recall of a previously recommended NAMED thing: match the demonstrative noun
    phrase's content terms against recommendation atoms and return the proper name (with its
    stated location when the source gives one)."""
    ro = _ro()
    if not _REMIND_NAME_RE.search(query or ""):
        return "", []
    qkeys = {
        ro._count_term_key(t) for t in ro._query_terms(query)
        if t not in {"remind", "name", "wondering", "planning", "revisit", "visit", "trip",
                     "back", "going", "checking", "previous", "chat", "conversation"}
    }
    if not qkeys:
        return "", []
    best: tuple[int, float, object, str, str] | None = None
    for score, item, atom in atoms[:40]:
        text = atom.strip()
        m = _NAME_HEAD_RE.match(text) or _RECOMMEND_NAME_RE.search(ro._strip_role(text))
        if not m:
            continue
        name = m.group(1).strip()
        words = name.split()
        if len(words) == 1 and words[0].lower() in _GENERIC_NAME_WORDS:
            continue
        akeys = {ro._count_term_key(t) for t in ro._terms(atom)}
        hits = len(qkeys & akeys)
        if hits < 2:
            continue
        if best is None or (hits, score) > (best[0], best[1]):
            best = (hits, score, item, atom, name)
    if best is None:
        return "", []
    _hits, score, item, atom, name = best
    loc = _LOCATED_AT_RE.search(atom)
    answer = f"{name} at {loc.group(1)}" if loc else name
    return answer, [(score, item, atom)]


_AFFINITY_MARKER_RE = re.compile(
    r"\b(?:have|has|had|got|own|owns|owned|collect|collects|collected|love|loves|loved|"
    r"like|likes|liked|enjoy|enjoys|enjoyed|favou?rite|fan\s+of|into|passionate\s+about)\b",
    re.I,
)
_LIKELY_INFERENCE_RE = re.compile(
    r"\bwould\b[^.?!]{0,80}\blikely\b|\blikely\s+(?:have|has|enjoy|like|want|be|get|buy)\b",
    re.I,
)


def _premise_affinity_answer(query: str, atoms: list[tuple[float, object, str]]) -> tuple[str, list[tuple[float, object, str]]]:
    """'Would X likely have/enjoy Y?' -> 'Yes - <source premise>' when a non-negated affinity
    statement by/about X overlaps Y's content terms. No matching premise -> no answer (fallback)."""
    ro = _ro()
    if not _LIKELY_INFERENCE_RE.search(query or ""):
        return "", []
    entity_terms = ro._subject_entity_terms(query)
    target_terms = {
        ro._count_term_key(t)
        for t in ro._query_terms(query) - entity_terms
        if t not in {"likely", "would", "have", "has", "had", "own", "enjoy", "like",
                     "want", "be", "get", "buy", "her", "his", "their", "them", "there"}
        and not t.isdigit()
    }
    if not target_terms:
        return "", []
    best: tuple[int, float, object, str, str] | None = None
    for score, item, atom in atoms[:30]:
        text = ro._strip_role(atom)
        if entity_terms and ro._entity_hit_count(entity_terms, ro._item_match_text(item, atom)) == 0:
            continue
        if not _AFFINITY_MARKER_RE.search(text):
            continue
        atom_keys = {ro._count_term_key(t) for t in ro._expanded_terms(text)}
        hits = target_terms & atom_keys
        if not hits:
            continue
        low = text.lower()
        negated = any(
            re.search(
                rf"\b{re.escape(h)}[a-z]*[-\s]?(?:free|less)\b|"
                rf"\b(?:no|not|never|avoid|without|hate|hates|dislike|dislikes)\b[^.;!?]{{0,25}}\b{re.escape(h)}",
                low,
            )
            for h in hits
        )
        if negated:
            continue
        if best is None or (len(hits), score) > (best[0], best[1]):
            best = (len(hits), score, item, atom, text)
    if best is None:
        return "", []
    _hits, score, item, atom, text = best
    return f"Yes - {ro._clean(text)}", [(score, item, atom)]
