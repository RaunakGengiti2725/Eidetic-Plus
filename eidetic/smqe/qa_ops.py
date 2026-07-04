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
_PLEASANTRY_ANSWER_RE = re.compile(
    r"^\s*(?:hey|hi|hello|thanks|thank\s+you|wow|congrats|congratulations|awesome|great|"
    r"cool|nice|yeah|yes|no|ok|okay)\b[\s,!.]*(?:[A-Z][\w'-]+[\s,!.]*)?(?:thanks|thank\s+you)?[\s,!.]*$",
    re.I,
)


def _wh_class(text: str) -> str:
    m = re.search(r"\b(what|which|who|whose|where|when|why|how)\b", text or "", re.I)
    if not m:
        return ""
    w = m.group(1).lower()
    return {"which": "what", "whose": "who"}.get(w, w)


_PROBLEM_QUERY_PREDICATES = (
    (re.compile(r"\bwhy\b.*\bdecid|\bdecid\w*\b.*\bwhy\b", re.I), "decision", "rationale"),
    (re.compile(r"\bdecid\w*\b|\bdecision\b", re.I), "decision", ""),
    (re.compile(r"\bblock(?:er|ers|ing|ed)\b", re.I), "blocker", ""),
    (re.compile(r"\bhypothes\w*\b|\btheor\w*\b", re.I), "hypothesis", ""),
    (re.compile(r"\bhand(?:off|offs|ed\s+off)\b", re.I), "handoff", ""),
    (re.compile(r"\bstatus\b|\bstate\s+of\b", re.I), "status", ""),
    (re.compile(r"\bgoal\b|\btrying\s+to\b", re.I), "goal", ""),
)


def _problem_claim_answer(query: str, atoms: list[tuple[float, object, str]]) -> tuple[str, list[tuple[float, object, str]]]:
    """P6 typed SELECT over war-room claims: 'what did we decide about X' reads decision
    claims (object tied to X), 'why' variants answer the stored rationale, blockers and
    hypotheses enumerate, status/goal answer latest-wins. Fires only when problem claims
    exist in the pool (PROBLEM_CLAIMS off means none exist -- byte-identical baseline)."""
    ro = _ro()
    pool = [(s, i, a) for s, i, a in atoms
            if isinstance(i, ClaimRecord) and i.claim_type == "problem"]
    if not pool:
        return "", []
    q = query or ""
    predicate = rationale_slot = None
    for pat, pred, slot in _PROBLEM_QUERY_PREDICATES:
        if pat.search(q):
            predicate, rationale_slot = pred, slot
            break
    if predicate is None:
        return "", []
    qterms = {t for t in ro._query_terms(q)}
    rows = []
    for score, item, atom in pool:
        if item.predicate != predicate:
            continue
        tie_text = f"{item.object} {item.filters.get('rationale', '')} {item.filters.get('goal', '')}"
        tie_terms = ro._expanded_terms(tie_text)
        content = qterms - {predicate, "decide", "decided", "status", "goal", "blocker",
                            "blockers", "hypothesis", "hypotheses", "handoff", "handoffs",
                            "problem", "have", "know"}
        single_problem = len({i.filters.get("problem_id") for _s, i, _a in pool}) == 1
        if content and not single_problem and not (
                ro._expanded_terms(" ".join(content)) & tie_terms):
            continue
        rows.append((float(item.valid_at or 0.0), score, item, atom))
    if not rows:
        return "", []
    rows.sort(key=lambda r: (-r[0], -r[1]))
    if predicate in {"blocker", "hypothesis", "handoff"} and len(rows) > 1:
        vals, selected, seen = [], [], set()
        for _va, score, item, atom in rows:
            key = ro._norm_key(item.object)
            if key in seen:
                continue
            seen.add(key)
            vals.append(item.object)
            selected.append((score, item, atom))
        return "; ".join(vals[:6]), selected[:6]
    _va, score, item, atom = rows[0]
    if rationale_slot:
        value = str(item.filters.get(rationale_slot) or "")
        if not value:
            return "", []
        return value, [(score, item, atom)]
    return str(item.object), [(score, item, atom)]


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
        # A crystal only answers a question of the SAME wh-class: 'How did the tournament GO?'
        # shares content words with 'WHAT game was the tournament?' but its recorded answer
        # addresses a different slot entirely.
        q_wh, rq_wh = _wh_class(query), _wh_class(recorded_q)
        if q_wh and rq_wh and q_wh != rq_wh:
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
        # ... and EVERY query content term must appear in the recorded question or its
        # answer: 'what is X working on OPENING?' matched a broader working-on crystal whose
        # reply never mentions opening -- the slot-defining term was covered by nothing, and
        # the wrong instance shipped verified on the fresh holdout.
        atom_keys = {ro._count_term_key(t) for t in ro._query_terms(atom)}
        if (qkeys - entity_keys) - rq_keys - atom_keys:
            continue
        if entity_terms and ro._entity_hit_count(entity_terms, ro._item_match_text(item, atom)) == 0:
            continue
        if best is None or (len(overlap), score) > (best[0], best[1]):
            best = (len(overlap), score, item, atom)
    if best is None:
        return "", []
    _overlap, score, item, atom = best
    value = ro._answer_value(query, atom, item) or ro._clean(ro._strip_role(atom))
    if not value or _PLEASANTRY_ANSWER_RE.match(value):
        # A greeting-only crystal ('Hey Joanna, thanks!') answers nothing.
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
    answers 'Yes - <premise>' anchored on the asserting atom. A stored NEGATED assertion of the
    proposition ('I've never been to a jazz festival') is antimemory and answers 'No - <premise>'
    the same way. When both polarities are asserted (a retraction or a re-assertion), the LATEST
    assertion wins.

    Absence stays closed: no matching assertion of either polarity produces no answer here (the
    reader keeps closed-world judgment), and the embedded premise makes the strict query-aware
    verification hypothesis self-evident against the source record."""
    q = query or ""
    if not _YESNO_HEAD_RE.match(q) or " or " in q.lower():
        return "", []
    if _ADVICE_REQUEST_RE.search(q) or _PROP_NEGATION_RE.search(q):
        return "", []
    ro = _ro()
    # The leading auxiliary is the interrogative marker, not proposition content. Raw morphology
    # only (_terms): the _query_terms topical packs would inject unrelated expansion words into
    # the proposition and inflate the coverage requirement.
    q_body = _YESNO_HEAD_RE.sub("", q, count=1)
    raw_terms = {
        t for t in ro._terms(q_body)
        if len(t) > 2 and t not in _PROP_QUERY_STOP and not t.isdigit()
    }
    # One matchable UNIT per content word: morphological duplicates from term expansion
    # ("taking" also yields "tak") collapse into the unit, and verbs match family-wide
    # (take/takes/took/taken) so inflection never breaks proposition coverage.
    units: list[set[str]] = []
    for term in sorted(raw_terms, key=len, reverse=True):
        key = ro._count_term_key(term)
        if any(term in unit or key in unit for unit in units):
            continue
        exp = {term, key}
        for base in _verb_base_forms(term):
            exp |= ro._verb_variants(base)
        unit = {ro._count_term_key(x) for x in exp if len(x) > 2}
        if unit:
            units.append(unit)
    if len(units) < 2:
        return "", []
    # A two-unit proposition must hit BOTH units; longer ones must hit at least half.
    required = max(2, (len(units) + 1) // 2)

    def unit_index(token: str) -> Optional[int]:
        key = ro._count_term_key(re.sub(r"'s$", "", token))
        for pos, unit in enumerate(units):
            if key in unit or token in unit:
                return pos
        return None

    # Textually adjacent modifier+head content pairs ("botanical garden"): an atom that matches
    # the head but not its modifier refers to a DIFFERENT thing ("sculpture garden") and must
    # not confirm the proposition. Progressive verbs are not modifiers ("taking pottery").
    token_seq = re.findall(r"[a-z0-9][a-z0-9'-]*", q_body.lower())
    modifier_pairs: list[tuple[int, int]] = []
    for tok_a, tok_b in zip(token_seq, token_seq[1:]):
        if tok_a.endswith("ing"):
            continue
        ua, ub = unit_index(tok_a), unit_index(tok_b)
        if ua is not None and ub is not None and ua != ub:
            modifier_pairs.append((ua, ub))
    best: dict[str, tuple[int, float, float, object, str]] = {}
    for score, item, atom in atoms[:30]:
        text = ro._strip_role(atom)
        # Unit coverage sees the speaker/subject (a question naming the speaker must match the
        # atom they said); polarity is judged on the stripped assertion text itself.
        atom_keys = {ro._count_term_key(t)
                     for t in ro._expanded_terms(ro._item_match_text(item, atom))}
        matched = {pos for pos, unit in enumerate(units) if unit & atom_keys}
        hits = len(matched)
        if hits < required:
            continue
        if any(head in matched and mod not in matched for mod, head in modifier_pairs):
            continue
        polarity = "no" if _PROP_NEGATION_RE.search(text) else "yes"
        valid_at = float(getattr(item, "valid_at", 0.0) or 0.0)
        prev = best.get(polarity)
        if prev is None or (hits, score) > (prev[0], prev[1]):
            best[polarity] = (hits, score, valid_at, item, atom)
    if not best:
        return "", []
    # Retraction rule: with both polarities asserted, the latest assertion is the current truth.
    if len(best) == 2:
        polarity = max(best, key=lambda p: best[p][2])
    else:
        polarity = next(iter(best))
    _hits, score, _valid_at, item, atom = best[polarity]
    label = "Yes" if polarity == "yes" else "No"
    return f"{label} - {ro._clean(ro._strip_role(atom))}", [(score, item, atom)]


_PLURAL_WH_RE = re.compile(r"\b(?:which|what)\s+([a-z][a-z'-]{3,})\b", re.I)
_ENUM_SKIP_RE = re.compile(
    r"\bhow\s+many\b|\bfirst\b|\blast\b|\bmost\b|\bleast\b|\b\w+est\b|\bnumber\s+of\b",
    re.I,
)
_ENUM_VALUE_STOP_RE = re.compile(
    r"\s+\b(?:recently|lately|in|on|at|during|before|after|because|while|when|where|"
    r"which|that|so|this|last|past|next|for|with|and)\b",
    re.I,
)


def _plural_head_noun(query: str) -> str:
    """The plural wh-head noun of an enumeration question ('which COUNTRIES have I ...'), or
    ''. Plurality is morphological (the count key differs from the surface form), which keeps
    singular s-final nouns ('class', 'bus') out."""
    m = _PLURAL_WH_RE.search(query or "")
    if not m:
        return ""
    noun = m.group(1).lower()
    ro = _ro()
    key = ro._count_term_key(noun)
    if key == noun or len(key) < 3:
        return ""
    return noun


def _is_plural_enumeration_query(query: str) -> bool:
    from eidetic.config import get_settings
    if not get_settings().plural_enumeration_enabled:
        return False
    q = query or ""
    if _ENUM_SKIP_RE.search(q) or _ADVICE_REQUEST_RE.search(q):
        return False
    return bool(_plural_head_noun(q))


def _plural_enumeration_answer(query: str, atoms: list[tuple[float, object, str]]) -> tuple[str, list[tuple[float, object, str]]]:
    """'Which countries have I visited?' -> enumerate DISTINCT slot values across records, one
    support per contributing record (distinct memory_ids keep the composition verifiable under
    the witness rule). A 1-of-N single-record atom is never the verified answer to an
    enumeration question. Fewer than two distinct values -> fall through to today's paths."""
    if not _is_plural_enumeration_query(query):
        return "", []
    ro = _ro()
    action_terms, _target_terms = ro._count_profile(query)
    if not action_terms:
        return "", []
    variants = sorted((t for t in action_terms if len(t) > 2), key=len, reverse=True)
    values: list[str] = []
    selected: list[tuple[float, object, str]] = []
    seen_values: set[str] = set()
    seen_records: set[str] = set()
    for score, item, atom in atoms[:40]:
        text = ro._strip_role(atom)
        if not (ro._expanded_terms(text) & action_terms):
            continue
        value = ""
        for variant in variants:
            m = re.search(
                rf"\b{re.escape(variant)}\b\s+(?:the\s+|a\s+|an\s+|my\s+|some\s+|new\s+)?([^.;!?]+)",
                text,
                re.I,
            )
            if not m:
                continue
            # The category noun names the TYPE; the value is an INSTANCE, so no
            # category-term check on the value itself (unlike the count machinery).
            phrase = _ENUM_VALUE_STOP_RE.split(m.group(1), maxsplit=1)[0]
            phrase = ro._clean(phrase)
            # An enumerable VALUE is a short noun phrase, not a clause: promotion evaluation
            # showed unbounded captures assemble verified-wrong junk lists ("you know the
            # movie well"). <=4 words, no pronoun/verb-ish head, no trailing preposition.
            words = phrase.split()
            if (phrase and not phrase.isdigit() and 1 <= len(words) <= 4
                    and words[0].lower() not in {
                        "you", "i", "we", "he", "she", "they", "it", "even", "know",
                        "getting", "being", "trying", "going"}
                    and words[-1].lower() not in {"of", "in", "on", "at", "to", "with"}):
                value = phrase
                break
        if not value:
            continue
        key = ro._norm_key(value)
        rec_key = ro._group_key(item)
        if not key or key in seen_values or rec_key in seen_records:
            continue
        seen_values.add(key)
        seen_records.add(rec_key)
        values.append(value)
        selected.append((score, item, atom))
        if len(values) >= 8:
            break
    if len(values) < 2:
        return "", []
    if len(values) == 2:
        return f"{values[0]} and {values[1]}", selected
    return ", ".join(values[:-1]) + f", and {values[-1]}", selected


_ENUM_QUERY_VERB_RE = re.compile(
    r"\b(?:enjoy|enjoys|like|likes|love|loves|do|does|done|has|have|pursue|pursues|"
    r"practice|practices|play|plays|visit|visited|been|traveled|travelled|know|knows|"
    r"read|reads|offered|received|given)\b", re.I)
_ENUM_QUERY_HEAD_RE = re.compile(
    r"\b(?:hobbies|interests|activities|sports|games|pastimes|passions|cities|countries|"
    r"places|towns|tricks|skills|books|novels|deals|endorsements|gifts|presents)\b", re.I)
# Query verb -> the claim-predicate family that answers it: 'which cities has Jon VISITED'
# selects visit-family claims, never like-family ones. Query verbs outside every family
# (do/does/has) fall back to the flat union.
_ENUM_VERB_FAMILIES: tuple[frozenset, ...] = (
    frozenset({"enjoy", "enjoys", "enjoyed", "like", "likes", "liked", "love", "loves",
               "loved", "into"}),
    frozenset({"practice", "practices", "practiced", "play", "plays", "played",
               "pursue", "pursues", "pursued"}),
    frozenset({"visit", "visits", "visited", "been", "travel", "travels", "traveled",
               "travelled", "went", "toured"}),
    frozenset({"know", "knows", "knew", "taught", "teach", "learned", "learnt"}),
    frozenset({"read", "reads", "reread", "saw", "seen", "watched", "wrote", "written"}),
    frozenset({"offered", "received", "given", "gotten", "promised", "awarded"}),
)
_ENUM_VERB_FAMILY = frozenset().union(*_ENUM_VERB_FAMILIES) | {"do", "does", "did", "done"}


def _enum_predicate_family_for_query(query: str) -> frozenset:
    """Union of families whose verbs appear in the query; empty -> flat fallback."""
    qverbs = set(re.findall(r"[a-z']+", (query or "").lower()))
    allowed: set = set()
    for family in _ENUM_VERB_FAMILIES:
        if qverbs & family:
            allowed |= family
    return frozenset(allowed)
_ENUM_OBJECT_HEAD_STOP = {
    "at", "he", "her", "his", "i", "in", "it", "my", "on", "our", "she", "that", "their",
    "they", "this", "we", "you", "your",
}
_ENUM_OBJECT_JUNK = {
    "awesome", "cool", "definitely", "fine", "good", "great", "great job", "nice", "no",
    "ok", "okay", "really", "similarly", "sure", "thank you", "thanks", "well", "wow", "yes",
}


def _claim_enumeration_answer(query: str, atoms: list[tuple[float, object, str]]) -> tuple[str, list[tuple[float, object, str]]]:
    """Collector-rewrite step 1: enumeration answers come from TIER-1 CLAIMS.

    A typed claim already carries subject + predicate + object + a verbatim proof atom
    extracted once at write time; 'what hobbies does Dave enjoy?' is a SELECT over claims
    whose subject matches the question's person and whose predicate sits in the question
    verb's family - each returned object carries its own proof atom, so the composition
    verifies per item. Replaces per-query regex re-parsing of raw text (the junk factories).
    Fewer than two credible objects falls through to the legacy path."""
    ro = _ro()
    q = query or ""
    if not (_ENUM_QUERY_HEAD_RE.search(q) and _ENUM_QUERY_VERB_RE.search(q)):
        return "", []
    if re.match(r"\s*how\s+(?:many|much)\b", q, re.I):
        # A count question owns its own operator; enumerating the members here would
        # answer 'how many books' with the titles instead of the number.
        return "", []
    people = ro._query_people(q)
    person_terms = {t.lower() for t in ro._terms(people[0])} if people else set()
    allowed_preds = _enum_predicate_family_for_query(q) or _ENUM_VERB_FAMILY
    values: list[str] = []
    selected: list[tuple[float, object, str]] = []
    seen_vals: set[str] = set()
    for score, item, atom in atoms[:80]:
        if not isinstance(item, ClaimRecord):
            continue
        pred = str(getattr(item, "predicate", "") or "").lower()
        if not ({w for w in re.findall(r"[a-z']+", pred)} & allowed_preds):
            continue
        subject = str(getattr(item, "subject", "") or "").lower()
        if person_terms and not (person_terms & {t.lower() for t in ro._terms(subject)}):
            continue
        obj = ro._clean(str(getattr(item, "object", "") or ""))
        words = obj.split()
        low = obj.lower()
        if (not obj or len(obj) < 4 or len(words) > 5
                or low in _ENUM_OBJECT_JUNK
                or words[0].lower() in _ENUM_OBJECT_HEAD_STOP):
            continue
        # Place-name heads (cities/countries/towns) enumerate PROPER NOUNS only: the visit
        # family also carries 'charity thing'/'event' objects that must never appear in a
        # which-cities answer.
        if (re.search(r"\b(?:cities|countries|towns)\b", q, re.I)
                and not obj[:1].isupper()):
            continue
        key = ro._norm_key(obj)
        if key in seen_vals:
            continue
        seen_vals.add(key)
        values.append(obj)
        selected.append((score, item, atom))
        if len(values) >= 8:
            break
    if len(values) < 2:
        return "", []
    if len(values) == 2:
        return f"{values[0]} and {values[1]}", selected
    return ", ".join(values[:-1]) + f", and {values[-1]}", selected


_ORDINAL_ANCHOR_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth)\s+([a-z][a-z'-]{2,})\b", re.I)


def _ordinal_anchor_slot_answer(query: str, atoms: list[tuple[float, object, str]]) -> tuple[str, list[tuple[float, object, str]]]:
    """'What game was the SECOND tournament based on?' - conversations self-label ordinals
    ('Last week I won my second tournament!'), so the labeled occurrence is findable exactly,
    and the asked slot lives in the SAME record's dialogue. Answer the TitleCase phrase that
    directly modifies the anchor noun there ('the local Street Fighter TOURNAMENT'); no such
    phrase in the anchor record means fail closed, never a value from another occurrence."""
    ro = _ro()
    q = query or ""
    m = _ORDINAL_ANCHOR_RE.search(q)
    wh = re.search(r"\b(?:what|which)\s+([a-z][a-z'-]{2,})\b", q, re.I)
    if not m or not wh:
        return "", []
    ordinal, noun = m.group(1).lower(), m.group(2).lower()
    noun_key = ro._count_term_key(noun)
    anchor: tuple[float, object, str] | None = None
    for score, item, atom in atoms[:60]:
        low = atom.lower()
        if ordinal in low and noun_key in {ro._count_term_key(t) for t in ro._terms(atom)}:
            anchor = (score, item, atom)
            break
    if anchor is None:
        return "", []
    score, item, atom = anchor
    source = str(getattr(item, "text", "") or "")
    if not source:
        rec_text = getattr(item, "value", "") or ""
        source = str(rec_text)
    # the TitleCase phrase directly modifying the anchor noun, searched over the WHOLE anchor
    # record (the slot is often stated a turn or two after the self-labeled anchor)
    pat = re.compile(
        rf"((?:[A-Z][\w:'-]+)(?:\s+[A-Z][\w:'-]+){{0,3}})\s+{re.escape(noun)}", )
    hay = source or atom
    best = ""
    for mm in pat.finditer(hay):
        cand = ro._clean(mm.group(1))
        head = cand.split()[0].lower() if cand else ""
        if not cand or head in {"the", "a", "an", "my", "our", "his", "her", "their", "i"}:
            continue
        best = cand
        break
    if not best:
        return "", []
    return best, [(score, item, atom)]


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
