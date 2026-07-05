"""Claim-tier QA operators: dialogue Q->A crystal matching, premise-affinity inference, and
experience-location extraction.

These operators are generic by construction: they classify question SHAPE and match against claim
metadata (recorded dialogue questions, affinity predicates, action stems). They must never branch
on benchmark sample ids, fixed questions, or dataset entities.
"""
from __future__ import annotations

import re
import time

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
        # A greeting-only crystal ('Hey Noor, thanks!') answers nothing.
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
    """A yes/no question whose proposition a memory literally asserts ('Is my brother using the
    same budgeting method as me?' <- 'my brother is actually on the same budgeting app as me now')
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
        if (isinstance(item, ClaimRecord)
                and (getattr(item, "filters", None) or {}).get("untyped") == "1"):
            # Junk-quarantined fallback claims never contribute enumeration text.
            continue
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
# Query verb -> the claim-predicate family that answers it: 'which cities has Wei VISITED'
# selects visit-family claims, never like-family ones. Query verbs outside every family
# (do/does/has) fall back to the flat union.
_ENUM_VERB_FAMILIES: tuple[frozenset, ...] = (
    frozenset({"enjoy", "enjoys", "enjoyed", "like", "likes", "liked", "love", "loves",
               "loved", "into"}),
    # tried/started/took extend the pursue family for the write-side activity claims
    # (claim_extraction._activity_claims_from_atom) -- one shared table, so write and
    # read cannot drift.
    frozenset({"practice", "practices", "practiced", "play", "plays", "played",
               "pursue", "pursues", "pursued", "tried", "started", "took"}),
    frozenset({"visit", "visits", "visited", "been", "travel", "travels", "traveled",
               "travelled", "went", "toured", "go", "goes", "going"}),
    frozenset({"know", "knows", "knew", "taught", "teach", "learned", "learnt"}),
    frozenset({"read", "reads", "reread", "saw", "seen", "watched", "wrote", "written"}),
    frozenset({"offered", "received", "given", "gotten", "promised", "awarded"}),
)
_ENUM_VERB_FAMILY = frozenset().union(*_ENUM_VERB_FAMILIES) | {"do", "does", "did", "done"}
# The families under which write-side ACTIVITY claims (filters enum_fact=1,
# action=activity) may be selected: the query must EXPLICITLY use an enjoy/pursue
# family verb. Bare do-support ('what ... does X do?') is NOT an activity query --
# nearly every English wh-question carries do-support, so keying on it invented
# acquisitions and one-off events as hobbies.
_ACTIVITY_QUERY_FAMILIES = _ENUM_VERB_FAMILIES[0] | _ENUM_VERB_FAMILIES[1]


def _enum_fact_selectable(filters: dict, pred_tokens: set[str],
                          resolved_families: frozenset) -> bool:
    """Selection rule for write-side single-fact claims (filters enum_fact=1): the
    query must have RESOLVED a predicate family (never the flat do-support fallback),
    and the claim must match it -- activity claims answer enjoy/pursue-family queries;
    every other action type (acquire, attempt, ...) must match on its own predicate
    lemma, so an acquisition can never enumerate sideways into a gifts question."""
    if not resolved_families:
        return False
    if filters.get("action") == "activity":
        return bool(resolved_families & _ACTIVITY_QUERY_FAMILIES)
    return bool(pred_tokens & resolved_families)


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


def _enum_query_head_key(query: str) -> str:
    ro = _ro()
    q = query or ""
    head_m = _ENUM_QUERY_HEAD_RE.search(q)
    qverbs = set(re.findall(r"[a-z']+", q.lower()))
    episodic = _ENUM_VERB_FAMILIES[2] | _ENUM_VERB_FAMILIES[4] | _ENUM_VERB_FAMILIES[5]
    return (ro._count_term_key(head_m.group(0).lower())
            if head_m and not (qverbs & episodic) else "")


def _enum_list_label_hit(item: ClaimRecord, query_head_key: str) -> bool:
    ro = _ro()
    filters = getattr(item, "filters", None) or {}
    return bool(
        query_head_key
        and filters.get("list") == "item"
        and query_head_key in {
            ro._count_term_key(t)
            for t in re.findall(r"[a-z][\w'-]*", str(filters.get("list_label") or "").lower())
        }
    )


def _credible_enum_object(obj: str, list_label_hit: bool, q: str) -> bool:
    """The per-item enum floor, shared verbatim between the direct enumeration and the
    completion sweep (pure refactor of the inline block)."""
    words = obj.split()
    low = obj.lower()
    if "," in obj:
        # A comma inside a single enum "item" is a LIST shape, not an item: the
        # extractor's whole-list aggregate claim ('gardening, reading, and baking')
        # otherwise re-enters the enumeration as a fourth member, double-counting
        # every real item in a garbled verified answer.
        return False
    if (not obj or len(obj) < (3 if list_label_hit else 4) or len(words) > 5
            or low in _ENUM_OBJECT_JUNK
            or words[0].lower() in _ENUM_OBJECT_HEAD_STOP
            or words[-1].lower() in {"to", "of", "in", "on", "at", "with", "for",
                                     "from", "or", "and", "but", "the", "a", "an"}):
        return False
    # Place-name heads (cities/countries/towns) enumerate PROPER NOUNS only: the visit
    # family also carries 'charity thing'/'event' objects that must never appear in a
    # which-cities answer.
    if (re.search(r"\b(?:cities|countries|towns)\b", q, re.I)
            and not obj[:1].isupper()):
        return False
    return True


def _claim_enumeration_values(
    query: str, atoms: list[tuple[float, object, str]],
) -> tuple[bool, list[str], list[tuple[float, object, str]]]:
    """(shape_ok, values, selected): the SELECT over typed claims, without the <2-values
    fall-through (evaluated by the caller AFTER the completion sweep)."""
    ro = _ro()
    q = query or ""
    if not (_ENUM_QUERY_HEAD_RE.search(q) and _ENUM_QUERY_VERB_RE.search(q)):
        return False, [], []
    if re.match(r"\s*how\s+(?:many|much)\b", q, re.I):
        # A count question owns its own operator; enumerating the members here would
        # answer 'how many books' with the titles instead of the number.
        return False, [], []
    if _ADVICE_REQUEST_RE.search(q):
        # Recommendation lists are speaker-attributed dialogue, not disposition claims;
        # the flat-union predicate fallback composed junk here (verified-wrong on a live
        # dev row). The reader/speaker paths own these.
        return False, [], []
    people = ro._query_people(q)
    person_terms = {t.lower() for t in ro._terms(people[0])} if people else set()
    resolved_families = _enum_predicate_family_for_query(q)
    allowed_preds = resolved_families or _ENUM_VERB_FAMILY
    query_head_key = _enum_query_head_key(q)
    values: list[str] = []
    selected: list[tuple[float, object, str]] = []
    seen_vals: set[str] = set()
    for score, item, atom in atoms[:80]:
        if not isinstance(item, ClaimRecord):
            continue
        pred = str(getattr(item, "predicate", "") or "").lower()
        filters = getattr(item, "filters", None) or {}
        if filters.get("event") == "dated":
            # Event-dated family claims are when-question evidence, never enum items
            # (same spirit as the list/naming exclusions).
            continue
        if filters.get("untyped") == "1":
            # Junk-quarantined fallback claims never contribute answer text.
            continue
        pred_tokens = {w for w in re.findall(r"[a-z']+", pred)}
        list_label_hit = _enum_list_label_hit(item, query_head_key)
        if filters.get("enum_fact") == "1":
            # Write-side single-fact claims never ride the flat do-support fallback.
            if not _enum_fact_selectable(filters, pred_tokens, resolved_families):
                continue
        elif not list_label_hit and not (pred_tokens & allowed_preds):
            continue
        subject = str(getattr(item, "subject", "") or "").lower()
        if person_terms and not (person_terms & {t.lower() for t in ro._terms(subject)}):
            continue
        obj = ro._clean(str(getattr(item, "object", "") or ""))
        if not _credible_enum_object(obj, list_label_hit, q):
            continue
        key = ro._norm_key(obj)
        if key in seen_vals:
            continue
        seen_vals.add(key)
        values.append(obj)
        selected.append((score, item, atom))
        if len(values) >= 8:
            break
    return True, values, selected


def _format_enum_values(values: list[str]) -> str:
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


# Plural nouns that DENOTE a bridged possession entity ("<Person>'s pets"): the bridge
# may only fire on these heads, never on the question's own enumeration head noun.
_BRIDGE_POSSESSION_HEADS = frozenset({
    "pets", "dogs", "cats", "puppies", "kittens", "birds", "parrots", "rabbits",
    "hamsters", "horses", "animals", "kids", "children", "sons", "daughters",
})


def _drop_covered_aggregates(
    values: list[str], selected: list[tuple[float, object, str]],
) -> tuple[list[str], list[tuple[float, object, str]]]:
    """An 'A and B' object whose split members are all already items of their own is
    the extractor's whole-list aggregate claim wearing an item costume -- keeping it
    double-counts every member. (Comma-shaped aggregates are already rejected by the
    per-item floor; this catches the comma-free two-member shape.) Trailing supports
    beyond the value list (bridge claims) are preserved."""
    ro = _ro()
    keys = [ro._norm_key(v) for v in values]
    keyset = set(keys)
    keep_v: list[str] = []
    keep_s: list[tuple[float, object, str]] = []
    for i, v in enumerate(values):
        parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", v)
                 if p.strip()]
        if len(parts) >= 2:
            part_keys = {ro._norm_key(p) for p in parts}
            if part_keys and part_keys <= (keyset - {keys[i]}):
                continue
        keep_v.append(v)
        keep_s.append(selected[i])
    return keep_v, keep_s + selected[len(values):]


def _enum_completion_sweep(
    query: str,
    values: list[str],
    selected: list[tuple[float, object, str]],
    claim_pool: list,
) -> tuple[list[str], list[tuple[float, object, str]], bool]:
    """Sibling union with per-item typed provenance, then the completeness gate.

    Union sources, strictly widening but always typed: (1) same list_id (recovers
    siblings the scored/truncated atom window dropped), (2) same list_label + subject
    across list_ids, (3) lemma-compatible predicate family + subject (including the
    write-side activity claims), (4) a conservative owner->possession entity bridge
    ("<Person>'s <plural-noun>"). Every candidate passes the SAME per-item floor as the
    direct enumeration. Returns (values, selected, complete_ok): complete_ok=False means
    a selected list is provably missing members (invalidated/expired siblings) -- the
    enum path must DECLINE rather than ship a known-incomplete enumeration verified."""
    ro = _ro()
    q = query or ""
    values = list(values)
    selected = list(selected)
    pool = [c for c in (claim_pool or []) if isinstance(c, ClaimRecord)]
    if not pool:
        return values, selected, True
    now_ts = time.time()
    seen = {ro._norm_key(v) for v in values}
    people = ro._query_people(q)
    person_terms = {t.lower() for t in ro._terms(people[0])} if people else set()
    resolved_families = _enum_predicate_family_for_query(q)
    family_union = resolved_families or _ENUM_VERB_FAMILY
    query_head_key = _enum_query_head_key(q)

    def _subject_terms(claim: ClaimRecord) -> set[str]:
        return {t.lower() for t in ro._terms(str(claim.subject or ""))}

    def _label_keys(claim: ClaimRecord) -> set[str]:
        return {ro._count_term_key(t) for t in re.findall(
            r"[a-z][\w'-]*", str((claim.filters or {}).get("list_label") or "").lower())}

    def _invalidated(claim: ClaimRecord) -> bool:
        inv = getattr(claim, "invalid_at", None)
        return inv is not None and inv <= now_ts

    def _try_add(claim: ClaimRecord) -> bool:
        filters = claim.filters or {}
        if filters.get("untyped") == "1" or filters.get("event") == "dated":
            return False
        if _invalidated(claim):
            # Defense in depth: the production pool is pre-filtered by
            # active_claims_at, but the sweep must never resurrect a retracted
            # sibling handed to it by a less careful caller.
            return False
        obj = ro._clean(str(claim.object or ""))
        if not _credible_enum_object(obj, _enum_list_label_hit(claim, query_head_key), q):
            return False
        key = ro._norm_key(obj)
        if not key:
            return False
        if key in seen:
            return True                      # content already covered by another item
        if len(values) >= 8:
            return False
        seen.add(key)
        values.append(obj)
        selected.append((0.5, claim, claim.proof_atom or str(claim.value or "")))
        return True

    def _family_selectable(claim: ClaimRecord) -> bool:
        filters = claim.filters or {}
        pred_tokens = set(re.findall(r"[a-z']+", str(claim.predicate or "").lower()))
        if filters.get("enum_fact") == "1":
            # Same rule as the direct SELECT: write-side single-fact claims need an
            # explicitly resolved query family plus an action-compatible match, never
            # the flat do-support fallback.
            return _enum_fact_selectable(filters, pred_tokens, resolved_families)
        if pred_tokens & family_union:
            return True
        return _enum_list_label_hit(claim, query_head_key)

    def _family_union_pass(subject_terms: set[str]) -> None:
        for claim in pool:
            filters = claim.filters or {}
            if filters.get("untyped") == "1" or filters.get("event") == "dated":
                continue
            if not _family_selectable(claim):
                continue
            if subject_terms and not (subject_terms & _subject_terms(claim)):
                continue
            _try_add(claim)

    sel_claims = [item for _s, item, _a in selected if isinstance(item, ClaimRecord)]

    # (1) same list_id: pull ALL pool siblings of every selected list item.
    sel_list_ids = {str((c.filters or {}).get("list_id"))
                    for c in sel_claims if (c.filters or {}).get("list") == "item"}
    for lid in sorted(sel_list_ids):
        members = [c for c in pool if (c.filters or {}).get("list_id") == lid]
        for claim in sorted(members,
                            key=lambda c: int((c.filters or {}).get("list_index") or 0)):
            _try_add(claim)

    # (2) same list_label + subject across list_ids (other sessions' versions).
    sel_label_keys: set[str] = set()
    sel_subject_terms: set[str] = set()
    for claim in sel_claims:
        if (claim.filters or {}).get("list") == "item":
            sel_label_keys |= _label_keys(claim)
            sel_subject_terms |= _subject_terms(claim)
    if sel_label_keys:
        for claim in pool:
            if (claim.filters or {}).get("list") != "item":
                continue
            if not (_label_keys(claim) & sel_label_keys):
                continue
            subj = _subject_terms(claim)
            if sel_subject_terms and subj and not (subj & sel_subject_terms):
                continue
            _try_add(claim)

    # (3) lemma-compatible predicate family + subject. ONLY for queries that name a
    # person: with no subject gate this pass would union every family-matching claim
    # of every speaker in the namespace-wide pool ('What hobbies does she enjoy?'
    # must not merge two people's hobbies into one verified answer).
    if person_terms:
        _family_union_pass(person_terms)

    # (4) entity bridge, conservative: "<Person>'s <possession-plural>" / "<Person> and
    # (his|her|their) <possession-plural>". The plural noun must DENOTE the bridged
    # entities themselves (pets/kids-class possession heads) -- never the question's own
    # enumeration head ('Priya's hobbies' asks about Priya, not about an entity Priya
    # owns) -- and the query must have resolved a predicate family, so bridged claims
    # match the asked family rather than the flat fallback. Bridged subjects are
    # TitleCase possessions of the named person; each bridged item still carries its own
    # claim support, and the bridge claim rides along as an additional support.
    bridge_m = (re.search(r"\b([A-Z][\w'-]+)'s\s+([a-z][\w'-]*s)\b", q)
                or re.search(r"\b([A-Z][\w'-]+)\s+and\s+(?:his|her|their)\s+([a-z][\w'-]*s)\b", q))
    if (bridge_m and person_terms and resolved_families
            and bridge_m.group(2).lower() in _BRIDGE_POSSESSION_HEADS):
        bridged: list[ClaimRecord] = []
        for claim in pool:
            if len(bridged) >= 3:
                break
            if not (person_terms & _subject_terms(claim)):
                continue
            filters = claim.filters or {}
            pred_tokens = set(re.findall(r"[a-z']+", str(claim.predicate or "").lower()))
            if not (pred_tokens & {"has", "have", "had", "adopted", "got", "gotten",
                                   "owns", "own", "named"} or filters.get("naming")):
                continue
            obj = str(claim.object or "").strip()
            if not re.fullmatch(r"[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2}", obj):
                continue
            bridged.append(claim)
        for bridge_claim in bridged:
            n_before = len(values)
            _family_union_pass({t.lower() for t in ro._terms(str(bridge_claim.object))})
            if len(values) > n_before:
                selected.append((0.5, bridge_claim,
                                 bridge_claim.proof_atom or str(bridge_claim.value or "")))

    # Completeness gate: every list represented in the final selection must have every
    # position PRESENT AND LIVE in the pool. A position with no surviving row (its
    # sibling was invalidated or dropped) means the enumeration is provably missing a
    # member and must not ship verified. A position whose row IS in the pool but was
    # deliberately junk-filtered by the per-item floor, or truncated by the 8-value
    # cap, is OUR OWN read-side selection -- declining there would forfeit
    # previously-correct partial answers, so the gate lets those through.
    final_lids = {str((c.filters or {}).get("list_id"))
                  for _s, c, _a in selected
                  if isinstance(c, ClaimRecord) and (c.filters or {}).get("list") == "item"}
    for lid in final_lids:
        members = [c for c in pool if (c.filters or {}).get("list_id") == lid]
        list_size = 0
        live: set[int] = set()
        for claim in members:
            filters = claim.filters or {}
            try:
                list_size = max(list_size, int(filters.get("list_size") or 0))
            except (TypeError, ValueError):
                pass
            try:
                idx = int(filters.get("list_index"))
            except (TypeError, ValueError):
                continue
            if not _invalidated(claim):
                live.add(idx)
        if list_size and len(live & set(range(list_size))) < list_size:
            return values, selected, False
    return values, selected, True


def _claim_enumeration_answer(
    query: str, atoms: list[tuple[float, object, str]],
    claim_pool: list | None = None,
) -> tuple[str, list[tuple[float, object, str]]]:
    """Collector-rewrite step 1: enumeration answers come from TIER-1 CLAIMS.

    A typed claim already carries subject + predicate + object + a verbatim proof atom
    extracted once at write time; 'what hobbies does Farid enjoy?' is a SELECT over claims
    whose subject matches the question's person and whose predicate sits in the question
    verb's family - each returned object carries its own proof atom, so the composition
    verifies per item. Replaces per-query regex re-parsing of raw text (the junk factories).
    With a claim_pool (claim backend only), the completion sweep unions typed siblings the
    scored window missed and DECLINES known-incomplete lists; without one, behavior is
    byte-identical to the pre-sweep path. Fewer than two credible objects falls through to
    the legacy path."""
    shape_ok, values, selected = _claim_enumeration_values(query, atoms)
    if not shape_ok:
        return "", []
    if claim_pool is not None:
        values, selected, complete_ok = _enum_completion_sweep(
            query, values, selected, claim_pool)
        if not complete_ok:
            return "", []
    values, selected = _drop_covered_aggregates(values, selected)
    if len(values) < 2 or not selected:
        return "", []
    return _format_enum_values(values), selected


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
