"""Source-backed claim extraction helpers for SMQE consolidation."""
from __future__ import annotations

import calendar
import hashlib
import re
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Iterable, Optional

from eidetic import preferences
from eidetic.textseg import SENTENCE_SPLIT_RE
from eidetic.models import ClaimRecord, MemoryRecord


_CLAIM_TYPES = {"quantity", "state", "event", "interval", "table", "preference"}
_NUM_WORD_PATTERN = "eleven|twelve|three|seven|eight|zero|four|five|nine|one|two|six|ten|an|a"
_MONEY_AMOUNT_RE = re.compile(
    r"(?:[$€£]\s*\d+(?:,\d{3})*(?:\.\d+)?|"
    rf"\b(?:\d+(?:,\d{{3}})*(?:\.\d+)?|{_NUM_WORD_PATTERN})\s*(?:dollars?|usd|bucks|€|£)\b)",
    re.I,
)
_DURATION_AMOUNT_RE = re.compile(
    rf"\b(?:\d+(?:\.\d+)?|{_NUM_WORD_PATTERN})[-\s]*(?:hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?)\b",
    re.I,
)


@lru_cache(maxsize=4096)
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
        pieces.extend(s.strip() for s in SENTENCE_SPLIT_RE.split(line) if s.strip())
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
    speaker = _speaker_for_atom(rec, atom)
    if speaker and re.search(r"\b(?:i|we|my|our|i've|i'm|i'd|i'll|we've|we're|we'd|we'll)\b", atom or "", re.I):
        return speaker
    m = re.match(r"\s*(user|assistant|system|human|ai)\s*:\s*", atom, re.I)
    if m:
        return m.group(1).lower()
    m = re.match(r"\s*([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\b", atom)
    if m:
        subject = m.group(1)
        # Sentence-initial adverbs/imperatives/interjections are not subjects ('Remember to
        # water...', 'Simply add...') - they polluted the claim tier the enumerator reads.
        _NONSUBJECT = {
            "also", "anyway", "congrats", "congratulations", "hello", "hey", "hi", "no",
            "ok", "okay", "plus", "remember", "simply", "sure", "thanks", "well", "wow",
            "yeah", "yes", "these", "those", "this", "that", "my", "our",
        }
        if speaker and subject.lower() in _NONSUBJECT:
            return speaker
        if subject.lower() in _NONSUBJECT:
            return rec.source or "memory"
        return subject
    return rec.source or "memory"


def _record_speaker_hint(rec: MemoryRecord) -> str:
    m = re.search(r"\b([A-Z][A-Za-z'_-]{1,32})\s*:", rec.text or "")
    return m.group(1) if m else ""


def _speaker_for_atom(rec: MemoryRecord, atom: str) -> str:
    needle = _norm(atom)
    body_needle = _norm(re.sub(r"^[A-Z][A-Za-z'_-]{1,32}:\s*", "", atom or ""))
    for m in re.finditer(r"\b([A-Z][A-Za-z'_-]{1,32})\s*:\s*([^\n]+)", rec.text or ""):
        body = _norm(m.group(2))
        if needle and (needle in body or needle in _norm(m.group(0)) or body_needle in body):
            return m.group(1)
    return _record_speaker_hint(rec)


def _predicate_for(atom: str) -> str:
    low = atom.lower()
    for pat in (
        r"\b(?:is|was|are|were|am)\s+([a-z][a-z0-9_-]+)",
        r"\b(?:prefer|prefers|preferred|favorite|favourite|like|likes|liked|love|loves|"
        r"enjoy|enjoys|enjoyed|avoid|avoids)\b",
        r"\b(?:went|visited|met|bought|attended|finished|started|left|arrived|scheduled|called|emailed)\b",
    ):
        m = re.search(pat, low)
        if m:
            return m.group(0).strip()
    terms = re.findall(r"[a-z0-9][a-z0-9_-]*", low)
    return " ".join(terms[:4])


def _object_for(atom: str) -> str:
    text = re.sub(r"^\s*(?:user|assistant|system|human|ai)\s*:\s*", "", atom, flags=re.I)
    m = re.search(
        r"\b(?:is|was|are|were|am|prefer|prefers|preferred|like|likes|liked|love|loves|"
        r"enjoy|enjoys|enjoyed|visited|bought|attended|at|in|to|from)\s+"
        r"(?:really\s+|also\s+|just\s+|going\s+)?([^.;!?]+)", text, re.I)
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


def _acquisition_claims_from_atom(rec: MemoryRecord, atom: str) -> list[ClaimRecord]:
    text = re.sub(r"\s+", " ", atom or "").strip()
    speaker = _speaker_for_atom(rec, text)
    if not text or not speaker:
        return []
    patterns = (
        r"\b(?:i|we)\s+(?P<verb>bought|purchased|got|received|picked\s+up)\s+(?P<object>[^.;!?]{2,90}?)(?=\s+(?:yesterday|today|last|this|recently|for|from|at|because|remind|reminds)\b|[.;!?]|$)",
        r"\b(?P<object>[^.;!?]{2,90}?)\s+(?:i|we)\s+(?P<verb>bought|purchased|got|received|picked\s+up)\b",
    )
    out: list[ClaimRecord] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            obj = _clean_relation_object(m.group("object"))
            if not obj:
                continue
            verb = "buy" if re.search(r"\b(?:bought|purchased|picked)\b", m.group("verb"), re.I) else "receive"
            claim = _claim_from_atom(rec, text, "event", predicate=verb, value=text)
            if claim is None:
                continue
            claim.subject = speaker
            claim.object = obj
            claim.filters = {"action": "acquire"}
            out.append(claim)
    return out


def _duration_answer_claims_from_text(rec: MemoryRecord) -> list[ClaimRecord]:
    atoms = _sentences(rec.text or "")
    out: list[ClaimRecord] = []
    for idx, question in enumerate(atoms[:-1]):
        if not re.search(r"\bhow\s+long\b", question, re.I):
            continue
        answer = atoms[idx + 1]
        m = _DURATION_AMOUNT_RE.search(answer)
        if not m:
            continue
        claim = _claim_from_atom(rec, answer, "quantity", predicate="duration answer", value=answer)
        if claim is None:
            continue
        claim.subject = _speaker_for_atom(rec, answer) or claim.subject
        claim.object = m.group(0).strip()
        claim.unit = re.sub(r"^.*\s+", "", claim.object).lower()
        claim.filters = {
            "question": re.sub(r"^[A-Z][A-Za-z'_-]{1,32}:\s*", "", question).strip(),
            "answer_type": "duration",
        }
        out.append(claim)
    return out


def _action_object_claims_from_atom(rec: MemoryRecord, atom: str) -> list[ClaimRecord]:
    text = re.sub(r"\s+", " ", atom or "").strip()
    speaker = _speaker_for_atom(rec, text)
    if not text or not speaker:
        return []
    out: list[ClaimRecord] = []
    patterns = (
        r"\b(?:i|we)\s+(?:(?:just|also|finally|recently|officially)\s+){0,2}(?P<verb>[a-z][a-z'-]{2,}(?:ed|t))\s+(?P<object>[^.;!?]{3,90}?)(?=\s+(?:last|this|recently|because|while|when|where|which|that|so|together|again|alone|yesterday|today|ago|now|(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:days?|weeks?|months?|years?)|\d)\b|[.;!?]|$)",
        r"\b(?:i|we)\s+(?:(?:just|also|finally|recently|officially)\s+){0,2}(?P<verb>[a-z][a-z'-]{2,}(?:ed|t))\s+(?:at|in|to|from)\s+(?P<object>[^.;!?]{3,90}?)(?=\s+(?:last|this|recently|because|while|when|where|which|that|so)\b|[.;!?]|$)",
        r"\b(?:i|we)\s+(?P<verb>teamed|linked|partnered|paired)\s+up\s+with\s+(?P<object>[^.;!?]{3,60}?)(?=\s+(?:last|this|for|and|because|when)\b|[.;!?]|$)",
        r"\b(?:my|our)\s+(?P<object>[a-z][a-z' -]{2,40}?)\s+(?:just\s+|finally\s+|officially\s+){0,2}(?P<verb>dropped|released|launched|debuted|opened|started|arrived|premiered)\b",
        # Irregular pasts the ed|t suffix rule can never see (\'I read The Alchemist\',
        # \'we saw Hamilton\'), clitic-tolerant; and the offered/given passive (\'I\'ve been
        # offered a deal with Nike\') whose object is the enumerable fact.
        r"\b(?:i|we)(?:\'ve|\'d)?\s+(?:just\s+|also\s+|recently\s+|finally\s+)?(?P<verb>read|reread|wrote|saw|met|took)\s+(?P<object>[A-Z][^.;!?]{2,90}?)(?=\s+(?:last|this|recently|because|while|when|and|so|together|again|yesterday|today)\b|[.;!?]|$)",
        r"\b(?:i|we)(?:\'ve|\'d)?\s+been\s+(?P<verb>offered|given|promised)\s+(?P<object>[^.;!?]{3,90}?)(?=\s+(?:last|this|recently|because|while|when|so|by)\b|[.;!?]|$)",
        # Location visits that never phrase as went/been-to: 'I WAS IN Chicago', 'my TRIP TO
        # Seattle', 'we FLEW TO Paris'. TitleCase object keeps precision (places capitalize).
        r"\b(?:i|we)\s+(?P<verb>was|were)\s+in\s+(?P<object>[A-Z][\w' -]{2,40}?)(?=[.,;!?]|\s+(?:last|this|for|and|it|when|where|because|yesterday|today)\b)",
        r"\bmy\s+(?P<verb>trip)\s+to\s+(?P<object>[A-Z][\w' -]{2,40}?)(?=[.,;!?]|\s+(?:last|this|for|and|it|when|where|because|was|is)\b)",
        r"\b(?:i|we)(?:\'ve|\'d)?\s+(?:just\s+|also\s+)?(?P<verb>flew)\s+to\s+(?P<object>[A-Z][\w' -]{2,40}?)(?=[.,;!?]|\s+(?:last|this|for|and|when|because)\b)",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            obj = _clean_relation_object(m.group("object"))
            verb = m.group("verb").lower().replace("'", "")
            if verb in {"was", "were", "trip", "flew"}:
                verb = "visited"           # location visits normalize into the visit family
            if not obj or verb in {"asked", "answered", "said", "told", "wanted"}:
                continue
            claim = _claim_from_atom(rec, text, "event", predicate=verb, value=text)
            if claim is None:
                continue
            claim.subject = speaker
            claim.object = obj
            claim.filters = {"action": "object"}
            out.append(claim)
    return out


def _action_location_claims_from_atom(rec: MemoryRecord, atom: str) -> list[ClaimRecord]:
    text = re.sub(r"\s+", " ", atom or "").strip()
    speaker = _speaker_for_atom(rec, text)
    if not text or not speaker:
        return []
    out: list[ClaimRecord] = []
    patterns = (
        r"\b(?:i|we)\s+(?:(?:just|also|even|then|recently|finally|actually)\s+){0,4}(?P<activity>[a-z][a-z'-]{2,}(?:ed|t|ing))\s+(?:at|in|from)\s+(?P<object>[^.;!?]{3,90}?)(?=\s+(?:last|this|recently|because|while|when|where|which|that|so|with|and)\b|[.;!?]|$)",
        r"\b(?:i|we)(?:'ve|'d)?\s+(?:(?:just|also|even|then|recently|finally|actually)\s+){0,4}(?P<activity>went|go|gone|been|traveled|travelled|drove|drive|walked|hiked|visited|moved)\s+to\s+(?P<object>[^.;!?]{3,90}?)(?=\s+(?:last|this|recently|because|while|when|where|which|that|so|with|and|together|again|alone|yesterday|today)\b|[.;!?]|$)",
        r"\b(?:i|we)\s+(?:(?:just|also|even|then|recently|finally|actually)\s+){0,4}(?:went|go|gone|took|take|had|have)\b[^.;!?]{0,70}?\b(?P<activity>[a-z][a-z'-]{2,}ing)(?:\s+(?:trip|outing|session|visit|hike|walk|run|ride|day|weekend|vacation)){0,2}\s+(?:at|in|from)\s+(?P<object>[^.;!?]{3,90}?)(?=\s+(?:last|this|recently|because|while|when|where|which|that|so|with|and)\b|[.;!?]|$)",
        r"\b(?:my|our|his|her|their|the|this|that)\s+(?:[a-z][a-z'-]{2,}\s+){0,4}?(?P<activity>[a-z][a-z'-]{2,}ing)(?:\s+(?:trip|outing|session|visit|hike|walk|run|ride|day|weekend|vacation)){0,2}\s+(?:at|in|from)\s+(?P<object>[^.;!?]{3,90}?)(?=\s+(?:last|this|recently|because|while|when|where|which|that|so|with|and)\b|[.;!?]|$)",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            loc = _clean_relation_object(m.group("object"))
            activity = m.group("activity").lower().replace("'", "")
            if not loc or activity in {"being", "doing", "going", "having", "taking"}:
                continue
            claim = _claim_from_atom(rec, text, "event", predicate=activity, value=text)
            if claim is None:
                continue
            claim.subject = speaker
            claim.object = loc
            claim.filters = {"action": "location"}
            out.append(claim)
    return out


def _relation_object_claims_from_atom(rec: MemoryRecord, atom: str) -> list[ClaimRecord]:
    text = re.sub(r"\s+", " ", atom or "").strip()
    if not text:
        return []
    out: list[ClaimRecord] = []
    relation_patterns = (
        (
            r"\b(?:this|that|the|my|his|her|their|our)\s+"
            r"(?P<object>[A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z][A-Za-z0-9'/-]*){0,4})"
            r"\s+(?:is|was|'s)\b"
            r"(?P<context>[^.!?]{0,140}?\b(?:gift|present|souvenir|keepsake)\s+from\s+"
            r"(?P<source>[^.;!?]+))",
            "gift from {source}",
        ),
        (
            r"\b(?:have|has|got|received|own|owns|wear|wears|keep|keeps)\s+"
            r"(?:a|an|the|my|his|her|their|our|this|that)?\s*"
            r"(?P<object>[A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z][A-Za-z0-9'/-]*){0,4})"
            r"\s+(?=(?:that|which|,|-|is|was|'s)\b)"
            r"(?P<context>[^.!?]{0,140}?\b(?:gift|present|souvenir|keepsake)\s+from\s+"
            r"(?P<source>[^.;!?]+))",
            "gift from {source}",
        ),
        (
            r"\b(?P<source>[A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z][A-Za-z0-9'/-]*){0,4})"
            r"\s+(?:gave|gifted|sent)\s+(?:me|us|him|her|them)?\s*"
            r"(?:a|an|the|my|his|her|their|our|this|that)?\s*"
            r"(?P<object>[A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z][A-Za-z0-9'/-]*){0,4})",
            "gift from {source}",
        ),
    )
    for pattern, predicate_template in relation_patterns:
        for m in re.finditer(pattern, text, re.I):
            obj = _clean_relation_object(m.group("object"))
            if not obj:
                continue
            source = re.sub(r"\s+", " ", (m.groupdict().get("source") or "").strip(" .,:;!?"))
            if source.lower() in {"i", "we", "me", "us"}:
                source = _speaker_for_atom(rec, text) or source
            predicate = predicate_template.format(source=source).strip()
            claim = _claim_from_atom(rec, obj, "state", predicate=predicate, value=obj)
            if claim is None:
                continue
            subject = _subject_for(text, rec)
            if subject.lower() in {"this", "that", "the", "my"}:
                subject = _record_speaker_hint(rec) or subject
            claim.subject = subject
            claim.object = obj
            filters = {"relation": "gift"}
            if source:
                filters["source"] = source
            claim.filters = filters
            out.append(claim)
    return out


_SUPPORT_SOURCE_STOP = {
    "what", "who", "that", "this", "it", "they", "them", "there", "then", "i", "we",
    "you", "he", "she", "me", "us", "him", "everyone", "someone", "people",
}


def _support_relation_claims_from_atom(rec: MemoryRecord, atom: str) -> list[ClaimRecord]:
    text = re.sub(r"\s+", " ", atom or "").strip()
    if not text or text.endswith("?"):
        return []
    out: list[ClaimRecord] = []
    patterns = (
        r"\b(?:help|assistance|support)\s+from\s+(?:out\s+|our\s+|my\s+|his\s+|her\s+|their\s+)?"
        r"(?P<source>[A-Za-z][\w' -]{2,40}?)(?=[.,;!?]|\s+(?:when|while|because|and|but|so|it)\b|$)",
        r"\b(?:my|our|his|her|their)\s+(?P<source>[a-z][\w'-]{2,20})\s+"
        r"(?:[\w' ]{0,30}?\s+)?(?:used\s+to\s+)?help(?:ed|s)?\s+(?:my|our|us|me)\b",
        r"\b(?P<source>[A-Z][\w'-]{2,20})\s+(?:[\w' ]{0,30}?\s+)?(?:used\s+to\s+)?"
        r"help(?:ed|s)?\s+(?:my|our|us|me)\s+(?:family|out)\b",
    )
    for pat in patterns:
        for m in re.finditer(pat, text):
            source = re.sub(r"\s+", " ", m.group("source").strip(" .,:;!?"))
            source = re.sub(r"^(?:out|our|my|his|her|their|the|a|an)\s+", "", source, flags=re.I)
            if not source or source.lower() in _SUPPORT_SOURCE_STOP or len(source) < 3:
                continue
            claim = _claim_from_atom(rec, text, "state",
                                     predicate=f"support from {source}", value=source)
            if claim is None:
                continue
            claim.subject = _speaker_for_atom(rec, text) or claim.subject
            claim.object = source
            claim.filters = {"relation": "support", "source": source}
            out.append(claim)
    return out


# ------------------------------------------------------------------ event-date family
# A claim is emitted ONLY when the source sentence itself carries a date expression tied
# to the event verb/noun. No in-atom date = no claim (the record falls back to the
# existing paths). This is the structural fix for mention-time-vs-event-time and
# ambient-year selection -- never fall back to rec.valid_at here: the session timestamp
# masquerading as the event date is exactly the bug this family exists to kill.

_EVENT_NEGATION_RE = re.compile(
    r"\b(?:not|never|didn't|did\s+not|couldn't|won't|wouldn't|can't|failed\s+to|missed|"
    r"cancell?ed|postponed)\b", re.I)
_EVENT_HYPOTHETICAL_RE = re.compile(
    r"\b(?:hoping|hope\s+to|planning|plan(?:s|ned)?\s+to|might|would|want(?:s|ed)?\s+to|"
    r"thinking\s+of|wish\s+to|if)\b", re.I)
# Scheduled-but-possibly-unrealized markers: 'was SUPPOSED TO be in June 2020' states a
# plan that a later clause may cancel -- crystallizing the never-happened date is worse
# than declining, so the whole segment is skipped (fail closed to the legacy paths).
_EVENT_SCHEDULED_RE = re.compile(
    r"\b(?:supposed\s+to|going\s+to\s+be|originally|meant\s+to|rescheduled|"
    r"was\s+to\s+be|were\s+to\s+be)\b", re.I)
_MONTH_NUM: dict[str, int] = {name.lower(): i for i, name in enumerate(calendar.month_name) if i}
_MONTH_NUM.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if i})
_MONTH_NUM["sept"] = 9
# FULL month names only (abbreviations stay out of phrase alternations): the earlier
# len()>4 filter silently dropped May/June/July, so 'last week of May' fell through to
# the generic 'last week' branch and anchored to the SESSION month.
_MONTH_NAMES_RE = "|".join(m.lower() for m in calendar.month_name if m)
# Date-correction tail: 'on April 9, 2024, not April 1' -- the trailing 'not <date>'
# corrects a DATE, it does not negate the event, and must neither trip the negation
# guard nor survive as a matchable date expression.
_NOT_DATE_CORRECTION_RE = re.compile(
    rf"[,\s]+not\s+(?:on\s+|in\s+)?(?:the\s+)?"
    rf"(?:(?:{_MONTH_NAMES_RE})\b[\s,]*\d{{0,2}}(?:st|nd|rd|th)?(?:[\s,]+20\d{{2}})?|"
    rf"\d{{1,2}}(?:st|nd|rd|th)|20\d{{2}}(?:-\d{{2}}-\d{{2}})?)[^,;]*", re.I)
_WINDOW_PHRASE_RE = re.compile(
    rf"\b(?:(?:the\s+)?(?:first|last)\s+week\s+of\s+(?:{_MONTH_NAMES_RE})|"
    rf"last\s+(?:week|month|{_MONTH_NAMES_RE}))\b", re.I)


@lru_cache(maxsize=1)
def _event_date_patterns() -> tuple[re.Pattern, re.Pattern]:
    from . import event_identity as ei

    verbs = "|".join(sorted(ei.DATED_EVENT_VERB_LEMMAS, key=len, reverse=True))
    nouns = "|".join(sorted(ei.EVENT_NOUN_LEMMAS, key=len, reverse=True))
    verb_re = re.compile(
        rf"\b(?:i|we)(?:'ve|'d)?\s+(?:(?:just|also|both|all|finally|recently|officially)\s+){{0,2}}"
        rf"(?P<verb>{verbs})\b(?P<obj>[^.;!?]*)", re.I)
    noun_re = re.compile(
        rf"\b(?:(?P<owner>[A-Z][\w'-]+)'s\s+|(?:[Mm]y|[Oo]ur|[Tt]he)\s+)"
        rf"(?P<noun>{nouns})\b[^.;!?]{{0,60}}?\b(?:was|is|were|took\s+place|happened)\b")
    return verb_re, noun_re


def _event_clause_segments(body: str) -> list[str]:
    """Clause-proximity segmentation (the ambient-year fix): parenthetical relative
    clauses are EXCISED (their dates can never qualify), then the text splits at clause
    boundaries so the qualifying date must live with the verb/noun. Date-internal commas
    are protected so 'March 3, 2024' never splits."""
    from . import event_identity as ei
    from .record_ops import _DATE_RE

    dated_verbs = "|".join(sorted(ei.DATED_EVENT_VERB_LEMMAS, key=len, reverse=True))
    prot = _DATE_RE.sub(lambda m: m.group(0).replace(",", "\x00"), body or "")
    prot = re.sub(r",\s+(?:who|whom|which|where)\b[^,;]*", "", prot, flags=re.I)
    # Restrictive (no-comma) relative clauses carry OTHER entities' dates too: 'a kitten
    # that was born in March 2023', 'my cousin who moved to Denver in March 2021', and
    # contact clauses 'the bakery she opened in March 2021' must never date the main verb.
    prot = re.sub(r"\s+(?:who|whom|which|where)\s+[^,;]*", "", prot, flags=re.I)
    prot = re.sub(
        r"\s+that\s+(?:was|were|is|are|had|has|got|went|she|he|they)\b[^,;]*",
        "", prot, flags=re.I)
    prot = re.sub(
        r"(?<=[a-z])\s+(?:she|he|they)\s+(?:'d\s+|had\s+)?"
        r"(?:[a-z]+ed|met|flew|went|held|took|made|got|was|were|had|ran|built|"
        r"sold|bought)\b[^,;]*",
        "", prot, flags=re.I)
    segs = re.split(
        rf";\s+|,?\s+(?:and|but|then|so)\s+"
        rf"(?=(?:(?:yesterday|today|tonight|recently|earlier|later|finally|eventually)\s+|"
        rf"(?:last|this|next)\s+\w+\s+)?(?:i|we|my|our)\b)|"
        # Coordinated dated verb with a shared subject ('...adopted a puppy and VISITED
        # the vet yesterday'): the trailing date belongs to the second verb's clause.
        rf",?\s+(?:and|but|then|so)\s+(?=(?:{dated_verbs})\b)|"
        rf"\s+when\s+(?:i|we)\s+(?:was|were)\b|\s+back\s+in\b",
        prot, flags=re.I)
    return [s.replace("\x00", ",").strip() for s in segs if s and s.strip()]


def _segment_event_date(rec: MemoryRecord, seg: str) -> Optional[tuple[str, int, str]]:
    """(iso, precision, date_phrase) for the date expression the segment itself states.
    None when the segment carries no resolvable date -- callers must then emit nothing.
    Bare years are deliberately skipped: a year-only anchor would render as 'January YYYY'
    at month granularity (fabricated month); the generic candidate loop already answers
    bare years honestly."""
    from . import event_identity as ei
    from .record_ops import _DATE_RE, _relative_date_from_atom

    m = _DATE_RE.search(seg)
    if m:
        raw = m.group(0)
        if re.match(r"20\d{2}-\d{2}-\d{2}$", raw):
            return raw, ei.PRECISION_EXPLICIT, ""
        dm = re.match(r"([a-z]+)\s+(\d{1,2}),?\s+(20\d{2})$", raw, re.I)
        if dm:
            mon = _MONTH_NUM.get(dm.group(1).lower())
            if mon:
                try:
                    return (date(int(dm.group(3)), mon, int(dm.group(2))).isoformat(),
                            ei.PRECISION_EXPLICIT, "")
                except ValueError:
                    pass
        my = re.match(r"([a-z]+)\s+(20\d{2})$", raw, re.I)
        if my:
            mon = _MONTH_NUM.get(my.group(1).lower())
            if mon:
                return f"{int(my.group(2)):04d}-{mon:02d}-01", ei.PRECISION_WINDOW, raw
    try:
        ref = datetime.fromtimestamp(float(getattr(rec, "valid_at", None))).date()
    except (OSError, OverflowError, ValueError, TypeError):
        ref = None
    wm = re.search(rf"\b(first|last)\s+week\s+of\s+({_MONTH_NAMES_RE})\b", seg, re.I)
    if wm and ref is not None:
        # Present-tense copular ('my graduation IS the first week of March') reads as an
        # upcoming event; the past-only year inference would date it a full year early.
        # Fail closed instead of guessing the year.
        if re.search(r"\b(?:is|are)\s+(?:during\s+|in\s+)?(?:the\s+)?(?:first|last)\s+week\b",
                     seg, re.I):
            return None
        mon = _MONTH_NUM[wm.group(2).lower()]
        year = ref.year if mon <= ref.month else ref.year - 1
        day = 1 if wm.group(1).lower() == "first" else calendar.monthrange(year, mon)[1] - 6
        return date(year, mon, day).isoformat(), ei.PRECISION_WINDOW, wm.group(0)
    rel = _relative_date_from_atom(rec, seg)
    if rel:
        if re.match(r"\d{4}-\d{2}-\d{2}$", rel):
            try:
                rd = date(*[int(x) for x in rel.split("-")])
            except ValueError:
                return None
            # Past-surface verb + strictly-future resolved date is self-contradictory:
            # 'I moved the whole house in two weeks' is a DURATION, not a placement.
            if ref is not None and rd > ref:
                return None
            # Granularity honesty: '(a|N) year(s) ago' bounds neither day nor month --
            # emit nothing; week/month-granular phrases anchor at WINDOW precision so
            # format_answer renders month-only; only day-granular phrases keep the day.
            if re.search(r"\byears?\s+ago\b", seg, re.I) or re.search(
                    r"\bfor\s+(?:about\s+|over\s+|nearly\s+|almost\s+)?[\w]+\s+years?\b",
                    seg, re.I):
                return None
            gm = re.search(
                r"\b(?:[\w]+\s+(?:weeks?|months?)|a\s+fortnight)\s+ago\b|"
                r"\bfor\s+(?:about\s+|over\s+|nearly\s+|almost\s+)?[\w]+\s+(?:weeks?|months?)\b",
                seg, re.I)
            if gm:
                return rel, ei.PRECISION_WINDOW, gm.group(0)
            return rel, ei.PRECISION_RELATIVE_DAY, ""
        phrase_m = _WINDOW_PHRASE_RE.search(seg)
        phrase = phrase_m.group(0) if phrase_m else rel
        isos = re.findall(r"\d{4}-\d{2}-\d{2}", rel)
        if isos:
            # A range whose endpoints land in different months ('last week' spoken July 3
            # covers Jun 26-Jul 2) has no honest single-month anchor: decline and let the
            # legacy path answer the range verbatim.
            if len(isos) > 1 and isos[0][:7] != isos[-1][:7]:
                return None
            if ref is not None and isos[0] > ref.isoformat():
                return None
            return isos[0], ei.PRECISION_WINDOW, phrase
        mm = re.match(r"([A-Za-z]+)\s+(\d{4})$", rel)
        if mm and _MONTH_NUM.get(mm.group(1).lower()):
            return (f"{int(mm.group(2)):04d}-{_MONTH_NUM[mm.group(1).lower()]:02d}-01",
                    ei.PRECISION_WINDOW, phrase)
    return None


def _segment_date_pos(seg: str, phrase: str) -> Optional[int]:
    """Character position of the segment's date expression, for nearest-verb attribution
    when one clause carries several dated verbs."""
    from .record_ops import _DATE_RE

    if phrase:
        i = seg.lower().find(phrase.lower())
        if i >= 0:
            return i
    m = _DATE_RE.search(seg)
    if m:
        return m.start()
    m = re.search(
        r"\byesterday\b|\btoday\b|\btonight\b|\bago\b|\blast\s+\w+|"
        r"\bthis\s+(?:past\s+)?\w+|\bon\s+the\s+\d{1,2}", seg, re.I)
    return m.start() if m else None


def _event_date_object(raw: str) -> str:
    obj = re.sub(r"\s+", " ", (raw or "").strip())
    obj = re.sub(r"^(?:up|into|in|to|at|from|with|for)\s+", "", obj, flags=re.I)
    obj = re.split(
        r"\b(?:on|in|at|during|for|with|from|by|last|this|next|yesterday|today|tomorrow|"
        r"when|while|because|back|ago|and|but|so|"
        r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)"
        r"\s+(?:days?|weeks?|months?|years?))\b",
        obj, maxsplit=1, flags=re.I)[0]
    return _clean_relation_object(obj)


def _event_date_claims_from_atom(rec: MemoryRecord, atom: str) -> list[ClaimRecord]:
    text = re.sub(r"\s+", " ", atom or "").strip()
    if not text or text.endswith("?") or _is_info_seeking_question(text):
        return []
    from . import event_identity as ei
    from .record_ops import _is_future_polarity_atom

    # Write-time future/plan polarity guard: replicates for the claim path what the
    # generic loop enforces at read time ('tomorrow' added -- record_ops's marker set
    # predates day-resolution relatives).
    if _is_future_polarity_atom(text) or re.search(r"\btomorrow\b", text, re.I):
        return []
    body = re.sub(r"^[A-Z][A-Za-z'_-]{1,32}:\s*", "", text)
    # Quoted third-party speech is REPORTED, not the speaker's own event: 'Rosa told me
    # "I adopted a kitten on March 3, 2024"' must never become a first-person claim.
    body = re.sub(r'"[^"]*"|“[^”]*”', " ", body)
    verb_re, noun_re = _event_date_patterns()
    if not (verb_re.search(body) or noun_re.search(body)):
        return []
    speaker = _speaker_for_atom(rec, text)
    out: list[ClaimRecord] = []
    seen: set[tuple[str, str, str]] = set()

    def _emit(lemma: str, obj: str, subject: str, iso: str, precision: int,
              place: str, phrase: str, owner: str = "") -> None:
        key = (lemma, _norm(obj), iso)
        if key in seen:
            return
        seen.add(key)
        head = ei.obj_head(obj)
        if not head:
            # An empty object head ('We opened IT on June 1') is unanchorable evidence:
            # downstream object ties auto-pass on empty heads, so such a claim can
            # hijack unrelated instances. Emit nothing.
            return
        claim = _claim_from_atom(rec, text, "event", predicate=lemma, value=iso)
        if claim is None:
            return
        if subject:
            claim.subject = subject
        claim.object = obj
        filters: dict[str, Any] = {
            "event": "dated", "lemma": lemma, "obj_head": head,
            "event_date": iso, "date_precision": precision,
        }
        if place:
            filters["place"] = place
        if phrase:
            filters["date_phrase"] = phrase
        if owner:
            # Third-party possessive ('Mina's wedding'): the read side must never serve
            # this for a first-person question, nor the speaker's own event for an
            # owner-named question.
            filters["owner"] = owner
        claim.filters = filters
        out.append(claim)

    for seg in _event_clause_segments(body):
        # 'not April 1' after a stated date corrects the DATE; it is not event negation.
        seg = _NOT_DATE_CORRECTION_RE.sub("", seg)
        if _EVENT_NEGATION_RE.search(seg) or _EVENT_SCHEDULED_RE.search(seg):
            continue
        dated = _segment_event_date(rec, seg)
        if dated is None:
            continue  # no in-atom date tied to this clause -> no claim, ever
        iso, precision, phrase = dated
        place_m = re.search(
            rf"\b(?:in|at)\s+(?!(?i:{_MONTH_NAMES_RE})\b)"
            rf"([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){{0,2}})", seg)
        place = place_m.group(1) if place_m else ""
        cands: list[tuple[int, str, str, str, str]] = []
        for m in verb_re.finditer(seg):
            if _EVENT_HYPOTHETICAL_RE.search(seg[:m.start()]):
                continue
            lemma = ei.DATED_EVENT_VERB_LEMMAS.get(m.group("verb").lower(), "")
            if not lemma:
                continue
            cands.append((m.start("verb"), lemma, _event_date_object(m.group("obj")),
                          speaker, ""))
        for m in noun_re.finditer(seg):
            if _EVENT_HYPOTHETICAL_RE.search(seg[:m.start()]):
                continue
            noun = m.group("noun").lower()
            lemma = ei.EVENT_NOUN_LEMMAS.get(noun, "")
            if not lemma:
                continue
            owner = (m.group("owner") or "").strip()
            cands.append((m.start("noun"), lemma, noun, owner or speaker, owner))
        if len(cands) > 1:
            # One clause, one date, several dated verbs: the date belongs to the NEAREST
            # verb only ('I adopted a puppy and yesterday we visited the vet' must never
            # date the adoption with the vet visit's 'yesterday').
            dpos = _segment_date_pos(seg, phrase)
            if dpos is not None:
                cands = [min(cands, key=lambda c: abs(c[0] - dpos))]
        for _pos, lemma, obj, subject, owner in cands:
            _emit(lemma, obj, subject, iso, precision, place, phrase, owner)
    return out


def _clean_relation_object(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip(" .,:;!?"))
    value = re.sub(r"^[A-Z][A-Za-z'_-]{1,32}:\s*", "", value)
    value = re.sub(r"^(?:a|an|the|my|his|her|their|our|this|that|these|those)\s+", "", value, flags=re.I)
    value = re.split(r"\s+\b(?:i|we|that|which|is|was|from|for|with|because|but|after|before|today|yesterday|tomorrow|last|next|recently)\b", value, maxsplit=1, flags=re.I)[0]
    value = value.strip(" -")
    if len(value) < 3:
        return ""
    if re.fullmatch(r"(?:thing|things|item|items|one|ones|it|them|me|us|him|her|today|yesterday|tomorrow)", value, re.I):
        return ""
    return value[:120]


_INFO_SEEKING_QUESTION_RE = re.compile(
    r"^\s*(?:[A-Z][A-Za-z'_-]{1,32}:\s*)?"
    r"(?:what|when|where|which|who|whom|whose|why|how|do|does|did|is|are|was|were|have|has|had|"
    r"can|could|would|should|will|shall|any|anything)\b",
    re.I,
)


def _is_info_seeking_question(atom: str) -> bool:
    """True interrogatives carry no facts; rhetorical frames ("Remember when I got pre-approved
    for $400,000?") do and must still crystallize."""
    return bool(atom.rstrip().endswith("?") and _INFO_SEEKING_QUESTION_RE.match(atom or ""))


def _dialogue_answer_claims_from_text(rec: MemoryRecord) -> list[ClaimRecord]:
    """Q->A adjacency crystals: the sentence answering an in-conversation question carries the
    question's terms as filters, so paraphrased slot queries ("plans for the summer") match the
    stated answer even when the answer sentence itself never repeats the slot words."""
    atoms = _sentences(rec.text or "", limit=200)
    out: list[ClaimRecord] = []
    for idx, question in enumerate(atoms[:-1]):
        if not question.rstrip().endswith("?"):
            continue
        q_body = re.sub(r"^[A-Z][A-Za-z'_-]{1,32}:\s*", "", question).strip()
        if len(_norm(q_body)) < 12:
            continue
        answer = atoms[idx + 1]
        if answer.rstrip().endswith("?"):
            continue
        claim = _claim_from_atom(rec, answer)
        if claim is None:
            continue
        claim.subject = _speaker_for_atom(rec, answer) or claim.subject
        claim.filters = {**claim.filters, "question": q_body[:160], "dialogue": "answer"}
        out.append(claim)
    return out


_LIST_ITEM_JUNK = {
    "you", "me", "us", "him", "her", "them", "it", "that", "this", "these", "those",
    "one", "ones", "thing", "things", "stuff", "more", "lot", "lots", "etc", "everything",
    "anything", "something", "same", "i", "we", "he", "she", "they",
    "myself", "yourself", "herself", "himself", "ourselves", "themselves", "itself",
    "each", "both", "all", "some", "any",
}
_LIST_ITEM_STATE_WORDS = {
    "gone", "done", "over", "fine", "okay", "ok", "better", "worse", "great", "good",
    "bad", "hard", "easy", "back", "here", "there",
}
_LIST_ITEM_ADVERBS = {
    "thankfully", "finally", "luckily", "honestly", "hopefully", "sadly", "really",
    "mostly", "definitely", "probably", "quickly", "slowly", "recently", "lately",
    "actually", "eventually", "certainly", "surely",
}
_LIST_ITEM_TAIL_RE = re.compile(
    r"\s+\b(?:from|at|during|for|because|since|so|when|while|after|before|which|that|"
    r"who|where|last|next|yesterday|today|tomorrow|recently|too|though|although)\b.*$",
    re.I,
)
_LIST_ENUM_VERB_RE = (
    r"enjoy(?:s|ed)?|love(?:s|d)?|like(?:s|d)?|know(?:s)?|knew|learn(?:ed|t)?|"
    r"tried|do|does|did|can\s+do|play(?:s|ed)?|practice(?:s|d)?"
)


def _clean_list_item(raw: str) -> str:
    item = re.sub(r"\s+", " ", (raw or "").strip(" .,:;!?-"))
    item = re.sub(r"^(?:and|or|also|then|even|maybe|perhaps|probably|hopefully|definitely)\s+",
                  "", item, flags=re.I)
    item = re.sub(r"^(?:a|an|the|some|my|our|his|her|their|new|another)\s+", "", item, flags=re.I)
    item = _LIST_ITEM_TAIL_RE.sub("", item).strip(" .,:;!?-")
    if len(item) < 3 or len(item.split()) > 5:
        return ""
    if item.lower() in _LIST_ITEM_JUNK or item.split()[0].lower() in _LIST_ITEM_JUNK:
        return ""
    if item.split()[-1].lower() in {"to", "of", "in", "on", "at", "with", "for", "from",
                                    "or", "and", "but", "the", "a", "an"}:
        return ""
    if len(item.split()) == 1:
        low = item.lower()
        if low in _LIST_ITEM_STATE_WORDS or low in _LIST_ITEM_ADVERBS:
            return ""
        if len(low) >= 6 and low.endswith("ed"):
            return ""
    if item.isdigit():
        return ""
    return item[:80]


def _split_list_items(raw: str) -> list[str]:
    region = (raw or "").strip()
    parts = re.split(r",\s*(?:and\s+|or\s+)?|\s+and\s+|\s+as\s+well\s+as\s+", region, flags=re.I)
    items: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = _clean_list_item(part)
        key = _norm(item)
        if item and key not in seen:
            seen.add(key)
            items.append(item)
        if len(items) >= 8:
            break
    if "," not in region and not (
        len(items) == 2 and all(len(i.split()) <= 3 for i in items)
    ):
        return []
    return items


def _plural_label(label: str) -> str:
    label = re.sub(r"\s+", " ", (label or "").strip(" .,:;!?-"))
    label = re.sub(
        r"^(?:the|my|our|his|her|their|these|those|some|all|current|active|main|new)\s+",
        "", label, flags=re.I,
    )
    words = re.findall(r"[a-z][\w'-]*", label.lower())
    if not words or len(words) > 4:
        return ""
    last = words[-1]
    if not re.search(r"(?:s|es)$", last) or re.search(r"(?:ss|us|is)$", last):
        return ""
    return " ".join(words)


def _list_item_claims(rec: MemoryRecord, atom: str, label: str, verb: str,
                      items: list[str]) -> list[ClaimRecord]:
    if len(items) < 2:
        return []
    lid = hashlib.sha1(
        f"{rec.memory_id}|{label}|{verb}|{_norm(atom)}".encode()
    ).hexdigest()[:12]
    out: list[ClaimRecord] = []
    for idx, item in enumerate(items):
        claim = _claim_from_atom(rec, atom, "state", predicate=verb or label, value=item)
        if claim is None:
            continue
        claim.subject = _speaker_for_atom(rec, atom) or claim.subject
        claim.object = item
        claim.filters = {
            "list": "item", "list_id": lid, "list_label": label,
            "list_size": len(items), "list_index": idx,
        }
        out.append(claim)
    return out


def _list_item_claims_from_atom(rec: MemoryRecord, atom: str) -> list[ClaimRecord]:
    text = re.sub(r"\s+", " ", atom or "").strip()
    if not text or text.endswith("?"):
        return []
    body = re.sub(r"^[A-Z][A-Za-z'_-]{1,32}:\s*", "", text)
    out: list[ClaimRecord] = []

    m = re.search(
        r"^(?P<label>[\w' -]{2,60}?)\s+(?:are|were|include(?:s|d)?)\s+(?P<items>[^.;!?]+)",
        body, re.I,
    ) or re.search(
        r"^(?P<label>[\w' -]{2,60}?):\s+(?P<items>[^.;!?]+)", body,
    )
    if m:
        label = _plural_label(m.group("label"))
        if label:
            out.extend(_list_item_claims(
                rec, text, label, "", _split_list_items(m.group("items"))))

    for m in re.finditer(
        r"\b(?P<label>[a-z][\w-]*)\s+(?:like|such\s+as|including)\s+(?P<items>[^.;!?]+)",
        body, re.I,
    ):
        label = _plural_label(m.group("label"))
        if not label:
            continue
        out.extend(_list_item_claims(
            rec, text, label, "", _split_list_items(m.group("items"))))

    for clause in re.split(r",?\s+(?:and|but|then)\s+(?=(?:i|we)\b)", body, flags=re.I):
        for m in re.finditer(
            rf"\b(?:i|we)\s+(?:really\s+|also\s+|both\s+|all\s+)?"
            rf"(?P<verb>{_LIST_ENUM_VERB_RE})\s+(?P<items>[^.;!?]+)",
            clause, re.I,
        ):
            verb = re.sub(r"\s+", " ", m.group("verb").lower())
            region = re.split(r"\b(?:like|such\s+as|including)\b", m.group("items"),
                              maxsplit=1, flags=re.I)[0]
            region = re.split(r",?\s+(?:and\s+|but\s+|then\s+)?(?:i|we|he|she|they)\b",
                              region, maxsplit=1, flags=re.I)[0]
            out.extend(_list_item_claims(rec, text, "", verb, _split_list_items(region)))

    deduped: list[ClaimRecord] = []
    seen: set[tuple[str, str]] = set()
    for claim in out:
        key = (str(claim.filters.get("list_id")), _norm(claim.object))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped


_NAMED_CATEGORY_NOUNS = {
    "technique", "method", "diet", "challenge", "program", "workout", "routine",
    "strategy", "framework",
}
_TITLE_STOP_WORDS = {
    "i", "we", "he", "she", "they", "it", "the", "a", "an", "my", "our", "his", "her",
    "their", "this", "that", "so", "and", "but", "when", "then", "hey", "wow", "oh",
}
_VOCATIVE_LEAD_RE = re.compile(
    r"(?:^|[.!?,;]\s*)(?i:hey|hi|hello|yo|wow|oh|thanks|congrats|awesome|great|"
    r"sounds\s+great|good\s+luck)[\s,]+([A-Z][a-z]{1,12})\b",
)
_VOCATIVE_TAIL_RE = re.compile(r",\s*([A-Z][a-z]{1,12})[!?.]")


_THIRD_PARTY_OWNER_RE = re.compile(
    r"\b(?:my|his|her|their)\s+(brother|sister|mom|dad|mother|father|friend|cousin|"
    r"aunt|uncle|coworker|colleague|neighbor|boss|roommate|son|daughter|grandma|grandpa)\b",
    re.I,
)


def _naming_subject(rec: MemoryRecord, text: str, body: str, match_start: int) -> str:
    owner = None
    for om in _THIRD_PARTY_OWNER_RE.finditer(body[:match_start + 40]):
        owner = om.group(1).lower()
    if owner:
        return owner
    return _speaker_for_atom(rec, text) or ""


def _naming_claims_from_atom(rec: MemoryRecord, atom: str) -> list[ClaimRecord]:
    text = re.sub(r"\s+", " ", atom or "").strip()
    if not text or text.endswith("?") or len(text) > 600:
        return []
    body = re.sub(r"^[A-Z][A-Za-z'_-]{1,32}:\s*", "", text)
    out: list[ClaimRecord] = []

    for m in re.finditer(
        r"(?P<head>[a-z][\w'-]{0,24}(?:\s+[a-z][\w'-]{0,24}){0,3}?)\s+"
        r"(?:called|titled|named)\s+"
        r"(?:[\"“](?P<qname>[^\"“”]{2,60})[\"”]"
        r"|(?P<tname>[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,4}))",
        body,
    ):
        name = (m.group("qname") or m.group("tname") or "").strip(" .,:;!?")
        head_words = [w for w in re.findall(r"[a-z][\w'-]*", m.group("head").lower())
                      if w not in _TITLE_STOP_WORDS]
        if not name or not head_words:
            continue
        claim = _claim_from_atom(rec, text, "state", predicate="named", value=name)
        if claim is None:
            continue
        claim.subject = _naming_subject(rec, text, body, m.start()) or claim.subject
        claim.object = name
        claim.filters = {"naming": "title", "named_head": head_words[-1]}
        out.append(claim)

    for m in re.finditer(
        r"\b(?:the|a|an)\s+(?P<name>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,3})\b", body,
    ):
        name = m.group("name").strip()
        last = name.split()[-1].lower()
        if last not in _NAMED_CATEGORY_NOUNS or len(name.split()) < 2:
            continue
        claim = _claim_from_atom(rec, text, "state", predicate="named", value=name)
        if claim is None:
            continue
        claim.subject = _naming_subject(rec, text, body, m.start()) or claim.subject
        claim.object = name
        claim.filters = {"naming": "title", "named_head": last}
        out.append(claim)

    return out


_VOCATIVE_NOT_A_NAME = {
    "God", "Man", "Boy", "Girl", "Dude", "Buddy", "Honey", "Dear", "Lord", "Gosh",
    "Guys", "Folks", "Well", "Yeah", "Wait", "Stop", "Right", "Please", "Sorry",
}


def _nickname_claims_from_text(rec: MemoryRecord) -> list[ClaimRecord]:
    text = rec.text or ""
    speakers = set(re.findall(r"\b([A-Z][a-z]{2,32})\s*:", text))
    names = speakers | set(re.findall(r"\b([A-Z][a-z]{3,32})\b", text))
    out: list[ClaimRecord] = []
    seen: set[tuple[str, str]] = set()
    for m in re.finditer(r"\b([A-Z][A-Za-z'_-]{1,32})\s*:\s*([^\n]+)", text):
        speaker, utterance = m.group(1), m.group(2)
        for vm in list(_VOCATIVE_LEAD_RE.finditer(utterance)) + list(
            _VOCATIVE_TAIL_RE.finditer(utterance)
        ):
            addr = vm.group(1)
            if (addr == speaker or addr in names or len(addr) < 2
                    or addr in _VOCATIVE_NOT_A_NAME):
                continue
            targets = [n for n in speakers
                       if n != speaker and n != addr and n.lower().startswith(addr.lower())]
            if len(targets) != 1:
                continue
            key = (speaker, addr)
            if key in seen:
                continue
            seen.add(key)
            for sent in _sentences(utterance, limit=40):
                if addr in sent:
                    claim = _claim_from_atom(rec, sent, "state", predicate="nickname",
                                             value=addr)
                    if claim is None:
                        break
                    claim.subject = speaker
                    claim.object = addr
                    claim.filters = {"naming": "nickname", "target": targets[0]}
                    out.append(claim)
                    break
    return out


def heuristic_claims_from_text(rec: MemoryRecord) -> list[ClaimRecord]:
    claims = _duration_answer_claims_from_text(rec)
    claims.extend(_dialogue_answer_claims_from_text(rec))
    claims.extend(_nickname_claims_from_text(rec))
    for atom in _sentences(rec.text or "", limit=200):
        claims.extend(_acquisition_claims_from_atom(rec, atom))
        claims.extend(_action_location_claims_from_atom(rec, atom))
        claims.extend(_action_object_claims_from_atom(rec, atom))
        claims.extend(_list_item_claims_from_atom(rec, atom))
        claims.extend(_naming_claims_from_atom(rec, atom))
        claims.extend(_support_relation_claims_from_atom(rec, atom))
        claims.extend(_event_date_claims_from_atom(rec, atom))
        relation_claims = _relation_object_claims_from_atom(rec, atom)
        if relation_claims:
            # Keep the full-sentence claim TOO: the concise relation claim wins on precision,
            # but the sentence keeps the verb visible for attribution ("who gave ...").
            claims.extend(relation_claims)
        if _is_info_seeking_question(atom):
            continue  # questions are dialogue context, not facts; Q->A adjacency covers them
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
        if claim.filters.get("list") == "item" or claim.filters.get("naming"):
            discriminant = _norm(str(claim.object or ""))
        elif claim.filters.get("event") == "dated":
            # Two dated events in one sentence must not collapse.
            discriminant = (_norm(str(claim.object or "")) + "|"
                            + str(claim.filters.get("event_date")))
        else:
            discriminant = ""
        key = (claim.claim_type, _norm(claim.proof_atom), claim.source_memory_id, discriminant)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    _tag_event_identity(rec, deduped)
    return deduped


def _tag_event_identity(rec: MemoryRecord, claims: list[ClaimRecord]) -> None:
    """P2 write-time identity: once-ish event claims gain a canonical action lemma, the
    object head noun, and the best event date THE ATOM ITSELF states, with an explicit
    precision rank. Identity is decided here, where the phrasing still exists; read-time
    date clustering was reverted three times for guessing it wrong."""
    from datetime import datetime as _dt

    from . import event_identity as ei

    for claim in claims:
        if claim.claim_type != "event":
            continue
        lemma = ""
        for w in re.findall(r"[a-z][\w'-]*", (claim.predicate or "").lower()):
            lemma = ei.canon_lemma(w)
            if lemma:
                break
        if not lemma:
            for w in re.findall(r"[a-z][\w'-]*", (claim.proof_atom or "").lower()):
                lemma = ei.canon_lemma(w)
                if lemma:
                    break
        if not lemma:
            continue
        from .record_ops import _DATE_RE, _event_date, _relative_date_from_atom
        atom = claim.proof_atom or ""
        precision = ei.PRECISION_STATEMENT
        iso = ""
        if _DATE_RE.search(atom):
            d = _event_date(rec, atom)
            if d is not None:
                iso, precision = d.isoformat(), ei.PRECISION_EXPLICIT
        if not iso:
            rel = _relative_date_from_atom(rec, atom)
            if re.match(r"\d{4}-\d{2}-\d{2}$", rel or ""):
                iso, precision = rel, ei.PRECISION_RELATIVE_DAY
            elif rel:
                m = re.search(r"(\d{4})-(\d{2})-(\d{2})", rel)
                if m:
                    iso, precision = m.group(0), ei.PRECISION_WINDOW
        if not iso:
            try:
                iso = _dt.fromtimestamp(rec.valid_at).date().isoformat()
                precision = ei.PRECISION_STATEMENT
            except (OSError, OverflowError, ValueError, TypeError):
                continue
        claim.filters.setdefault("lemma", lemma)
        claim.filters.setdefault("obj_head", ei.obj_head(claim.object or ""))
        claim.filters.setdefault("event_date", iso)
        claim.filters.setdefault("date_precision", precision)
