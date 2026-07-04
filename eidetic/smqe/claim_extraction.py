"""Source-backed claim extraction helpers for SMQE consolidation."""
from __future__ import annotations

import hashlib
import re
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
        discriminant = (
            _norm(str(claim.object or ""))
            if (claim.filters.get("list") == "item" or claim.filters.get("naming"))
            else ""
        )
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
