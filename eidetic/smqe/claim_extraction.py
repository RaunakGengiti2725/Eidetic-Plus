"""Source-backed claim extraction helpers for SMQE consolidation."""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from eidetic import preferences
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
        pieces.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
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
        r"\b(?:i|we)\s+(?P<verb>[a-z][a-z'-]{2,}(?:ed|t))\s+(?P<object>[^.;!?]{3,90}?)(?=\s+(?:last|this|recently|because|while|when|where|which|that|so|together|again|alone|yesterday|today)\b|[.;!?]|$)",
        r"\b(?:i|we)\s+(?P<verb>[a-z][a-z'-]{2,}(?:ed|t))\s+(?:at|in|to|from)\s+(?P<object>[^.;!?]{3,90}?)(?=\s+(?:last|this|recently|because|while|when|where|which|that|so)\b|[.;!?]|$)",
        # Irregular pasts the ed|t suffix rule can never see (\'I read The Alchemist\',
        # \'we saw Hamilton\'), clitic-tolerant; and the offered/given passive (\'I\'ve been
        # offered a deal with Nike\') whose object is the enumerable fact.
        r"\b(?:i|we)(?:\'ve|\'d)?\s+(?:just\s+|also\s+|recently\s+|finally\s+)?(?P<verb>read|reread|wrote|saw|met|took)\s+(?P<object>[A-Z][^.;!?]{2,90}?)(?=\s+(?:last|this|recently|because|while|when|and|so|together|again|yesterday|today)\b|[.;!?]|$)",
        r"\b(?:i|we)(?:\'ve|\'d)?\s+been\s+(?P<verb>offered|given|promised)\s+(?P<object>[^.;!?]{3,90}?)(?=\s+(?:last|this|recently|because|while|when|so|by)\b|[.;!?]|$)",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            obj = _clean_relation_object(m.group("object"))
            verb = m.group("verb").lower().replace("'", "")
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


def heuristic_claims_from_text(rec: MemoryRecord) -> list[ClaimRecord]:
    claims = _duration_answer_claims_from_text(rec)
    claims.extend(_dialogue_answer_claims_from_text(rec))
    for atom in _sentences(rec.text or "", limit=200):
        claims.extend(_acquisition_claims_from_atom(rec, atom))
        claims.extend(_action_location_claims_from_atom(rec, atom))
        claims.extend(_action_object_claims_from_atom(rec, atom))
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
        key = (claim.claim_type, _norm(claim.proof_atom), claim.source_memory_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped
