"""Verification bridge from SMQE results to normal Eidetic answers."""
from __future__ import annotations

import re

from eidetic.models import Answer, Citation, NLILabel, StructuredAnswerResult


def _query_answer_hypothesis(query: str, answer: str) -> str:
    query = re.sub(r"\s+", " ", (query or "").strip())
    answer = re.sub(r"\s+", " ", (answer or "").strip())
    if not query:
        return answer
    return f"Question: {query}\nAnswer: {answer}"


_COMPUTED_OPS = {"temporal_delta", "count_aggregate", "multi_session_sum", "event_order",
                 "relative_temporal", "table_lookup"}
_OPTION_CHOICE_RE = re.compile(
    r"\b(?:would|prefer|rather|enjoy|choose|pick)\b[^.?!]*\bor\b", re.I)
_LIKELY_INFERENCE_RE = re.compile(
    r"\bwould\b[^.?!]{0,80}\blikely\b|\blikely\s+(?:have|has|enjoy|like|want|be|get|buy)\b",
    re.I,
)


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


_QUERY_TIE_STOP = {
    "about", "after", "before", "could", "did", "does", "ever", "from", "have", "into",
    "many", "should", "that", "their", "them", "there", "these", "they", "this", "were",
    "what", "when", "where", "which", "will", "with", "would", "your",
}


def _query_tie_hits(query: str, atom: str) -> int:
    """Content-word overlap between the query and a support atom, prefix-tolerant so
    inflection ('sign'/'signed') still ties."""
    qterms = {t for t in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (query or "").lower())
              if t not in _QUERY_TIE_STOP}
    aterms = {t for t in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (atom or "").lower())
              if t not in _QUERY_TIE_STOP}
    hits = 0
    for q in qterms:
        for a in aterms:
            if q == a or (min(len(q), len(a)) >= 4 and (q.startswith(a) or a.startswith(q))):
                hits += 1
                break
    return hits


_ENUMERATED_ANSWER_RE = re.compile(r"^[^.;!?]{0,120},\s[^.;!?]{1,120},\s")
_ENUM_ITEM_JUNK = {
    "awesome", "cool", "definitely", "fine", "good", "great", "great job", "nice", "no",
    "ok", "okay", "really", "similarly", "sure", "thank you", "thanks", "well", "wow", "yes",
}
_ENUM_ITEM_HEAD_STOP = {
    "at", "he", "her", "his", "i", "in", "it", "like", "my", "on", "our",
    "she", "that", "their", "these", "they", "this", "those", "to", "we", "what",
    "you", "your",
}


_OPTION_SPLIT_RE = re.compile(
    r"\b(?:would|prefer|rather|choose|pick|enjoy)\b([^.?!]*?)\bor\b([^.?!]*)", re.I)
# 'How many items do I need to pick up or return' is a COUNT question wearing an 'or': the
# disjunction joins verb phrases, not answer options. A non-choice wh-head fixes the answer
# type to something no option name can satisfy, so the form floor must not apply.
_NON_CHOICE_WH_RE = re.compile(r"^\s*(?:how\s+(?:many|much|long|often)|when|where|who|why)\b", re.I)


def _option_terms(segment: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (segment or "").lower())
            if t not in _QUERY_TIE_STOP}


def _answer_adds_information(query: str, answer: str) -> bool:
    """False when every content token of the answer already appears in the question (exact
    or prefix-tolerant) -- such an answer restates the question rather than answering it."""
    ans_terms = _option_terms(answer)
    if not ans_terms:
        # Clock times and bare numbers ('11 pm') tokenize to nothing here; an empty set is
        # unevaluable, not uninformative -- fail open.
        return True
    qterms = _option_terms(query)
    for a in ans_terms:
        if a in qterms:
            continue
        if any(min(len(a), len(q)) >= 4 and (a.startswith(q) or q.startswith(a))
               for q in qterms):
            continue
        return True
    return False


def _option_choice_answer_names_option(query: str, answer: str) -> bool:
    """An 'A or B?' question is answered by NAMING one of the options. The option-choice
    anchor exemption lets executor logic over preference evidence skip the strict hypothesis,
    which shipped a verbatim-but-irrelevant fragment verified ('ten, I've been fascinated with
    how machines work' for 'Dodge Charger or Subaru Forester?'). Exact-token overlap with
    either option segment is the deterministic form floor; answers naming neither fall back
    to the reader. Exact match on purpose -- prefix tolerance would let 'work' claim
    'working on'."""
    if _NON_CHOICE_WH_RE.match(query or ""):
        return True
    m = _OPTION_SPLIT_RE.search(query or "")
    if not m:
        return True
    opts = _option_terms(m.group(1)) | _option_terms(m.group(2))
    if not opts:
        return True
    return bool(_option_terms(answer) & opts)


def _enumeration_items_credible(answer: str) -> bool:
    """Every comma item of an assembled list must be a short content noun phrase.

    Conversational filler is quotable, so a junk list ('Good, Ok, You Get') anchor-verifies
    while answering nothing - that exact shape shipped verified-wrong at n=40. Interjections,
    pronoun/preposition heads, and sub-4-char fragments disqualify the WHOLE enumeration from
    the anchor exemption (it must then survive the strict query-aware hypothesis)."""
    payload = re.sub(r"^[^,:;]{0,80}:\s*", "", answer or "")   # drop a "Header: " prefix
    items = [i.strip() for i in re.split(r",\s*(?:and\s+)?", payload) if i.strip()]
    if len(items) < 2:
        return True
    for item in items:
        words = item.split()
        low = item.lower()
        if len(item) < 4 or len(words) > 6:
            return False
        if low in _ENUM_ITEM_JUNK or words[0].lower() in _ENUM_ITEM_HEAD_STOP:
            return False
    return True


def _atom_anchor_allowed(query: str, result: StructuredAnswerResult) -> bool:
    """Anchor-level verification is the honest standard when the answer is DERIVED rather than
    quoted: multi-support composition (joins/orderings), computed operators (arithmetic over
    anchors), and option choices (executor logic over preference evidence). Everything else must
    survive the strict query-aware hypothesis."""
    # An ASSEMBLED ENUMERATION from a non-computed op earns the anchor exemption only when
    # every item is a credible content phrase; fragment lists must face the strict hypothesis.
    if (_ENUMERATED_ANSWER_RE.match(result.answer or "")
            and result.op not in _COMPUTED_OPS
            and not _enumeration_items_credible(result.answer)):
        return False
    if len(result.supports) > 1:
        # Witness rule: INDEPENDENT witnesses (distinct records) earn the exemption outright.
        # Two quotable atoms from the SAME record are one source wearing two hats; they keep the
        # exemption only when the composition is query-tied - some support atom must share real
        # content terms with the question. Untied same-record pairs fall through to the other
        # exemptions (computed ops, option choices) or the strict hypothesis.
        if len({s.memory_id for s in result.supports}) > 1:
            return True
        if result.op == "preference_synth":
            # Suggestion synthesis picks advice atoms that rarely echo the question's words;
            # the suggestion machinery's own gates (advice-evidence provenance, deferral)
            # bound what can be composed here.
            return True
        if any(_query_tie_hits(query, s.proof_atom or s.answer_atom or "") >= 2
               for s in result.supports):
            return True
    if ":date_anchored" in (result.note or ""):
        # The explicit-date window filter already PROVED the winning atom's event date matches
        # the queried day deterministically; asking NLI to re-derive that date link is what
        # flapped run to run. The verbatim anchor plus the deterministic date proof is the
        # honest standard here.
        return True
    if result.op in _COMPUTED_OPS:
        return True
    if _OPTION_CHOICE_RE.search(query or ""):
        return True
    # Explicitly speculative questions ("Would X likely ...?"): the verifiable content is the
    # cited premise; the yes/no marker is the executor's labeled inference over that premise.
    return bool(_LIKELY_INFERENCE_RE.search(query or ""))


def answer_from_result(retriever, query: str, result: StructuredAnswerResult,
                       *, verify: bool = True) -> Answer | None:
    if result is None or not result.answer or not result.supports:
        return None
    # Answer-FORM refusal: a non-credible enumeration from a non-computed op is malformed
    # regardless of entailment - live NLI was observed entailing fragment soup against a long
    # premise, shipping it verified. Deterministic refusal here covers EVERY producer path
    # (the dispatch decline only guards the first helper chain).
    if (verify
            and _ENUMERATED_ANSWER_RE.match(result.answer or "")
            and result.op not in _COMPUTED_OPS
            and result.op != "preference_synth"
            and not _enumeration_items_credible(result.answer)):
        # preference_synth keeps the same carve-out as the anchor rule: suggestion output is
        # context fragments by design and provenance-gated upstream.
        return None
    # Option-choice FORM refusal: 'A or B?' is answered by naming an option. Applies across
    # ops (preference_synth included -- its fragment carve-out is for suggestion synthesis,
    # not for dodging the question's own option set); computed ops are exempt because their
    # answers are derived values (counts, deltas) whose form the operator already fixes.
    if (verify and result.op not in _COMPUTED_OPS
            and not _option_choice_answer_names_option(query, result.answer)):
        return None
    # Zero-information FORM refusal: an answer whose every content token already appears in
    # the question restates it instead of answering ('My girlfriend' for 'what places have
    # Andrew and his girlfriend checked out?' shipped verified on the fresh holdout -- the
    # fragment is quotable, so it anchor-verifies while adding nothing). Computed ops are
    # exempt (a count IS query tokens plus a digit... a digit is new; but '2' when the query
    # says '2 sensors' is not, and the operator's arithmetic is the proof there). Option
    # choices are exempt by construction: naming an option MUST echo the question.
    if (verify and result.op not in _COMPUTED_OPS
            and not _OPTION_SPLIT_RE.search(query or "")
            and not re.match(r"\s*(?:yes|no)\b", result.answer or "", re.I)
            and not _answer_adds_information(query, result.answer)):
        return None
    citations: list[Citation] = []
    entailed = 0
    unresolved = 0
    for support in result.supports:
        rec = retriever.store.get_record(support.memory_id)
        if rec is None:
            unresolved += 1
            continue
        atom = support.proof_atom or support.answer_atom or result.answer
        label, conf = (NLILabel.ENTAILMENT, 1.0)
        if verify:
            premise = rec.text or rec.summary or ""
            try:
                premise = retriever._ground_truth(rec)
            except Exception:
                pass
            strict_hypothesis = (
                result.backend == "claim"
                and callable(getattr(retriever, "verify", None))
            )
            if strict_hypothesis:
                anchor_ok = _atom_anchor_allowed(query, result)
                if anchor_ok and _norm_ws(atom) and _norm_ws(atom) in _norm_ws(premise):
                    # A verbatim source anchor is the strongest possible proof for a derived
                    # answer; no model call needed.
                    label, conf = NLILabel.ENTAILMENT, 1.0
                else:
                    label, conf = retriever.verify_citation(
                        rec,
                        _query_answer_hypothesis(query, result.answer),
                    )
                    if label != NLILabel.ENTAILMENT and anchor_ok:
                        label, conf = retriever.verify_citation(rec, atom)
            elif re.sub(r"\s+", " ", atom.lower()).strip() in re.sub(r"\s+", " ", premise.lower()).strip():
                label, conf = NLILabel.ENTAILMENT, 1.0
            else:
                label, conf = retriever.verify_citation(rec, atom)
        citations.append(Citation(
            memory_id=rec.memory_id,
            content_hash=rec.content_hash,
            raw_uri=rec.raw_uri,
            source=rec.source,
            valid_at=rec.valid_at,
            snippet=re.sub(r"\s+", " ", atom or "")[:500],
            nli_label=label,
            nli_score=conf,
        ))
        if label == NLILabel.ENTAILMENT:
            entailed += 1
    if not citations:
        return None
    if verify and (unresolved or entailed != len(citations) or len(citations) != len(result.supports)):
        return None
    verified = bool(verify and entailed == len(citations) == len(result.supports))
    return Answer(
        question=query,
        answer=result.answer,
        verified=verified,
        confidence=result.confidence if (not verify or verified) else 0.0,
        citations=citations,
        unverified_claims=[],
        generated_by="smqe",
        retrieved_count=len(result.supports),
        note=result.note or f"smqe:{result.op}:{result.backend}",
    )
