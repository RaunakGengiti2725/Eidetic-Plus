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
# A numeric computed answer must carry a cardinal -- a digit or a spelled small number.
_NUMERIC_ANSWER_RE = re.compile(
    r"\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|no|none|zero)\b", re.I)
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
    inflection ('sign'/'signed') still ties, and hyphen-insensitive so a hyphen-variant
    spelling ('re-heat') ties its solid form ('reheat')."""
    qterms = {t.replace("-", "")
              for t in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (query or "").lower())
              if t not in _QUERY_TIE_STOP}
    aterms = {t.replace("-", "")
              for t in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (atom or "").lower())
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
    "just", "started", "haven't", "hey", "interested",
}
_ENUM_ITEM_HEAD_STOP = {
    "at", "he", "her", "his", "i", "in", "it", "like", "my", "on", "our",
    "she", "that", "their", "them", "these", "they", "this", "those", "to", "we", "what",
    "you", "your", "up", "with", "off", "for",
}


_OPTION_SPLIT_RE = re.compile(
    r"\b(?:would|prefer|rather|choose|pick|enjoy)\b([^.?!]*?)\bor\b([^.?!]*)", re.I)
# 'How many items must I pick up or return' is a COUNT question wearing an 'or': the
# disjunction joins verb phrases, not answer options. A non-choice wh-head fixes the answer
# type to something no option name can satisfy, so the form floor must not apply.
_NON_CHOICE_WH_RE = re.compile(r"^\s*(?:how\s+(?:many|much|long|often)|when|where|who|why)\b", re.I)


def _option_terms(segment: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (segment or "").lower())
            if t not in _QUERY_TIE_STOP}


_FIRST_PERSON_TOKENS = frozenset({
    "i'm", "i've", "i'd", "i'll", "we're", "we've", "we'd", "we'll", "she's", "he's",
    "it's", "they're", "they've", "that's", "there's",
})


def _answer_adds_information(query: str, answer: str) -> bool:
    """False when every content token of the answer already appears in the question (exact
    or prefix-tolerant) -- such an answer restates the question rather than answering it.
    Pronoun contractions are speaker scaffolding, not information ('I'm reading' answers a
    what-books question with nothing), so they never count as new content."""
    # A double-quoted span is a NAME regardless of its tokens: a show literally titled
    # "That" is stopwords to the tokenizer but real information to the reader.
    if re.search(r'"[^"]{1,60}"', answer or "") and not re.search(r'"[^"]{1,60}"', query or ""):
        return True
    ans_terms = _option_terms(answer) - _FIRST_PERSON_TOKENS
    if not ans_terms:
        # Clock times and bare numbers ('11 pm') tokenize to nothing here; an empty set is
        # unevaluable, not uninformative -- fail open. A pronoun-only answer is different:
        # it HAD content tokens and every one was scaffolding.
        return not _option_terms(answer)
    qterms = _option_terms(query)
    for a in ans_terms:
        if a in qterms:
            continue
        if re.match(r"\d{4}-\d{2}-\d{2}$", a):
            return True
        if any(min(len(a), len(q)) >= 4 and (a.startswith(q) or q.startswith(a))
               for q in qterms):
            continue
        return True
    return False


def _option_choice_answer_names_option(query: str, answer: str) -> bool:
    """An 'A or B?' question is answered by NAMING one of the options. The option-choice
    anchor exemption lets executor logic over preference evidence skip the strict hypothesis,
    which shipped a verbatim-but-irrelevant mid-clause fragment verified for an A-or-B
    car-choice question that it never named either option of. Exact-token overlap with
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
        if ":suggestion_synth" in (result.note or ""):
            # Suggestion synthesis picks advice atoms that rarely echo the question's words;
            # the suggestion machinery's own gates (advice-evidence provenance, deferral)
            # bound what can be composed here. The TAG marks that deliberate carve-out;
            # untagged preference answers face the strict query-aware hypothesis.
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
    # The exemption therefore requires that marker -- a bare fragment under a likelihood
    # question is not a labeled inference and shipped verified on a dev arm.
    return bool(_LIKELY_INFERENCE_RE.search(query or "")
                and re.match(r"\s*(?:yes|no|likely|unlikely)\b", result.answer or "", re.I))


_ANSWER_JUNK_SINGLETONS = _ENUM_ITEM_JUNK | {
    "check", "yeah", "yep", "right", "exactly", "totally", "hey", "hi", "hello",
    # vague quantity/manner fillers: a teaser like "money-wise, I've gotten some cool
    # stuff" says nothing -- strip the filler and only the question's own words remain
    "gotten", "some", "stuff", "things", "money-wise",
}
# A when-question's answer must carry a temporal token; a what/where/who-question's answer
# must not be ONLY a date. Both directions shipped verified on the fresh holdout ('When did
# Noor make the tart?' -> the recipe ingredients; 'what did he and his father do?' ->
# '2023-10-05').
_TEMPORAL_TOKEN_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b|"
    r"\b(?:week|weekend|month|year|day|yesterday|today|tomorrow|tonight|morning|afternoon|"
    r"evening|night|ago|last|next|earlier|later|spring|summer|autumn|fall|winter)\b|"
    r"\b\d{1,2}(?:st|nd|rd|th)\b|\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b",
    re.I,
)
_PURE_DATE_ANSWER_RE = re.compile(
    r"^\s*(?:on\s+|in\s+|around\s+)?(?:(?:19|20)\d{2}-\d{2}-\d{2}|(?:19|20)\d{2}|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|"
    r"december)\s+(?:19|20)\d{2})\s*[.!]?\s*$",
    re.I,
)
# The yes/no form exemption belongs to POLARITY questions only ('Did they ...?'); a
# what-question answered 'Yes - glad you have people to lean on' is junk wearing a yes.
_POLARITY_QUERY_RE = re.compile(
    r"^\s*(?:is|are|was|were|am|do|does|did|has|have|had|can|could|will|would|should|"
    r"must|might)\b", re.I)
_WH_AUX_RE = re.compile(
    r"\b(when|what|where|who|which|why|how)\s+"
    r"(?:is|are|was|were|am|do|does|did|has|have|had|can|could|will|would|should)\b", re.I)


def _main_wh(query: str) -> str:
    """The question's MAIN wh-word: the last wh immediately followed by an auxiliary.
    'When Farid was a child, what did he do?' is a WHAT-question -- its leading 'when'
    heads a subordinate clause ('when Farid...', no auxiliary adjacency). Falls back to the
    leading token when no wh+aux pair exists."""
    matches = _WH_AUX_RE.findall(query or "")
    if matches:
        return matches[-1].lower()
    m = re.match(r"\s*(when|what|where|who|which|why|how)\b", query or "", re.I)
    return m.group(1).lower() if m else ""
_SOURCE_REF_RE = re.compile(r"\s*\[S\d+\]")

# ------------------------------------------------------------------ preference form floor
# Untagged preference_synth answers must be short, well-bounded noun phrases (the fav-object
# and slot extractors' native shape). Conversational fragments quote verbatim, so they
# anchor-entail while answering nothing -- the fragment-shard class of verified-wrongs.
_PREF_FRAGMENT_HEAD_STOP = frozenset(_ENUM_ITEM_HEAD_STOP) | frozenset({
    "i", "i've", "i'm", "i'd", "we", "we've", "it's", "there's", "that's",
    "and", "but", "so", "well", "btw", "yeah", "work's",
    "oh", "ah", "um", "hmm", "huh", "uh",
})
_PREF_TAIL_STOP = frozenset({"to", "of", "in", "on", "at", "with", "for", "from",
                             "or", "and", "but", "the", "a", "an"})
_PREF_AFTERTHOUGHT_RE = re.compile(r",\s*(?:btw|though|too|anyway)\s*[.!?]?\s*$", re.I)
_PREF_FIRST_PERSON_CLAUSE_RE = re.compile(
    r"\b(?:i|we)\s+(?:am|'m|'ve|have|had|was|were|will|'ll|'d|do|don't|can|can't)\b", re.I)


_TITLE_SMALL_WORDS = frozenset({
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "of", "on", "or",
    "the", "to", "with",
})
_TITLE_HEAD_BLOCK = frozenset({
    "yes", "no", "yeah", "ok", "okay", "wow", "thanks", "well", "hey", "hi", "hello",
    "good", "nice", "great", "cool", "sure",
})


def _titlecase_name_answer(text: str) -> bool:
    """An unquoted multi-word Title Case phrase ('I Am Legend', 'The Lord of the Rings:
    The Return of the King') is a NAME regardless of its token heads or length: every
    word is capitalized or a title small-word, no commas, and at least two words carry
    capitals. Interjection heads never start real titles the extractors mint."""
    if "," in text or not text[:1].isupper():
        return False
    words = text.split()
    if len(words) < 2 or words[0].lower() in _TITLE_HEAD_BLOCK:
        return False
    caps = 0
    for w in words:
        core = re.sub(r"[^\w'-]+", "", w)
        if not core:
            return False
        if core[:1].isupper():
            caps += 1
        elif core.lower() not in _TITLE_SMALL_WORDS:
            return False
    return caps >= 2


def preference_answer_form_credible(query: str, answer: str) -> bool:
    """Deterministic form floor for UNTAGGED preference_synth answers. Fail-open shapes
    first (polarity markers on polarity/likely questions, WHOLE-answer quoted titles,
    Title Case names, bare numbers/times); then mid-clause heads, dangling tails,
    trailing afterthoughts, and finite first-person clauses are rejected -- a preference
    answer is a bounded noun phrase, never a quoted sentence shard. Failure yields None
    upstream: the reader (with its own analogous floors) or an honest abstention
    competes, never a sideways ship."""
    text = (answer or "").strip()
    if not text:
        return False
    if re.match(r"\s*(?:yes|no)\b", text, re.I):
        # Polarity markers are the labeled-inference shape ('Yes - <premise>'): credible
        # exactly when the question itself is polarity- or likelihood-shaped.
        return bool(_POLARITY_QUERY_RE.match(query or "")
                    or _LIKELY_INFERENCE_RE.search(query or ""))
    if re.fullmatch(r'"[^"]{1,60}"[.!?]?', text):
        # The WHOLE answer is one double-quoted span: a bare title. A quoted span
        # EMBEDDED in a longer answer is quoted speech inside a fragment ('I told him
        # "no way", btw') and gets no pass -- the rejection rules judge the frame.
        return True
    if re.match(r"\s*\[\d{4}-\d{2}-\d{2}\]", text):
        # A dated timeline ('[2023-02-05] helped ...; [2023-02-10] ...') is a derived
        # deterministic compose the planner routes through this op; not a fragment.
        return True
    if _titlecase_name_answer(text):
        return True
    if not _option_terms(text):
        # Only digit-bearing answers ('11 pm', '7:30') are genuinely unevaluable value
        # shapes; all-stopword/interjection strings ('oh no') are evaluable junk.
        return bool(re.search(r"\d", text))
    tokens = re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())
    if not tokens:
        return True
    if tokens[0] in _PREF_FRAGMENT_HEAD_STOP:
        return False
    if tokens[-1] in _PREF_TAIL_STOP:
        return False
    if _PREF_AFTERTHOUGHT_RE.search(text):
        return False
    if len(tokens) > 8:
        return False
    if _PREF_FIRST_PERSON_CLAUSE_RE.search(text) or (set(tokens) & _FIRST_PERSON_TOKENS):
        return False
    # A comma-spliced coordinated clause ('all styles, but tango is my top pick')
    # is a sentence, not a noun phrase; the extractor should have named the NP itself.
    if re.search(r",\s*(?:but|so)\b", text, re.I):
        return False
    return True


# Words whose following OBJECT is a legitimately extractable preference value: stated
# preference/stative verbs ('even though I love shortbread' -> 'shortbread') and the
# prepositional preference frames English actually uses ("I'm really into pottery",
# "obsessed with jazz", "we usually go out for sushi", "head to bed at 11 pm", "my
# weekends revolve around hiking"). The rejection target is a value quoted from the
# middle of an UNRELATED clause ('browsing woodworking kits at the fair' does not state
# a woodworking-kit preference), so the allowlist is frames, not specific rows.
_PREF_OBJECT_FRAME_WORDS = frozenset({
    "love", "loves", "loved", "adore", "adores", "adored", "like", "likes", "liked",
    "enjoy", "enjoys", "enjoyed", "prefer", "prefers", "preferred", "favorite",
    "favourite", "is", "was", "are", "were", "am", "be", "chose", "choose", "picked",
    "pick", "want", "wants", "wanted", "miss", "misses", "missed", "crave", "craves",
    "craved", "fancy", "fancies", "fancied",
    # prepositional preference frames
    "into", "about", "with", "for", "at", "on", "in", "around", "over",
})


def _pref_premise_position_ok(answer: str, premise: str) -> bool:
    """Mid-sentence provenance floor for single-support preference answers: an answer
    quoted VERBATIM from the middle of a source clause (preceded by a lowercase letter or
    a comma) is a shard, not a stated value -- unless the preceding word is a preference/
    stative verb frame whose object the extractor legitimately names. Sentence boundaries,
    colons, turn markers, and capitalized starts pass; unlocatable answers pass (NLI
    decides)."""
    a = re.sub(r"\s+", " ", (answer or "")).strip()
    p = re.sub(r"\s+", " ", (premise or ""))
    if not a or not p:
        return True
    low_p, low_a = p.lower(), a.lower()
    start = 0
    found = False
    while True:
        i = low_p.find(low_a, start)
        if i == -1:
            break
        found = True
        if p[i].isupper():
            return True
        j = i - 1
        while j >= 0 and p[j] == " ":
            j -= 1
        if j < 0:
            return True
        ch = p[j]
        if not (ch.islower() or ch == ","):
            return True
        m = re.search(r"([a-z][\w'-]*)$", low_p[:j + 1])
        if m and m.group(1) in _PREF_OBJECT_FRAME_WORDS:
            return True
        start = i + 1
    return not found


def reader_answer_form_credible(query: str, answer: str) -> bool:
    """Universal deterministic form floors for READER-path answers. The photographic reader
    quotes sources verbatim, so a conversational fragment ('I'm reading', 'Check,', 'Yeah,
    Priya') entails trivially and ships verified while answering nothing -- every one of a
    fresh holdout slice's 18 verified-wrong rows came through this path. Same primitives as
    the structured floors: junk singletons, non-credible enumerations, enumerations for
    why-questions, option-choice naming, zero-information echoes. Computed shapes do not
    exist here (the reader never does arithmetic), so no op carve-outs apply."""
    text = _SOURCE_REF_RE.sub("", answer or "").strip()
    if not text:
        return False
    low = re.sub(r"[.,!?;:\s]+$", "", text.lower()).strip()
    if low in _ANSWER_JUNK_SINGLETONS:
        return False
    # Degenerate repetition ('involved and involved with organizations') is assembly noise,
    # not an answer -- the same token on both sides of a conjunction never occurs in a
    # meaningful reply.
    if re.search(r"\b(\w+)\s+and\s+\1\b", low):
        return False
    # A polarity answer ('Yes, Farid can work with engines') is credible by form -- but ONLY
    # for a polarity question. A what-question answered 'Yes - glad you have people to lean
    # on' is junk wearing a yes.
    if re.match(r"\s*(?:yes|no)\b", text, re.I):
        return bool(_POLARITY_QUERY_RE.match(query or ""))
    # Wh/temporal type agreement, both directions, on the question's MAIN wh ('When Farid
    # was a child, what did he do?' is a WHAT-question).
    wh = _main_wh(query)
    if wh == "when" and not _TEMPORAL_TOKEN_RE.search(text):
        return False
    if (wh in {"what", "where", "who", "which", "why", "how"}
            and not re.match(r"\s*how\s+(?:long|often|old|many|much)\b", query or "", re.I)
            and _PURE_DATE_ANSWER_RE.match(text)):
        return False
    if _ENUMERATED_ANSWER_RE.match(text):
        # Reader answers are PROSE with commas more often than lists ('Marco enjoys
        # reading, especially while snuggled under the covers, ...'); only a list-like
        # shape -- every comma segment short -- faces the enumeration rules. This differs
        # from the structured path on purpose: executors assemble lists, readers write
        # sentences.
        items = [i.strip() for i in re.split(r",\s*(?:and\s+)?", text) if i.strip()]
        if items and all(len(i.split()) <= 6 for i in items):
            if not _enumeration_items_credible(text):
                return False
            if (re.match(r"\s*why\b", query or "", re.I)
                    and not re.search(r"\b(?:because|since)\b", text, re.I)):
                return False
    if not _option_choice_answer_names_option(query, text):
        return False
    # A double-quoted span is a NAME regardless of its tokens (a show titled "That"): the
    # token pipeline below destroys quote marks, so check the RAW text first.
    if re.search(r'"[^"]{1,60}"', text) and not re.search(r'"[^"]{1,60}"', query or ""):
        return True
    # Junk tokens are not information either: 'Yeah, Priya' for a question about Priya is
    # acknowledgment plus echo, so junk words are stripped before the echo test.
    echo_text = " ".join(t for t in re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())
                         if t not in _ANSWER_JUNK_SINGLETONS)
    if not _answer_adds_information(query, echo_text or text):
        return False
    return True


_CLEAN_FACT_STOP = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "i", "we", "my", "he", "she", "it", "his", "her", "their",
    "they", "them", "for", "with", "as", "by", "that", "this", "these", "those", "has", "have",
    "had", "do", "does", "did", "not", "so",
})
_CLEAN_FACT_PRONOUN_HEAD_RE = re.compile(r"^(?:i|it|we|i'm|i've|it's)\b", re.I)


def _clean_fact_form_credible(query: str, result: StructuredAnswerResult) -> bool:
    """General clean-fact form floor for NON-computed, NON-polarity structured ops. A verified
    structured answer is a self-contained extracted fact, not a raw turn shard: two shapes that
    the preference floor already rejects surface across latest_value / open_inference too and
    NLI-anchor against the source while answering nothing. This applies those two floors to
    every producer path. Computed ops (bare dates/counts are legitimate), polarity answers
    (their own floors), and suggestion_synth (provenance-gated carve-out) are exempt.

    Deliberately NARROW -- only two high-precision shapes, validated to reject no correct answer
    on the regression set. It never rejects wrong-but-clean values (a plausible wrong date
    passes -- form cannot know it is wrong), and never a legitimate list whose items share a
    common noun, nor a parallel timeline whose entries repeat a verb phrase -- those are real
    answers, not garble."""
    answer = (result.answer or "").strip()
    if not answer:
        return True
    content = [w for w in re.findall(r"[a-z0-9][a-z0-9'-]*", answer.lower())
               if w not in _CLEAN_FACT_STOP]

    # (1) First-person conversational lead with no factual anchor is a verbatim turn opening,
    # not an answer -- a bare pronoun+verb clause that adds no value. A first-person clause that
    # DOES carry an anchor (a quoted title, a Capitalized proper noun, or a digit) is a real
    # answer and passes: the anchor is the fact.
    if (_CLEAN_FACT_PRONOUN_HEAD_RE.match(answer)
            and not re.search(r'"[^"]{2,}"', answer)
            and not re.search(r"\b[A-Z][a-z]{2,}", answer)
            and not re.search(r"\d", answer)
            and len(content) <= 4):
        return False

    # (2) A comma-list whose item carries a speaker turn-header shape ('<Name>: <free text>') is
    # a malformed enumeration -- the extractor grabbed a dialogue turn, not a list value. Keyed
    # strictly on the capitalized-token-then-colon header shape so legitimate lists never trip.
    if "," in answer:
        segs = [s.strip() for s in re.split(r",\s*(?:and\s+)?", answer) if s.strip()]
        if len(segs) >= 2 and any(re.match(r"[A-Z][a-zA-Z]+\s*:\s+\S", s) for s in segs):
            return False

    return True


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
            and ":suggestion_synth" not in (result.note or "")
            and ":claim_list_enum" not in (result.note or "")
            and not _enumeration_items_credible(result.answer)):
        # ':suggestion_synth' keeps the same carve-out as the anchor rule: suggestion output
        # is context fragments by design and provenance-gated upstream (every OTHER
        # preference_synth answer now faces this floor). ':claim_list_enum'
        # compositions share it for the FORM floor only -- every item there is backed by its
        # own typed write-time claim (junk-filtered at extraction, per-item proof), so a
        # 3-char trick name ('sit') is not fragment soup; the per-support strict hypothesis
        # below still runs unchanged, so nothing ships without live entailment.
        return None
    # Numeric computed-op TYPE floor: a count / sum / elapsed-delta answer is a NUMBER by
    # construction. Computed ops are exempt from the prose form floors (their form is derived),
    # but that exemption let non-numeric garbage ("Apply the Dr" for a money sum) ship
    # verified=True -- a verified-precision hole. Require a numeric token (digit or number word)
    # for the three numeric ops only; event_order/table_lookup answers need not be numeric.
    if (verify and result.op in {"count_aggregate", "multi_session_sum", "temporal_delta"}
            and not _NUMERIC_ANSWER_RE.search(result.answer or "")):
        return None
    # Preference FORM refusal: untagged preference_synth answers must be bounded noun
    # phrases (or protected polarity/title/number shapes). The ':suggestion_synth' tag
    # marks the one deliberate fragment carve-out (provenance-gated suggestion synthesis).
    pref_floor = bool(verify and result.op == "preference_synth"
                      and ":suggestion_synth" not in (result.note or ""))
    if pref_floor and not preference_answer_form_credible(query, result.answer):
        return None
    # Option-choice FORM refusal: 'A or B?' is answered by naming an option. Applies across
    # ops (preference_synth included -- its fragment carve-out is for suggestion synthesis,
    # not for dodging the question's own option set); computed ops are exempt because their
    # answers are derived values (counts, deltas) whose form the operator already fixes.
    if (verify and result.op not in _COMPUTED_OPS
            and not _option_choice_answer_names_option(query, result.answer)):
        return None
    # WHY-question FORM refusal: a reason has clause shape ('because/since/to keep...'), never
    # comma-list shape. Credible-item enumerations still answer nothing causal ('Friday,
    # adoption agency interviews, LGBTQ, Research' shipped verified for a why-question on the
    # fresh holdout) -- the items are quotable nouns, so NLI anchors happily. Enumerations
    # for why-questions fall to the reader, which composes an actual reason.
    if (verify and result.op not in _COMPUTED_OPS
            and re.match(r"\s*why\b", query or "", re.I)
            and _ENUMERATED_ANSWER_RE.match(result.answer or "")
            and not re.search(r"\b(?:because|since)\b", result.answer or "", re.I)):
        return None
    # When-question type agreement, structured side: a when-answer without a single temporal
    # token is malformed regardless of which operator derived it (an ongoing-activity
    # fragment with no date word shipped verified for a when-took-place question).
    if (verify and _main_wh(query) == "when"
            and not _TEMPORAL_TOKEN_RE.search(result.answer or "")):
        return None
    # Zero-information FORM refusal: an answer whose every content token already appears in
    # the question restates it instead of answering ('My girlfriend' for 'what places have
    # Ravi and his girlfriend checked out?' shipped verified on the fresh holdout -- the
    # fragment is quotable, so it anchor-verifies while adding nothing). Computed ops are
    # exempt (a count IS query tokens plus a digit... a digit is new; but '2' when the query
    # says '2 sensors' is not, and the operator's arithmetic is the proof there). Option
    # choices are exempt by construction: naming an option MUST echo the question.
    if (verify and result.op not in _COMPUTED_OPS
            and not _OPTION_SPLIT_RE.search(query or "")
            and not re.match(r"\s*(?:yes|no)\b", result.answer or "", re.I)
            and not _answer_adds_information(query, result.answer)):
        return None
    # Clean-fact FORM refusal: a verified structured answer must be a self-contained fact, not
    # a conversational turn shard (a bare first-person pronoun+verb opening) or a comma-list
    # that captured a dialogue turn-header. Non-computed ops only (computed values are bare by
    # design); polarity and suggestion_synth keep their own carve-outs.
    if (verify and result.op not in _COMPUTED_OPS
            and ":suggestion_synth" not in (result.note or "")
            and not re.match(r"\s*(?:yes|no)\b", result.answer or "", re.I)
            and not _clean_fact_form_credible(query, result)):
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
            # Mid-sentence PROVENANCE refusal (single-support preference answers): a
            # verbatim quote that starts mid-clause in its own premise is a fragment
            # shard regardless of entailment.
            if (pref_floor and len(result.supports) == 1
                    and not _pref_premise_position_ok(result.answer, premise)):
                return None
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
