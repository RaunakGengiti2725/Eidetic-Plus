"""First-class typed preferences (single-session-preference is field-wide weak).

Deterministic detector (offline-testable): flag turns that STATE a user preference so they
are typed as preference memories and accumulated into a per-user profile, instead of being
buried in generic fact chunks and never surfaced in top-k. The LLM may refine the canonical
preference text during consolidation, but typing/surfacing does not depend on it.

Fairness note: the rubric-aware *prompt* lives in the SHARED reader (lifts all three
systems equally). Eidetic's preference WIN comes from SURFACING typed preferences in the
retrieved context, not from a different reader.
"""
from __future__ import annotations

import re

# First-person preference cues. Kept conservative to avoid over-typing generic statements.
_PREF_RE = re.compile(
    r"\b(i|my)\b.{0,30}\b("
    r"prefer|prefers|preferred|like|likes|love|loves|enjoy|enjoys|favou?rite|"
    r"hate|hates|dislike|dislikes|can'?t stand|cannot stand|not a fan|"
    r"avoid|avoids|allergic|can'?t (?:eat|drink|have|wear|use)|"
    r"cannot (?:eat|drink|have|wear|use)|do(?:es)?n'?t (?:like|eat|drink|have|wear|use)|"
    r"always|never|usually|tend to|would rather|rather not"
    r")\b",
    re.I,
)
_ROLE_PREFIX = re.compile(r"^\s*(user|assistant|human|ai)\s*:\s*", re.I)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:;|,\s+and|,\s+but|\band\b|\bbut\b)\s+(?=i(?:\s|'))", re.I)
_SPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.!?]+$")


def _clean(text: str) -> str:
    body = _ROLE_PREFIX.sub("", text or "").strip()
    body = _SPACE_RE.sub(" ", body)
    body = re.sub(r"\bI'm\b", "I am", body, flags=re.I)
    body = re.sub(r"\bI can't\b", "I cannot", body, flags=re.I)
    body = re.sub(r"\bI don't\b", "I do not", body, flags=re.I)
    body = re.sub(r"\bI won't\b", "I will not", body, flags=re.I)
    return body.strip()


def _target(text: str) -> str:
    text = _TRAILING_PUNCT_RE.sub("", text or "")
    text = text.strip(" \t\n\r\"'`")
    return _SPACE_RE.sub(" ", text).strip()


def _profile(stem: str, target: str) -> str | None:
    obj = _target(target)
    if not obj:
        return None
    return f"User {stem} {obj}."


def _ensure_period(text: str) -> str:
    text = _target(text)
    return f"{text}." if text else text


def _iter_preference_segments(text: str):
    """Yield preference-bearing sentence/clause fragments in order."""
    for line in (text or "").splitlines():
        body = _ROLE_PREFIX.sub("", line)
        for sentence in _SENTENCE_SPLIT_RE.split(body):
            for clause in _CLAUSE_SPLIT_RE.split(sentence):
                s = clause.strip()
                if s and _PREF_RE.search(s):
                    yield s


def is_preference(text: str) -> bool:
    """True if the text states a first-person user preference."""
    body = _ROLE_PREFIX.sub("", text or "")
    return bool(_PREF_RE.search(body))


def extract_preference(text: str) -> str | None:
    """Return the preference-bearing sentence (the profile-worthy statement), or None."""
    if not text:
        return None
    body = _ROLE_PREFIX.sub("", text)
    for sentence in _SENTENCE_SPLIT_RE.split(body):
        if _PREF_RE.search(sentence):
            return sentence.strip()
    return body.strip() if is_preference(text) else None


def canonicalize_preference(text: str) -> str | None:
    """Return a stable profile sentence for a stated preference.

    This is intentionally deterministic and conservative: clear paraphrases collapse
    ("I love jazz" and "I like jazz"), while opposite polarity stays separate.
    """
    s = _clean(text)
    if not s:
        return None
    if re.match(r"^user(?:'s)?\s+", s, re.I):
        return _ensure_period(s)

    patterns: list[tuple[str, str]] = [
        (r"^i\s+(?:do not|never)\s+(?:really\s+)?(?:like|love|enjoy)\s+(.+)$", "dislikes"),
        (r"^i\s+(?:am\s+)?not\s+(?:a\s+)?fan\s+of\s+(.+)$", "dislikes"),
        (r"^i\s+(?:hate|dislike|cannot stand)\s+(.+)$", "dislikes"),
        (r"^i\s+(?:always\s+|usually\s+)?avoid\s+(.+)$", "avoids"),
        (r"^i\s+(?:can\s+not|cannot)\s+(eat|drink|have|wear|use)\s+(.+)$", "cannot"),
        (r"^i\s+(?:am\s+)?allergic\s+to\s+(.+)$", "is allergic to"),
        (r"^i\s+(?:would\s+)?prefer\s+(.+)$", "prefers"),
        (r"^i\s+(?:really\s+|absolutely\s+|definitely\s+)?(?:like|love|enjoy)\s+(.+)$", "likes"),
        (r"^i\s+(?:am\s+)?(?:a\s+)?fan\s+of\s+(.+)$", "likes"),
        (r"^i\s+would\s+rather\s+not\s+(.+)$", "would rather not"),
        (r"^i\s+would\s+rather\s+(.+)$", "would rather"),
        (r"^i\s+tend\s+to\s+(.+)$", "tends to"),
        (r"^i\s+usually\s+(.+)$", "usually"),
        (r"^i\s+always\s+(.+)$", "always"),
        (r"^i\s+never\s+(.+)$", "never"),
    ]
    for pattern, stem in patterns:
        m = re.match(pattern, s, re.I)
        if not m:
            continue
        if stem == "cannot":
            verb, obj = m.group(1), m.group(2)
            return _profile(f"cannot {verb.lower()}", obj)
        return _profile(stem, m.group(1))

    m = re.match(r"^my\s+favou?rite(?:\s+(.+?))?\s+(?:is|are)\s+(.+)$", s, re.I)
    if m:
        slot = _target(m.group(1) or "")
        obj = _target(m.group(2))
        if obj:
            if slot:
                return f"User's favorite {slot} is {obj}."
            return f"User's favorite is {obj}."

    return _ensure_period(s) if is_preference(s) else None


def preference_dedup_key(text: str) -> str:
    """Stable key for profile preference deduplication."""
    canon = canonicalize_preference(text) or _clean(text)
    key = _TRAILING_PUNCT_RE.sub("", canon).lower()
    key = _SPACE_RE.sub(" ", key).strip()
    key = re.sub(r"^user(?:'s)?\s+", "", key)
    return key


def preference_polarity(text: str) -> str:
    """Return positive/negative/neutral for profile invalidation.

    This is not a sentiment classifier. It only reads canonical preference predicates so the active
    profile can stop surfacing outdated opposite preferences while the raw memories remain intact.
    """
    canon = (canonicalize_preference(text) or _ensure_period(text)).lower()
    if re.search(r"\b(dislikes|avoids|cannot|is allergic to|would rather not|never)\b", canon):
        return "negative"
    if re.search(r"\b(likes|prefers|favorite|would rather|tends to|usually|always)\b", canon):
        return "positive"
    return "neutral"


def preference_update_key(text: str) -> str:
    """Topic key for deciding when a newer preference supersedes an older profile line.

    Favorite slots supersede by slot ("favorite music"), while positive/negative preferences
    supersede by normalized object ("coffee"). Empty means "do not auto-invalidate".
    """
    canon = canonicalize_preference(text) or _ensure_period(text)
    low = _TRAILING_PUNCT_RE.sub("", canon).lower()
    low = _SPACE_RE.sub(" ", low).strip()

    m = re.match(r"^user'?s favorite(?: ([a-z0-9][a-z0-9 \-']*?))? is (.+)$", low)
    if m:
        slot = _target(m.group(1) or "default").lower()
        return f"favorite:{slot}"

    patterns = [
        r"^user (?:likes|dislikes|prefers|avoids) (.+)$",
        r"^user (?:cannot (?:eat|drink|have|wear|use)) (.+)$",
        r"^user is allergic to (.+)$",
        r"^user would rather not (.+)$",
        r"^user would rather (.+)$",
        r"^user tends to (.+)$",
        r"^user usually (.+)$",
        r"^user always (.+)$",
        r"^user never (.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, low)
        if match:
            target = _target(match.group(1)).lower()
            target = re.sub(r"\b(?:now|currently|these days|lately)\b", "", target)
            target = _SPACE_RE.sub(" ", target).strip()
            return f"target:{target}" if target else ""
    return ""


def extract_all_preferences(text: str) -> list[str]:
    """Return EVERY preference-bearing sentence in `text`, one per stated preference, in order
    and deduped. A session blob mixes several turns; `extract_preference` returns only the first
    match, so mid-conversation preferences are lost. The sentence scan (gated PREF_SENTENCE_SCAN)
    surfaces all of them. Splits per line first (turn boundaries) then per sentence. Pure +
    deterministic (offline-testable)."""
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for s in _iter_preference_segments(text):
        key = preference_dedup_key(s) or s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out
