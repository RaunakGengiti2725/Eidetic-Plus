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
    r"hate|hates|dislike|dislikes|can'?t stand|avoid|avoids|allergic|"
    r"always|never|usually|tend to|would rather|rather not"
    r")\b",
    re.I,
)
_ROLE_PREFIX = re.compile(r"^\s*(user|assistant|human|ai)\s*:\s*", re.I)


def is_preference(text: str) -> bool:
    """True if the text states a first-person user preference."""
    body = _ROLE_PREFIX.sub("", text or "")
    return bool(_PREF_RE.search(body))


def extract_preference(text: str) -> str | None:
    """Return the preference-bearing sentence (the profile-worthy statement), or None."""
    if not text:
        return None
    body = _ROLE_PREFIX.sub("", text)
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        if _PREF_RE.search(sentence):
            return sentence.strip()
    return body.strip() if is_preference(text) else None


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
    for line in text.splitlines():
        body = _ROLE_PREFIX.sub("", line)
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            s = sentence.strip()
            if s and _PREF_RE.search(s) and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
    return out
