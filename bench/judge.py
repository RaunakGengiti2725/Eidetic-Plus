"""The ONE fixed judge + ONE fixed reader prompt, applied to every benchmark row.

This is the neutrality guarantee: identical grading for Eidetic-Plus, RAG, Mem0, and Graphiti.
Default judge = qwen3-max (Qwen stack). Configurable to GPT-4o via JUDGE_BASE_URL +
JUDGE_API_KEY + JUDGE_MODEL for a leaderboard-comparable headline number; the harness
records which judge was used. LongMemEval uses category-specific judge semantics (temporal
off-by-one tolerance, knowledge-update old-info tolerance, preference rubric leniency,
abstention detection); LoCoMo uses the LLM-as-judge J score.

No mocks: a missing judge key fails loud.
"""
from __future__ import annotations

import os
import re

from eidetic.config import get_settings
from eidetic.dashscope_client import ModelCallError

# The single fixed reader/system prompt used to ANSWER (shared by every adapter that
# defers answering to the harness; adapters that answer internally still use this text).
FIXED_READER_PROMPT = (
    "You are a memory-augmented assistant. Answer the user's question using ONLY the "
    "provided memory/context. Be concise and direct. If the context does not contain the "
    "answer, say you do not have that information."
)

# Photographic / extractive reader (READER_MODE=photographic). Quote verbatim spans with [S#]
# tags instead of paraphrasing, which attacks two measured loss modes: reader over-generation
# (listing more than the gold wants) and paraphrase temporal errors (relative-date drift). It is
# selected in bench/reader.py and applied to ALL systems equally (lives in the shared reader), so
# the comparison stays fair. Default mode keeps FIXED_READER_PROMPT byte-identical.
FIXED_READER_PHOTOGRAPHIC_PROMPT = (
    "You are a memory agent with photographic recall. You may ONLY state facts that appear "
    "VERBATIM or as direct entailments in the cited sources [S0], [S1], ...\n\n"
    "Rules:\n"
    "1. Quote the exact words from memory when stating specific facts (names, dates, titles, "
    "numbers).\n"
    "2. Tag every claim with its source: [S3] \"exact quote here\".\n"
    "3. For temporal questions: use the [Session date YYYY-MM-DD] prefix on each block as ground "
    "truth, and answer the relative expression the question asks for.\n"
    "4. If the exact fact is not in any source block, say: \"I do not have that in memory.\"\n"
    "5. Do NOT infer, generalize, or combine facts the sources do not explicitly support.\n"
    "6. Prefer a short, exact answer over a long, approximate one."
)

_YESNO = re.compile(r"\b(yes|no|correct|incorrect|true|false)\b", re.I)
_WORD = re.compile(r"[a-z0-9]+")
_SOURCE_TAG = re.compile(r"\s*\[(?:s|source)\s*\d+\]\s*", re.I)
_DECLINE = re.compile(r"\b(?:do not|don't) have (?:that|enough|the)|cannot answer|insufficient evidence", re.I)


def _is_yes(text: str) -> bool:
    m = _YESNO.search(text or "")
    if not m:
        return False
    return m.group(1).lower() in ("yes", "correct", "true")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall((text or "").lower()))


def _strip_source_tags(text: str) -> str:
    return _SOURCE_TAG.sub(" ", text or "")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(_strip_source_tags(text).lower())


def _aliases(gold: str, meta: dict | None = None) -> list[str]:
    meta = meta or {}
    raw = meta.get("gold_aliases") or meta.get("aliases") or meta.get("answers") or []
    if isinstance(raw, str):
        raw = [raw]
    aliases = [str(x) for x in raw if str(x).strip()]
    if gold and str(gold).strip():
        aliases.insert(0, str(gold))
    return list(dict.fromkeys(aliases))


def exact_match(predicted: str, aliases: list[str]) -> bool:
    p = _tokens(predicted)
    return bool(p) and any(p == _tokens(a) for a in aliases)


def substring_exact_match(predicted: str, aliases: list[str]) -> bool:
    p = _tokens(predicted)
    if not p:
        return False
    for alias in aliases:
        a = _tokens(alias)
        if not a or len(a) > len(p):
            continue
        for i in range(0, len(p) - len(a) + 1):
            if p[i:i + len(a)] == a:
                return True
    return False


def short_answer_exact_match(gold: str, hypothesis: str, meta: dict | None = None) -> bool:
    """Conservative deterministic judge for short extractive answers.

    This fixes obvious false negatives such as `Summer Vibes [S4]` vs gold `Summer Vibes`, while
    leaving compositional/list/narrative answers to the LLM judge.
    """
    if _DECLINE.search(hypothesis or ""):
        return False
    aliases = _aliases(gold, meta)
    if not aliases:
        return False
    short_aliases = [a for a in aliases if 0 < len(_tokens(a)) <= 8]
    if not short_aliases:
        return False
    # Exact citation-stripped match is always safe. Contiguous substring is safe only when the
    # response is still short enough that it is not a long, partially-correct list explanation.
    if exact_match(hypothesis, short_aliases):
        return True
    return len(_tokens(hypothesis)) <= 16 and substring_exact_match(hypothesis, short_aliases)


class Judge:
    def __init__(self):
        self.model = os.environ.get("JUDGE_MODEL", "").strip() or get_settings().gen_model
        self.base_url = os.environ.get("JUDGE_BASE_URL", "").strip()
        self.api_key = os.environ.get("JUDGE_API_KEY", "").strip()
        self._ds = None

    def describe(self) -> dict:
        return {"judge_model": self.model,
                "judge_backend": "openai-compatible" if self.base_url else "dashscope"}

    def _call(self, system: str, user: str) -> str:
        if self.base_url:  # OpenAI-compatible judge (e.g. GPT-4o)
            if not self.api_key:
                raise ModelCallError("JUDGE_BASE_URL set but JUDGE_API_KEY missing (no fake judging).")
            import httpx

            with httpx.Client(timeout=60.0) as h:
                r = h.post(f"{self.base_url.rstrip('/')}/chat/completions",
                           headers={"Authorization": f"Bearer {self.api_key}"},
                           json={"model": self.model, "temperature": 0.0, "max_tokens": 16,
                                 "messages": [{"role": "system", "content": system},
                                              {"role": "user", "content": user}]})
                if r.status_code >= 300:
                    raise ModelCallError(f"Judge call failed (HTTP {r.status_code}): {r.text[:200]}")
                return r.json()["choices"][0]["message"]["content"].strip()
        if self._ds is None:
            from eidetic.dashscope_client import get_client
            self._ds = get_client()
        return self._ds.chat(self.model, system, user, temperature=0.0, max_tokens=16)

    # ---- LongMemEval (category-specific) ---------------------------------
    def judge_longmemeval(self, question: str, gold: str, hypothesis: str, category: str) -> bool:
        cat = (category or "").lower()
        if "abs" in cat or "abstention" in cat or gold.strip().lower() in ("", "n/a", "no answer", "unanswerable"):
            # Correct iff the system abstained / declined rather than fabricating.
            sys = ("You judge ABSTENTION. The question is unanswerable from memory. Reply 'yes' "
                   "ONLY if the response correctly declines / says it lacks the information; "
                   "reply 'no' if it fabricates an answer.")
            return _is_yes(self._call(sys, f"Question: {question}\nResponse: {hypothesis}"))
        if _DECLINE.search(hypothesis or ""):
            return False
        if short_answer_exact_match(gold, hypothesis):
            return True

        extra = ""
        if "temporal" in cat:
            extra = " Allow off-by-one date tolerance and equivalent date phrasings."
        elif "knowledge-update" in cat or "update" in cat:
            extra = " The newest fact is correct; tolerate omission of superseded old info."
        elif "preference" in cat:
            extra = " Grade leniently against the user's preference; capture the gist."
        sys = ("You are a strict QA grader. Decide if the RESPONSE is correct given the GOLD "
               "answer to the QUESTION. Reply ONLY 'yes' or 'no'." + extra)
        user = f"Question: {question}\nGold: {gold}\nResponse: {hypothesis}"
        return _is_yes(self._call(sys, user))

    # ---- LoCoMo (J score) -------------------------------------------------
    def judge_locomo(self, question: str, gold: str, hypothesis: str) -> bool:
        sys = ("You are an LLM judge for conversational QA (LoCoMo J score). Decide if the "
               "RESPONSE conveys the same answer as the GOLD answer to the QUESTION, allowing "
               "paraphrase. Reply ONLY 'yes' or 'no'.")
        user = f"Question: {question}\nGold: {gold}\nResponse: {hypothesis}"
        return _is_yes(self._call(sys, user))

    def judge_generic_memory(self, question: str, gold: str, hypothesis: str, category: str) -> bool:
        sys = ("You are a strict judge for memory benchmark QA. Decide if the RESPONSE is "
               "correct given the GOLD answer to the QUESTION, allowing paraphrase but not "
               "unsupported extra facts. Reply ONLY 'yes' or 'no'.")
        user = f"Category: {category}\nQuestion: {question}\nGold: {gold}\nResponse: {hypothesis}"
        return _is_yes(self._call(sys, user))

    def judge_memoryagentbench(self, gold: str, hypothesis: str, meta: dict | None = None) -> bool:
        aliases = _aliases(gold, meta)
        if not aliases:
            raise ModelCallError("MemoryAgentBench sample has no gold answer or aliases.")
        return substring_exact_match(hypothesis, aliases)

    def judge_beam(self, gold: str, hypothesis: str, meta: dict | None = None) -> bool:
        aliases = _aliases(gold, meta)
        if not aliases:
            raise ModelCallError("BEAM sample has no gold answer or aliases.")
        return exact_match(hypothesis, aliases) or substring_exact_match(hypothesis, aliases)
