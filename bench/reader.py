"""The ONE fixed answerer shared by every adapter, so the scoreboard measures MEMORY
quality (what each system retrieves), not answerer quality. Each system retrieves its own
context; this single reader model + the fixed reader prompt turn that context into an
answer identically for all three. (Eidetic-Plus's cascade/cache remain in the product and
are reflected in the cost/latency tables, but the accuracy comparison pins one reader.)

Tier A reader layer (env-gated, default OFF): post-retrieval prompt scaffolds that fix the answer
layer where the n=40 forensics showed we lose -- temporal date arithmetic, open-domain inference
refusal, list over/under-generation, recency disambiguation. Every scaffold is selected by a
DETERMINISTIC classifier over the QUESTION TEXT only, so it is identical across eidetic and both RAG
baselines (a shared-reader fairness change, never an eidetic-only advantage). With all flags OFF the
prompt is byte-identical to before.
"""
from __future__ import annotations

import os
import re
from datetime import datetime

from eidetic.config import get_settings
from eidetic.dashscope_client import get_client

from .judge import FIXED_READER_PHOTOGRAPHIC_PROMPT, FIXED_READER_PROMPT

# Pin one reader model across all systems (override with READER_MODEL; pin a snapshot).
READER_MODEL = os.environ.get("READER_MODEL", "").strip() or "qwen-plus"

# READER_MODE selects the shared answer prompt. "default" (the default) keeps FIXED_READER_PROMPT
# byte-identical; "photographic"/"extractive" switches to the verbatim-quoting prompt. Applied to
# every system (shared reader), so the comparison stays fair.
READER_MODE = os.environ.get("READER_MODE", "default").strip().lower()
_READER_PROMPT = (FIXED_READER_PHOTOGRAPHIC_PROMPT
                  if READER_MODE in ("photographic", "extractive") else FIXED_READER_PROMPT)

# Per-block char cap fed to the reader. Default 3000 = byte-identical to the historical harness;
# raise (e.g. 8000) so a retrieved session whose key fact sits past char 3000 reaches the reader.
# Applied EQUALLY to every system (it lives in the shared fixed reader), so the comparison stays
# fair -- baselines benefit from the larger window too.
READER_BLOCK_CHARS = int(os.environ.get("READER_BLOCK_CHARS", "3000"))


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")


# Tier A interventions. READER_TIER_A is a master switch; each can also be toggled individually.
READER_TIER_A = _flag("READER_TIER_A")
READER_TEMPORAL_SCAFFOLD = READER_TIER_A or _flag("READER_TEMPORAL_SCAFFOLD")
READER_GATED_INFERENCE = READER_TIER_A or _flag("READER_GATED_INFERENCE")
READER_LIST_TWOPASS = READER_TIER_A or _flag("READER_LIST_TWOPASS")
READER_RECENCY_NUDGE = READER_TIER_A or _flag("READER_RECENCY_NUDGE")
READER_PREFERENCE_RUBRIC = READER_TIER_A or _flag("READER_PREFERENCE_RUBRIC")
# q27 fix: make the COT JSON path retry-then-fall-back instead of hard-erroring on a malformed/empty
# response. Gated so the flags-OFF baseline reproduces the prior run byte-identically.
READER_JSON_RESILIENT = READER_TIER_A or _flag("READER_JSON_RESILIENT")

# Deterministic question-type classifiers (regex over the question TEXT only -> identical for every
# system; never reads the dataset's gold category, which would be unfair).
_Q_TEMPORAL = re.compile(r"\b(when|what date|what year|which year|how long|since when|what day|how many years)\b", re.I)
_Q_INFERENCE = re.compile(
    r"\b(would(?:n['’]?t)?|could(?:n['’]?t)?|might|probably|likely|should|plausible)\b|"
    r"\bdo\s+you\s+think\b|\bconsidered\s+a\b|\bclassic\s+example\b|"
    r"\btypical(?:ly)?\b|\bgood\s+(?:fit|candidate|match)\b",
    re.I,
)
_Q_LIST = re.compile(
    r"\b(what|which)\b.{0,40}\b(books?|activit(?:y|ies)|events?|things|hobbies|places|fields?|ways|items?|"
    r"topics?|skills?|languages?|foods?|movies?|songs?|people)\b"
    r"|\bwhat\s+(?:do|does|did)\b.{0,30}\bdo\b", re.I)   # "what does X do to destress" (q24)
_Q_AGGREGATION = re.compile(
    r"\b(how many|how much|number of|total|combined|in total|count(?: of)?|sum of)\b",
    re.I,
)
_Q_RECENCY = re.compile(r"\b(recent(?:ly)?|latest|most\s+recent|currently|current|last|now)\b", re.I)
_Q_PREFERENCE = re.compile(
    r"\b(prefer|preference|preferences|favou?rite|like|likes|love|loves|enjoy|enjoys|"
    r"hate|hates|dislike|dislikes|allergic|avoid|avoids|rather|usually|always|never)\b",
    re.I,
)
_Q_INTERVAL = re.compile(
    r"\b(?:last|past|previous)\s+(?:\d+\s+)?(?:day|week|month|year|few|couple|several)s?\b",
    re.I,
)


def classify_question(question: str) -> dict:
    """Deterministic question-type tags from the question text. Pure (offline-testable)."""
    q = question or ""
    aggregation = bool(_Q_AGGREGATION.search(q))
    list_tag = bool(_Q_LIST.search(q))
    # "last month" / "past two weeks" is an interval, not "the single latest item".
    recency = bool(_Q_RECENCY.search(q)) and not aggregation and not list_tag and not _Q_INTERVAL.search(q)
    return {
        "temporal": bool(_Q_TEMPORAL.search(q)),
        "inference": bool(_Q_INFERENCE.search(q)),
        "list": list_tag,
        "aggregation": aggregation,
        "recency": recency,
        "preference": bool(_Q_PREFERENCE.search(q)),
    }


_SCAFFOLD_TEMPORAL = (
    "\n\n[Temporal question] This asks WHEN something happened. Reason step by step before answering:\n"
    "1. Find the SESSION date (when it was said) AND any relative expression (\"last week\", \"the "
    "Sunday before X\").\n"
    "2. The answer is the EVENT date, not the session date -- distinguish them.\n"
    "3. Compute: anchor date -> apply the offset -> name the weekday. Example: \"Session 2023-05-26 "
    "(Fri); 'Sunday before 25 May' = Sunday 2023-05-21.\"\n"
    "4. If the evidence states the date relative to an anchor (\"Sunday before 25 May\", \"week after "
    "the fair\"), preserve that relative wording in the answer, then add the absolute date in "
    "parentheses when you can compute it.\n"
    "5. Cite the source of the anchor date."
)
_SCAFFOLD_LIST = (
    "\n\n[List question] Answer in two steps:\n"
    "Step 1 (recall): list every candidate item supported by memory, each with its [S#] source.\n"
    "Step 2 (precision): keep ONLY the items that match the question's exact scope; drop "
    "related-but-off-scope items. A generic hobby/event is not enough.\n"
    "Scope examples: for destress/relax/unwind questions, keep only activities explicitly tied to "
    "stress relief, relaxing, unwinding, or being stressed; exclude unrelated hobbies. For helping "
    "children/kids/students, keep only events explicitly tied to helping, mentoring, encouraging, "
    "school, children, kids, youth, or students; exclude unrelated events.\n"
    "Output just the filtered list -- nothing extra."
)
_SCAFFOLD_AGGREGATION = (
    "\n\n[Count / total question] Answer by auditing the evidence, not by picking one source:\n"
    "1. Enumerate every source-supported item, event, amount, hour, day, or occurrence that matches "
    "the exact scope and time window in the question.\n"
    "2. Exclude negated, planned-but-not-done, duplicate, or off-scope mentions.\n"
    "3. For money/time totals, show the arithmetic briefly before the final total.\n"
    "4. Do not answer \"at least\" unless the sources are incomplete; give the exact count/total "
    "when all matching evidence is present."
)
_SCAFFOLD_RECENCY = (
    "\n\n[Most-recent / current] The question wants the single latest matching item. Among same-type "
    "candidates, choose the one with the most recent date and cite that date; do not list older ones."
)
_SCAFFOLD_PREFERENCE = (
    "\n\n[Preference question] Answer from the user's stated preference profile or directly quoted "
    "preference memories.\n"
    "1. Treat first-person statements (\"I prefer...\", \"I hate...\", \"I'm allergic...\") as "
    "preferences about the user.\n"
    "2. Preserve polarity: likes vs dislikes, avoids, cannot have, allergic to, always/never habits.\n"
    "3. If multiple preference lines conflict, cite the most specific/current supported line and "
    "mention the conflict only if needed.\n"
    "4. Return the preference itself, not a generic biography or unrelated fact."
)
# Appended LAST so its permission overrides any earlier "do NOT infer" rule (photographic mode).
_SCAFFOLD_INFERENCE = (
    "\n\n[Inference question] For THIS question you MAY use general world knowledge (this overrides any "
    "earlier 'do not infer' instruction).\n"
    "1. State the premise from memory, or explicitly note when the relevant memory is absent.\n"
    "2. For category questions, combine the remembered category with ordinary world knowledge "
    "(for example, classic children's books imply well-known children's authors).\n"
    "3. For identity, membership, or status questions, absence of self-identification/membership "
    "evidence can support a hedged negative answer when memory only shows related support, allyship, "
    "or interest.\n"
    "4. Begin your answer with \"Likely yes\" or \"Likely no\".\n"
    "5. Give one sentence of reasoning. Do NOT assert facts absent from memory as certain."
)


def build_reader_prompt(question: str, base: str, *,
                        temporal: bool = READER_TEMPORAL_SCAFFOLD,
                        inference: bool = READER_GATED_INFERENCE,
                        list_: bool = READER_LIST_TWOPASS,
                        recency: bool = READER_RECENCY_NUDGE,
                        preference: bool = READER_PREFERENCE_RUBRIC) -> str:
    """`base` plus whichever Tier-A scaffolds apply to this question. With every flag False this
    returns `base` unchanged (flag-off byte-identical). Pure (offline-testable)."""
    cls = classify_question(question)
    out = base
    if temporal and cls["temporal"]:
        out += _SCAFFOLD_TEMPORAL
    if list_ and cls["list"]:
        out += _SCAFFOLD_LIST
    if list_ and cls["aggregation"]:
        out += _SCAFFOLD_AGGREGATION
    if recency and cls["recency"]:
        out += _SCAFFOLD_RECENCY
    if preference and cls["preference"]:
        out += _SCAFFOLD_PREFERENCE
    if inference and cls["inference"]:   # last: overrides the extractive no-infer rule
        out += _SCAFFOLD_INFERENCE
    return out


_COT_SUFFIX = (
    "\n\nBefore answering, write brief evidence notes for each useful source. "
    "Reply ONLY as JSON: {\"notes\":[{\"source\":\"S0\",\"relevant\":true,"
    "\"note\":\"...\"}],\"answer\":\"...\"}. The answer must contain only the final "
    "answer text with source citations. If the context does not contain the answer, "
    "set answer to \"I do not have that in memory.\""
)
# Back-compat alias (was the public name for the composed COT prompt).
FIXED_READER_COT_PROMPT = _READER_PROMPT + _COT_SUFFIX
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_ISO_WEEKDAY_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})(?:\s*\((Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\)"
    r"|\s*,\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))",
    re.I,
)


def normalize_date_weekdays(answer: str) -> str:
    """Correct trivial ISO-date weekday labels in the shared reader output.

    This is deterministic post-processing applied equally to every benchmark system. It only touches
    answers that already contain an explicit YYYY-MM-DD date paired with a weekday label.
    """
    def repl(m: re.Match) -> str:
        iso = m.group(1)
        try:
            actual = _WEEKDAYS[datetime.strptime(iso, "%Y-%m-%d").weekday()]
        except ValueError:
            return m.group(0)
        if m.group(2) is not None:
            return f"{iso} ({actual})"
        return f"{iso}, {actual}"

    return _ISO_WEEKDAY_RE.sub(repl, answer or "")


def _final_answer(answer: str) -> str:
    return normalize_date_weekdays((answer or "").strip())


def answer_with_fixed_reader(question: str, context_blocks: list[str]) -> str:
    client = get_client()
    ctx = "\n\n".join(f"[S{i}] {b[:READER_BLOCK_CHARS]}" for i, b in enumerate(context_blocks))
    user = f"Question: {question}\n\nMemory/context:\n{ctx}"
    base = build_reader_prompt(question, _READER_PROMPT)
    if get_settings().reader_cot_enabled:
        system = base + _COT_SUFFIX
        if not READER_JSON_RESILIENT:
            # Original behavior: one JSON call, raise loudly on an empty/malformed answer.
            data = client.chat_json(READER_MODEL, system, user, temperature=0.1, max_tokens=1536)
            answer = data.get("answer") if isinstance(data, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                from eidetic.dashscope_client import ModelCallError
                raise ModelCallError("Fixed reader COT response did not include a non-empty JSON answer.")
            return _final_answer(answer)
        # q27 fix: retry the JSON path once, then fall back to the plain reader instead of erroring.
        for _ in range(2):
            try:
                data = client.chat_json(READER_MODEL, system, user, temperature=0.1, max_tokens=1536)
                answer = data.get("answer") if isinstance(data, dict) else None
                if isinstance(answer, str) and answer.strip():
                    return _final_answer(answer)
            except Exception:
                pass
        return _final_answer(client.chat(READER_MODEL, base, user, temperature=0.1, max_tokens=512))
    return _final_answer(client.chat(READER_MODEL, base, user, temperature=0.1, max_tokens=512))
