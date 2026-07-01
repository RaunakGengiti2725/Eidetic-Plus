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


def _atom_anchor_allowed(query: str, result: StructuredAnswerResult) -> bool:
    """Anchor-level verification is the honest standard when the answer is DERIVED rather than
    quoted: multi-support composition (joins/orderings), computed operators (arithmetic over
    anchors), and option choices (executor logic over preference evidence). Everything else must
    survive the strict query-aware hypothesis."""
    if len(result.supports) > 1:
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
