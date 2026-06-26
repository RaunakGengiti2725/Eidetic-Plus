"""Working scratchpad of salient verified facts (Phase 6).

A small DERIVED list of the most retention-worthy memories (high static salience, tie-broken by
verified-helpful usage). It is a CONTEXT CHANNEL, never a source of truth: every entry links back
to the immutable raw source hash, and the caller passes only ACTIVE records so a superseded /
invalidated fact expires from the scratchpad automatically. Pure + offline-testable.
"""
from __future__ import annotations


def select_scratchpad(records, *, top_k: int = 5, min_salience: float = 0.6,
                      activation=None, weight: float = 0.0) -> list[dict]:
    """Pick up to `top_k` ACTIVE records with salience >= min_salience, ordered by salience then
    verified-helpful count. Each entry carries its content hash (provenance), never replacing raw.

    Track 9 Flow: with an `activation` map + `weight`, rank by salience + weight*activation so a
    field-warm fact surfaces in context even when salience alone would not top-k it. The
    min_salience eligibility is unchanged (the scratchpad stays salient facts). activation=None /
    weight=0 -> ranking is identical to today (byte-identical)."""
    act = activation or {}
    eligible = [r for r in records if float(getattr(r, "salience", 0.0)) >= min_salience]

    def _score(r) -> float:
        return float(getattr(r, "salience", 0.0)) + weight * float(act.get(r.memory_id, 0.0))

    eligible.sort(key=lambda r: (_score(r), int(getattr(r, "verified_helpful_count", 0))),
                  reverse=True)
    out: list[dict] = []
    for r in eligible[:max(0, top_k)]:
        out.append({
            "memory_id": r.memory_id,
            "content_hash": r.content_hash,          # links back to the immutable raw source
            "text": (r.text or r.summary or "")[:240],
            "salience": round(float(getattr(r, "salience", 0.0)), 3),
            "verified_helpful_count": int(getattr(r, "verified_helpful_count", 0)),
        })
    return out
