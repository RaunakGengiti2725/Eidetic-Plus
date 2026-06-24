"""The inferred-edge/fact gate. TOKEN-FREE by default: combines the KG-embedding plausibility
score with embedding SUPPORT (cosine between the inferred fact and its evidence) into a
confidence, and admits only above-threshold items into the separate inferred layer.

Optional enrichment: with use_llm_nli=True a real NLI call (premise = evidence text,
hypothesis = inferred fact) must ALSO return entailment. That costs tokens, so it is OFF by
default -- the token-free layer must deliver value without it (dossier §7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class GateResult:
    passed: bool
    confidence: float
    reason: str = ""


def confidence_from(score: float, support: float) -> float:
    """Blend a normalized KG-embedding plausibility score with embedding support (both 0..1)."""
    score = max(0.0, min(1.0, score))
    support = max(0.0, min(1.0, support))
    return 0.5 * score + 0.5 * support


def gate(score: float, support: float, threshold: float,
         llm_nli: Optional[Callable[[], bool]] = None) -> GateResult:
    """Admit an inferred item iff confidence >= threshold (token-free) and, if an optional
    `llm_nli` checker is supplied, it also entails. `llm_nli` is a zero-arg callable returning
    bool -- only invoked when confidence already clears the bar (so the cheap gate runs first)."""
    conf = confidence_from(score, support)
    if conf < threshold:
        return GateResult(False, conf, "below-confidence-threshold")
    if llm_nli is not None:
        try:
            if not llm_nli():
                return GateResult(False, conf, "llm-nli-not-entailed")
        except Exception as e:  # fail loud is for missing key at call sites; here be safe
            return GateResult(False, conf, f"llm-nli-error:{e}")
    return GateResult(True, conf, "ok")
