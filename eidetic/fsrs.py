"""Component 5 (forgetting state): the FSRS / DSR model.

Each memory carries Difficulty, Stability, Retrievability. Retrievability follows
the FSRS-6 power-law forgetting curve (fits human memory better than Ebbinghaus's
exponential). This sets the INDEX-PRIORITY weight ONLY -- it never deletes the raw
record, and (critically) it is kept out of the cued-retrieval ranking path so that
recall@k stays age-independent.

Reawakening: a strong cue / confirmed recall resets retrievability and boosts
stability (reconsolidation + immune-affinity-maturation analogy). Because the index
entry was merely down-weighted (the myonuclear-retention analogy), re-promotion is O(1).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .models import now

if TYPE_CHECKING:
    from .models import FSRSState

# FSRS-6 forgetting-curve constants.
DECAY = -0.5
FACTOR = 19.0 / 81.0  # so that R(S days) == 0.9


def retrievability(elapsed_days: float, stability: float) -> float:
    """Power-law recall probability after `elapsed_days` for a memory of `stability`."""
    stability = max(stability, 1e-6)
    elapsed_days = max(elapsed_days, 0.0)
    return float((1.0 + FACTOR * elapsed_days / stability) ** DECAY)


def current_retrievability(state: "FSRSState", at: float | None = None) -> float:
    at = now() if at is None else at
    elapsed_days = max(0.0, (at - state.last_review) / 86400.0)
    return retrievability(elapsed_days, state.stability)


def init_state(importance: float, surprise: float, at: float | None = None) -> "FSRSState":
    """Initial DSR state from the write-time salience signals (Component 4)."""
    from .models import FSRSState

    at = now() if at is None else at
    salience = max(0.0, min(1.0, 0.5 * importance + 0.5 * surprise))
    # More salient -> more stable (slower forgetting) and easier (lower difficulty).
    stability = 1.0 + 9.0 * salience          # 1..10 days
    difficulty = max(1.0, min(10.0, 7.0 - 5.0 * importance))
    return FSRSState(
        stability=stability,
        difficulty=difficulty,
        retrievability=1.0,
        last_review=at,
        reps=0,
        lapses=0,
    )


def reinforce(state: "FSRSState", *, importance: float = 0.6, at: float | None = None) -> "FSRSState":
    """Reconsolidation strengthening on a confirmed recall / reawakening.

    Resets retrievability to 1 and grows stability. This is reversible re-promotion
    of a down-weighted memory -- never a re-creation."""
    at = now() if at is None else at
    r = current_retrievability(state, at)
    # Larger gain when the memory had decayed (desirable-difficulty / testing effect).
    growth = 1.0 + (1.0 + importance) * (1.0 - r)
    state.stability = state.stability * max(1.05, growth)
    state.difficulty = max(1.0, state.difficulty - 0.2)
    state.retrievability = 1.0
    state.last_review = at
    state.reps += 1
    return state


def lapse(state: "FSRSState", at: float | None = None) -> "FSRSState":
    """A contradiction / failed recall reduces stability (but never deletes)."""
    at = now() if at is None else at
    state.stability = max(1.0, state.stability * 0.5)
    state.difficulty = min(10.0, state.difficulty + 0.5)
    state.retrievability = current_retrievability(state, at)
    state.last_review = at
    state.lapses += 1
    return state


def decay(state: "FSRSState", at: float | None = None) -> "FSRSState":
    """Refresh the cached retrievability (called by the offline FSRS sweep). Pure
    down-weight: it lowers priority, it does not remove anything."""
    at = now() if at is None else at
    state.retrievability = current_retrievability(state, at)
    return state
