"""The brain spine: logic over the observation-only contracts in models.py.

Three pieces, all offline and deterministic (no model calls):

  * BrainEventLog -- the single in-memory improvement stream. Bounded ring buffer.
  * build_evidence_packets -- turn an Answer's citations + its RecallTrace into the
    portable EvidencePacket shape that proof, repair, and health all consume.
  * QualityGate -- the mandatory pass/fail checks every new brain connection must clear
    (flag-off preserves baseline, no raw mutation, no scope leak, no age bias, proof
    coverage non-decreasing, latency within budget).

INTEGRITY-WALL NOTE: BrainEventLog is in-memory and NON-LEARNING. The moment events are
persisted or fed to any learner, the log MUST route namespaces through
feedback.is_benchmark_namespace exactly like FeedbackBuffer, or it becomes a backdoor
around the dev/test integrity wall. This slice deliberately keeps events ephemeral.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

from .models import (Answer, BrainEvent, BrainEventType, EvidencePacket, NLILabel,
                     QualityGateResult, RecallTrace, Scope)


# --------------------------------------------------------------------------
# Event stream.
# --------------------------------------------------------------------------
class BrainEventLog:
    """A bounded, in-memory stream of BrainEvents. Append-only from the caller's view;
    oldest events are evicted past `capacity`. Non-learning: it never reads the store and
    never feeds a learner (see the integrity-wall note at the top of this module)."""

    def __init__(self, capacity: int = 4096):
        self.capacity = int(capacity)
        self._events: deque[BrainEvent] = deque(maxlen=self.capacity)

    def emit(self, event: BrainEvent) -> BrainEvent:
        self._events.append(event)
        return event

    def recent(self, limit: int = 50) -> list[BrainEvent]:
        """Most-recent events, newest first."""
        n = len(self._events)
        return list(self._events)[max(0, n - limit):][::-1]

    def by_type(self, etype: BrainEventType) -> list[BrainEvent]:
        return [e for e in self._events if e.type == etype]

    def counts(self, namespace: Optional[str] = None) -> dict[str, int]:
        """Per-type event counts, optionally scoped to one namespace (so scoped consumers like
        brain_health_score never mix activity across namespaces)."""
        out: dict[str, int] = {}
        for e in self._events:
            if namespace is not None and e.namespace != namespace:
                continue
            out[e.type.value] = out.get(e.type.value, 0) + 1
        return out

    def __len__(self) -> int:
        return len(self._events)


# --------------------------------------------------------------------------
# Evidence packets.
# --------------------------------------------------------------------------
def build_evidence_packets(answer: Answer,
                           trace: Optional[RecallTrace] = None) -> list[EvidencePacket]:
    """Convert an Answer's citations into EvidencePackets, enriching each with its retrieval
    paths/channel scores from `trace` when the trace belongs to THIS answer.

    Cache-safety: paths are attached only when `trace.query == answer.question`. On a
    semantic-cache hit (the answer was served without a fresh retrieve) or any stale/mismatched
    trace, packets are still built but carry no paths -- never misattributed evidence."""
    aligned = trace is not None and trace.query == answer.question
    packets: list[EvidencePacket] = []
    for c in answer.citations or []:
        paths: list[str] = []
        channel_scores: dict[str, float] = {}
        derived_from = ""
        if aligned:
            paths = trace.paths_for(c.memory_id)
            # Rank-based per-channel score (higher = nearer the top of that channel's list). paths
            # are a subset of channel_results keys, so the index lookup is always safe.
            channel_scores = {
                ch: float(len(trace.channel_results[ch]) - trace.channel_results[ch].index(c.memory_id))
                for ch in paths
            }
            derived_from = trace.gist_provenance.get(c.memory_id, "")
        packets.append(EvidencePacket(
            memory_id=c.memory_id, content_hash=c.content_hash, raw_uri=c.raw_uri,
            raw_span=c.snippet, valid_at=c.valid_at,
            retrieval_paths=paths, channel_scores=channel_scores,
            nli_label=c.nli_label, nli_score=c.nli_score,
            derived_from=derived_from,
        ))
    return packets


# --------------------------------------------------------------------------
# Quality gate.
# --------------------------------------------------------------------------
class QualityGate:
    """Pure pass/fail checks that make 'better' mandatory. Each check is a static predicate so
    a test (or a dev A/B) can feed it synthetic inputs; `evaluate` aggregates a subset into a
    QualityGateResult. Only checks that are actually run count toward `passed`."""

    @staticmethod
    def flag_off_preserves_baseline(baseline_ids: Iterable[str],
                                    flag_off_ids: Iterable[str]) -> bool:
        """The flags-off path must return the identical candidate ids in the identical order."""
        return list(baseline_ids) == list(flag_off_ids)

    @staticmethod
    def no_raw_mutation(before_hashes: Iterable[str], after_hashes: Iterable[str]) -> bool:
        """The immutable raw substrate must be byte-identical before and after the operation."""
        return set(before_hashes) == set(after_hashes)

    @staticmethod
    def no_scope_leak(result_scopes: Iterable[Scope], query_scope: Scope) -> bool:
        """Every returned record must be visible to the query scope (no cross-namespace bleed)."""
        return all(rs.visible_to(query_scope) for rs in result_scopes)

    @staticmethod
    def no_age_bias(recall_slope_per_year: float, latency_slope_ms_per_year: float,
                    recall_tol: float = 0.05, latency_tol: float = 1.0) -> bool:
        """Recall and p95 latency must stay flat vs memory age (the signature invariant). Reads
        the slopes engine.prove_age_independence already computes."""
        return abs(recall_slope_per_year) < recall_tol and abs(latency_slope_ms_per_year) < latency_tol

    @staticmethod
    def proof_coverage_not_dropped(before_grounded: int, after_grounded: int) -> bool:
        """A connection may not reduce the number of grounded (entailed) citations."""
        return after_grounded >= before_grounded

    @staticmethod
    def latency_within_budget(p95_ms: float, budget_ms: float) -> bool:
        return p95_ms <= budget_ms

    @classmethod
    def evaluate(cls, feature: str, *, checks: dict[str, bool]) -> QualityGateResult:
        """Aggregate already-computed boolean checks. `passed` is the AND over the supplied
        checks; an empty check set fails closed (you proved nothing)."""
        notes = [f"{k}: {'ok' if v else 'FAIL'}" for k, v in checks.items()]
        passed = bool(checks) and all(checks.values())
        return QualityGateResult(feature=feature, passed=passed, checks=dict(checks), notes=notes)
