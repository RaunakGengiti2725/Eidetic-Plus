"""LifecycleController: one shared execution path for wake / sleep / idle / repair.

Connected Brain Loop, Phase 1. The audit found API `consolidate` (LLM summaries) and MCP
`consolidate` (token-free `dream()`) had drifted apart. This controller is the single place
those semantics are defined, so both transports and the engine route brain-loop side effects
through ONE coordinator instead of growing divergent behavior.

It is THIN: it owns no storage and no model logic; it only sequences existing engine methods
and emits BrainEvents. Every method is a safe pass-through when the brain flags are off, so
adding it changes nothing about the baseline read/write path.

Sleep semantics, made explicit (the composite the plan asks for):
    sleep() = consolidate_pending()   # process LLM-free fast-written records (facts/events/types)
            -> dream()                # token-free derived layer (replay / infer / multires gists)
            -> consolidate()          # OPTIONAL LLM semantic summaries (only when asked)
`consolidate_pending` is a no-op (no model call) when nothing is pending, so an idle sleep over a
quiet scope is free and offline.
"""
from __future__ import annotations

from typing import Optional

from .models import Answer, BrainEventType, MemoryRecord, Scope


class LifecycleController:
    def __init__(self, engine):
        self.engine = engine

    # ---- wake hooks (thin; the engine already does the core writes) -------------------------
    def after_ingest(self, record: MemoryRecord, scope: Optional[Scope] = None) -> MemoryRecord:
        """Hook called after a wake write. The engine has already emitted MEMORY_INGESTED; this is
        the single seam where deferred consolidation could be queued. Pass-through today."""
        return record

    def after_recall(self, answer: Answer, scope: Optional[Scope] = None) -> Answer:
        """Hook called after a wake read. The engine already reconsolidated + emitted recall
        events; this updates the channel-win ledger from the RecallTrace (gated). Pass-through
        when BRAIN_EVENTS is off."""
        if self.engine.settings.brain_events_enabled:
            try:
                self.engine.record_channel_wins(answer)
            except Exception:
                pass
        return answer

    # ---- sleep (the unified composite) ------------------------------------------------------
    def sleep(self, scope: Optional[Scope] = None, *, llm_summaries: bool = False) -> dict:
        """The one sleep path: consolidate_pending -> dream -> optional LLM summaries. Returns a
        per-phase report. Free + offline when nothing is pending and llm_summaries is False."""
        scope = scope or Scope()
        out: dict = {"consolidate_pending": self.engine.consolidate_pending(scope=scope)}
        out["dream"] = self.engine.dream(scope=scope)
        if llm_summaries:
            out["consolidate"] = self.engine.consolidate(scope=scope)
        return out

    # ---- idle (token-free background learning) ----------------------------------------------
    def idle_tick(self, *, run_dream: bool = False, scope: Optional[Scope] = None) -> dict:
        """One idle cadence: learn fusion weights from the dev feedback buffer (+ optional dream),
        then attach a connection-effectiveness snapshot from the brain event stream."""
        from .optim.daemon import OptimizerDaemon
        report = OptimizerDaemon(self.engine).idle_tick(run_dream=run_dream)
        try:
            report["reembed_drain"] = self.engine.drain_reembed_queue()   # S1 deferred re-embed
        except Exception:
            pass
        try:
            report["connection_effectiveness"] = self.engine.connection_effectiveness()
        except Exception:
            pass
        return report

    # ---- repair (proposal-only by default; guarded apply behind its own flag) ---------------
    def repair_tick(self, scope: Optional[Scope] = None) -> dict:
        """Run the MemMA self-repair sweep (proposal-only unless DREAM_REPAIR is on). Emits a
        REPAIR_PROPOSED event with the proposal count when any are produced."""
        scope = scope or Scope()
        out = self.engine.dream_repair(scope=scope)
        n = len(out.get("proposals", []) or []) if isinstance(out, dict) else 0
        if n:
            self.engine._brain(BrainEventType.REPAIR_PROPOSED, namespace=scope.namespace,
                               proposals=n)
        return out
