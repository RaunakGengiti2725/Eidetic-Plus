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

import threading
import time
from typing import Optional

from .models import Answer, BrainEventType, MemoryRecord, Scope


class LifecycleController:
    def __init__(self, engine):
        self.engine = engine
        self._auto_sleep_guard = threading.Lock()
        self._auto_sleep_locks: dict[str, threading.Lock] = {}
        self._auto_sleep_last_started: dict[str, float] = {}
        self._auto_sleep_last_schedule: dict[str, dict] = {}
        self._auto_sleep_last_report: dict[str, dict] = {}
        self._auto_sleep_last_error: dict[str, str] = {}

    @staticmethod
    def _scope_key(scope: Optional[Scope]) -> str:
        return (scope or Scope()).key()

    # ---- wake hooks (thin; the engine already does the core writes) -------------------------
    def after_ingest(self, record: MemoryRecord, scope: Optional[Scope] = None) -> MemoryRecord:
        """Hook called after a wake write.

        Fast host writes can return after immutable capture + embedding, then let the lifecycle
        thread drain typed extraction in the background. Neutral baseline keeps this disabled; the
        metabolism profile enables it for autonomous host-agent memory.
        """
        if record.metadata.get("pending_consolidation"):
            self.maybe_auto_sleep(scope or record.scope)
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
        out: dict = {"consolidate_pending": self.engine.consolidate_pending(
            scope=scope,
            score_importance=bool(getattr(self.engine.settings, "sleep_score_importance", True)),
        )}
        out["dream"] = self.engine.dream(scope=scope)
        if llm_summaries:
            out["consolidate"] = self.engine.consolidate(scope=scope)
        # Epistemic map refresh on the sleep cadence: deterministic enumerators over the
        # store, ZERO model calls, own derived sqlite -- sleep stays token-free. Curiosity
        # probes/trials NEVER run here; they live behind the explicit improve verb.
        kmap = getattr(self.engine, "knowledge_map_store", None)
        if kmap is not None:
            try:
                out["epistemic_map"] = kmap.rebuild(self.engine.store, scope)
            except Exception as e:
                out["epistemic_map_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return out

    # ---- idle (background learning / embedding warm-up) --------------------------------------
    def idle_tick(self, *, run_dream: bool = False, scope: Optional[Scope] = None) -> dict:
        """One idle cadence: learn fusion weights, drain background work, optionally warm prefetch,
        then attach a connection-effectiveness snapshot from the brain event stream."""
        from .optim.daemon import OptimizerDaemon
        report = OptimizerDaemon(self.engine).idle_tick(run_dream=run_dream)
        try:
            report["reembed_drain"] = self.engine.drain_reembed_queue()   # S1 deferred re-embed
        except Exception:
            pass
        if (self.engine.settings.markov_prefetch_enabled
                and self.engine.settings.flow_warmup_enabled):
            try:
                report["prefetch_warmup"] = self.engine.warmup_predicted_prefetch(scope=scope)
            except Exception as exc:
                report["prefetch_warmup"] = {"enabled": True, "warmed": 0, "error": str(exc)}
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
        proposals = out.get("proposals", []) or [] if isinstance(out, dict) else []
        if proposals:
            self.engine._brain(BrainEventType.REPAIR_PROPOSED, namespace=scope.namespace,
                               proposals=len(proposals))
        # MemMA proposals feed the research agenda (origin=repair, below frontier cells):
        # a proposal names a query-shaped gap; the ratchet decides if any mind change helps.
        agenda = getattr(self.engine, "research_agenda", None)
        if agenda is not None and proposals:
            try:
                from .autoresearch.types import FailureClass, ResearchTask
                for p in proposals[:8]:
                    query = str(p.get("query") or p.get("probe") or p.get("fact") or "").strip()
                    if not query:
                        continue
                    agenda.enqueue(ResearchTask(
                        query=query, namespace=scope.namespace, agent_id=scope.agent_id,
                        project_id=scope.project_id,
                        failure_class=FailureClass.REPAIR_PROPOSAL, origin="repair"))
            except Exception:
                pass
        return out

    # ---- autonomous host write drain --------------------------------------------------------
    def maybe_auto_sleep(self, scope: Optional[Scope] = None) -> dict:
        """Schedule one bounded background sleep for pending host writes in this scope.

        This is deliberately best-effort and coalesced: the write path never waits for extraction,
        and one running auto-sleep per scope is enough to drain all pending records visible there.
        """
        scope = scope or Scope()
        settings = self.engine.settings
        interval = max(0.0, float(getattr(settings, "host_auto_sleep_min_interval_sec", 0.0) or 0.0))
        if not getattr(settings, "host_auto_sleep_enabled", False):
            return {
                "enabled": False,
                "scheduled": False,
                "running": False,
                "reason": "disabled",
                "scope": scope.model_dump(),
                "min_interval_sec": interval,
            }
        key = self._scope_key(scope)
        now_s = time.time()
        with self._auto_sleep_guard:
            lock = self._auto_sleep_locks.setdefault(key, threading.Lock())
            if lock.locked():
                report = {
                    "enabled": True,
                    "scheduled": False,
                    "running": True,
                    "reason": "already_running",
                    "scope": scope.model_dump(),
                    "min_interval_sec": interval,
                }
                self._auto_sleep_last_schedule[key] = report
                return report
            last = self._auto_sleep_last_started.get(key)
            if last is not None and interval > 0.0 and now_s - last < interval:
                report = {
                    "enabled": True,
                    "scheduled": False,
                    "running": False,
                    "reason": "interval",
                    "scope": scope.model_dump(),
                    "min_interval_sec": interval,
                    "seconds_until_next": max(0.0, interval - (now_s - last)),
                }
                self._auto_sleep_last_schedule[key] = report
                return report
            if not lock.acquire(blocking=False):
                report = {
                    "enabled": True,
                    "scheduled": False,
                    "running": True,
                    "reason": "already_running",
                    "scope": scope.model_dump(),
                    "min_interval_sec": interval,
                }
                self._auto_sleep_last_schedule[key] = report
                return report
            self._auto_sleep_last_started[key] = now_s
            report = {
                "enabled": True,
                "scheduled": True,
                "running": True,
                "reason": "scheduled",
                "scope": scope.model_dump(),
                "min_interval_sec": interval,
                "started_at": now_s,
            }
            self._auto_sleep_last_schedule[key] = report

        t = threading.Thread(
            target=self._run_auto_sleep,
            args=(scope, key, lock),
            name=f"eidetic-auto-sleep-{scope.namespace}",
            daemon=True,
        )
        t.start()
        return report

    def _run_auto_sleep(self, scope: Scope, key: str, lock: threading.Lock) -> None:
        try:
            out = {
                "consolidate_pending": self.engine.consolidate_pending(
                    scope=scope,
                    score_importance=getattr(
                        self.engine.settings, "host_auto_sleep_score_importance", False),
                ),
                "dream": self.engine.dream(scope=scope),
            }
            with self._auto_sleep_guard:
                self._auto_sleep_last_report[key] = out
                self._auto_sleep_last_error.pop(key, None)
        except Exception as exc:
            try:
                self.engine._degraded("host-auto-sleep", exc)
            finally:
                with self._auto_sleep_guard:
                    self._auto_sleep_last_error[key] = str(exc)
        finally:
            lock.release()

    def auto_sleep_status(self, scope: Optional[Scope] = None) -> dict:
        scope = scope or Scope()
        key = self._scope_key(scope)
        settings = self.engine.settings
        interval = max(0.0, float(getattr(settings, "host_auto_sleep_min_interval_sec", 0.0) or 0.0))
        pending = sum(
            1 for r in self.engine.store.all_records(scope)
            if r.metadata.get("pending_consolidation")
        )
        with self._auto_sleep_guard:
            lock = self._auto_sleep_locks.get(key)
            return {
                "enabled": bool(getattr(settings, "host_auto_sleep_enabled", False)),
                "scope": scope.model_dump(),
                "pending_consolidation": pending,
                "running": bool(lock.locked()) if lock is not None else False,
                "min_interval_sec": interval,
                "score_importance": bool(
                    getattr(settings, "host_auto_sleep_score_importance", False)),
                "last_started_at": self._auto_sleep_last_started.get(key),
                "last_schedule": self._auto_sleep_last_schedule.get(key),
                "last_report": self._auto_sleep_last_report.get(key),
                "last_error": self._auto_sleep_last_error.get(key),
            }
