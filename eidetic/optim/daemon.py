"""The always-on optimizer supervisor: three cadences as DETERMINISTIC, individually-callable
tick methods (not a live thread). This is the deliberate, testable shape of the spec's "one
supervisor, three cadences sharing a feedback store":

  * hot     -- already inline in the request path (adaptive-k, conformal, cache, fan-out,
               fusion). The daemon does not touch the hot path.
  * idle    -- idle_tick(): token-free background work off a request. Replays the dev feedback
               buffer to update fusion weights, runs the Dreaming engine if present.
  * offline -- offline_sweep_command(): returns the exact dev-split sweep command an operator
               (or a cron) runs; the sweep writes a best_config.json artifact.

Atomic config swap is write-artifact + explicit reload (swap_config): the sweep writes a
best_config.json, swap_config applies its env + drops the cached Settings, so the NEXT engine
constructed reads the new config. It is NOT a hot-mutation of an already-running Engine (which
captured its frozen Settings at construction) -- the supported way to apply a tuned config to a
live server is to write the flags into .env and restart. Rebuild knobs are refused (the
OtterTune blacklist), so a swap can never trigger a ruinous index rebuild.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import REBUILD_KNOBS_ENV


class OptimizerDaemon:
    def __init__(self, engine):
        self.engine = engine

    # ---- idle cadence (token-free) ----------------------------------------
    def idle_tick(self, *, run_dream: bool = False) -> dict:
        """One idle pass: learn fusion weights from the dev feedback buffer, optionally run the
        Dreaming engine. Returns a small report. No model calls unless run_dream pulls in an
        LLM-enabled dream config (off by default)."""
        report: dict = {}
        try:
            report["fusion_weights"] = self.engine.learn_fusion_weights()
        except Exception as e:                 # idle work must never crash a caller
            report["fusion_error"] = str(e)
        if run_dream and hasattr(self.engine, "dream"):
            try:
                report["dream"] = self.engine.dream()
            except Exception as e:
                report["dream_error"] = str(e)
        return report

    # ---- offline cadence (operator/cron) ----------------------------------
    @staticmethod
    def offline_sweep_command(dataset: str = "locomo", subset: int = 50,
                              trials: int = 24, sampler: str = "tpe") -> str:
        """The exact dev-split sweep command. Always --split dev (integrity wall)."""
        return (f"python -m bench.sweep --sampler {sampler} --dataset {dataset} "
                f"--subset {subset} --trials {trials} --split dev")

    # ---- atomic config swap (between requests) ----------------------------
    @staticmethod
    def swap_config(best_config_path, *, apply: bool = True) -> dict:
        """Load a sweep's best_env artifact and apply it: set the env + drop the cached Settings,
        so the NEXT engine constructed reads the new config. This does NOT hot-mutate an
        already-running Engine (it captured frozen Settings at construction) -- to apply to a
        live server, write the flags into .env and restart. REBUILD knobs are refused (they
        require an explicit offline index rebuild)."""
        data = json.loads(Path(best_config_path).read_text())
        env = dict(data.get("best_env", {}))
        refused = {k: env.pop(k) for k in list(env) if k in REBUILD_KNOBS_ENV}
        if apply:
            for k, v in env.items():
                os.environ[k] = str(v)
            from ..config import get_settings
            get_settings.cache_clear()
        return {"applied": env, "refused_rebuild_knobs": refused}
