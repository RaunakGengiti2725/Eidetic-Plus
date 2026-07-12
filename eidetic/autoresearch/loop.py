"""The improve verb: curiosity first, then at most `max_trials` guarded experiments.

One drain = one metabolic cycle of the organism:
  1. token-free map refresh (enumerators over the current store)
  2. curiosity wave over the frontier (real prove-path probes; map + agenda update)
  3. pop the highest-priority task -> propose ONE untried hypothesis -> run_trial
  4. on ACCEPT: promote, then replay the task's own query same-store to close the loop
  5. map delta snapshot (the auditable "did the unknown set move" artifact)

Every stage is best-effort isolated: a failed trial never loses the curiosity
results, and everything is resumable (ledger-append + idempotent evals).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ..models import Scope, now
from .agenda import ResearchAgenda
from .proposer import propose_one
from .registry import ChampionRegistry, ResearchMemory
from .trials import load_trials, run_trial


def research_status(engine, *, last_n: int = 5) -> dict:
    """The MCP/inspection surface: agenda + champion + last trials + map counts.
    NO answer text, NO drafts -- method metadata only."""
    root = Path(engine.settings.autoresearch_dir)
    agenda = engine.research_agenda
    registry = ChampionRegistry(root)
    champ = registry.load()
    trials = load_trials(root / "trials.jsonl")
    scope = Scope()
    kmap = getattr(engine, "knowledge_map_store", None)
    map_counts = kmap.counts(scope) if kmap is not None else {}
    return {
        "agenda": agenda.stats() if agenda is not None else {},
        "champion": {"champion_id": champ.get("champion_id"),
                     "dev_acc": champ.get("dev_acc"),
                     "tier": champ.get("tier"),
                     "promoted_at": champ.get("promoted_at"),
                     "env_keys": sorted((champ.get("env") or {}).keys())},
        "trials_total": len(trials),
        "last_trials": [
            {k: t.get(k) for k in ("trial_id", "tier", "decision", "delta_pp",
                                   "mcnemar_p", "paired_n", "reason", "ts")}
            for t in trials[-last_n:]
        ],
        "knowledge_map": map_counts,
    }


def drain(engine, *, scope: Optional[Scope] = None, max_trials: Optional[int] = None,
          max_probes: Optional[int] = None, dry_run: bool = False,
          judge=None, lab=None) -> dict:
    """One improve cycle. `dry_run` proposes without running (zero model calls
    beyond nothing -- the proposal itself is rule-based)."""
    settings = engine.settings
    scope = scope or Scope()
    root = Path(settings.autoresearch_dir)
    agenda: Optional[ResearchAgenda] = engine.research_agenda
    if agenda is None:
        return {"ts": now(), "disabled": "AUTORESEARCH=0 (no agenda constructed)"}
    registry = ChampionRegistry(root)
    memory = ResearchMemory(root / "research_memory.jsonl")
    kmap = getattr(engine, "knowledge_map_store", None)
    report: dict = {"ts": now(), "dry_run": dry_run}

    # 1. token-free map refresh
    if kmap is not None and settings.epistemic_map_enabled:
        try:
            report["map_rebuild"] = kmap.rebuild(engine.store, scope)
        except Exception as e:
            report["map_rebuild_error"] = f"{type(e).__name__}: {str(e)[:160]}"

    map_before = kmap.counts(scope) if kmap is not None else {}
    report["map_before"] = map_before

    # 2. curiosity wave (model calls -- skipped on dry runs)
    if not dry_run and kmap is not None:
        from ..epistemic.curiosity import run_curiosity
        probes = settings.curiosity_max_probes_per_tick if max_probes is None else max_probes
        if probes > 0:
            try:
                report["curiosity"] = run_curiosity(
                    engine, scope, max_probes=probes, agenda=agenda,
                    probes_log=root / "probes.jsonl")
            except Exception as e:
                report["curiosity_error"] = f"{type(e).__name__}: {str(e)[:160]}"

    # 3./4. guarded trials
    n_trials = settings.autoresearch_max_trials_per_tick if max_trials is None else max_trials
    trials_run = []
    for _ in range(max(0, int(n_trials))):
        popped = agenda.pop_highest_priority()
        if popped is None:
            break
        key, task = popped
        hyp = propose_one(task, memory)
        if hyp is None:
            agenda.mark(key, "dropped")
            trials_run.append({"task": task.query[:80], "skipped": "no untried hypothesis"})
            continue
        if dry_run:
            agenda.mark(key, "queued")           # put it back; nothing ran
            trials_run.append({"task": task.query[:80], "dry_run_proposal": hyp.describe()})
            continue
        if lab is None:
            agenda.mark(key, "queued")
            trials_run.append({"task": task.query[:80],
                               "skipped": "no lab attached (pass lab= to drain)"})
            break
        try:
            trial = run_trial(hyp, lab=lab, registry=registry, memory=memory,
                              trials_path=root / "trials.jsonl", judge=judge,
                              map_counts=map_before)
            agenda.mark(key, "done")
            entry = {"task": task.query[:80], "trial_id": trial.trial_id,
                     "decision": trial.decision, "delta_pp": trial.delta_pp,
                     "mcnemar_p": trial.mcnemar_p}
            if trial.decision == "ACCEPT":
                registry.apply(apply_env=True)   # next Engine reads the new mind
                from .replay import replay_offline
                entry["replay"] = replay_offline(
                    engine, [{"query": task.query, "namespace": task.namespace,
                              "agent_id": task.agent_id, "project_id": task.project_id,
                              "prior_status": "ABSTAINED"}],
                    promotion_ts=now(),
                    out_path=root / f"replay_{trial.trial_id}.json")
            trials_run.append(entry)
        except Exception as e:
            agenda.mark(key, "queued")           # a crashed trial goes back to the queue
            trials_run.append({"task": task.query[:80],
                               "error": f"{type(e).__name__}: {str(e)[:200]}"})
        if settings.autoresearch_cooldown_sec > 0:
            time.sleep(settings.autoresearch_cooldown_sec)
    report["trials"] = trials_run

    # 5. map delta
    if kmap is not None:
        map_after = kmap.counts(scope)
        report["map_after"] = map_after
        report["map_delta"] = {
            "unknown": map_after.get("unknown_n", 0) - map_before.get("unknown_n", 0),
            "contested": map_after.get("contested_n", 0) - map_before.get("contested_n", 0),
            "known": map_after.get("known_n", 0) - map_before.get("known_n", 0),
        }
        try:
            root.mkdir(parents=True, exist_ok=True)
            with open(root / "improve_ticks.jsonl", "a") as fh:
                fh.write(json.dumps({k: report[k] for k in
                                     ("ts", "map_before", "map_after", "map_delta")}) + "\n")
        except OSError:
            pass
    return report
