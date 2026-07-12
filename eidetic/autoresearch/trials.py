"""run_trial: one hypothesis, one paired DEV experiment, one guard verdict, one
ledger row. The Karpathy ratchet with McNemar physics.

Champion eval is computed once per champion and cached (the paired control);
the challenger eval runs under the hypothesis env overlay on the SAME frozen lab
rows; `bench.guard.run_guard` (dev-artifact checks + pooled McNemar) decides.
ACCEPT -> ChampionRegistry.promote (proof-DNA wall re-validated inside).
Either way the trial is appended to trials.jsonl and ResearchMemory learns the
lesson so the proposer never retries the exact hypothesis.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ..models import now
from .lab import DevLab
from .registry import ChampionRegistry, ResearchMemory
from .types import ResearchHypothesis, ResearchTrial, validate_trial_row


def _champion_label(champion_id: str) -> str:
    return f"champion__{champion_id[:24]}"


def ensure_champion_eval(lab: DevLab, registry: ChampionRegistry, *, judge=None) -> Path:
    champ = registry.load()
    label = _champion_label(champ["champion_id"])
    return lab.eval_config(dict(champ.get("env") or {}), label, judge=judge)


def run_trial(hypothesis: ResearchHypothesis, *, lab: DevLab,
              registry: ChampionRegistry, memory: ResearchMemory,
              trials_path: Path, judge=None, min_delta_pp: float = 1.0,
              alpha: float = 0.05, map_counts: Optional[dict] = None) -> ResearchTrial:
    from bench.guard import run_guard
    t0 = time.perf_counter()
    champ = registry.load()
    champion_dir = ensure_champion_eval(lab, registry, judge=judge)

    overlay = dict(champ.get("env") or {})
    overlay.update(hypothesis.env_overlay())          # hypothesis mutates ON TOP of champion
    label = f"trial__{hypothesis.tier}__{hypothesis.key}"
    challenger_dir = lab.eval_config(overlay, label, judge=judge)

    verdict = run_guard(champion_dir, challenger_dir, system="eidetic-plus-full",
                        min_delta_pp=min_delta_pp, alpha=alpha)
    decision = "ACCEPT" if verdict.get("accept") else "REJECT"
    trial = ResearchTrial(
        trial_id=f"tr_{int(now())}_{hypothesis.key}",
        hypothesis=hypothesis,
        champion_id=champ["champion_id"],
        challenger_env={k: str(v) for k, v in overlay.items() if k != "EXPECT_STAGES"},
        dev_score=float(verdict.get("challenger_acc", 0.0)),
        champion_score=float(verdict.get("champion_acc", 0.0)),
        delta_pp=float(verdict.get("delta_pp", 0.0)),
        mcnemar_p=verdict.get("mcnemar_p"),
        paired_n=int(verdict.get("paired_n", 0)),
        decision=decision,
        reason=str(verdict.get("reason", "")),
        artifact_dir=str(challenger_dir),
        map_before=dict(map_counts or {}),
        duration_s=time.perf_counter() - t0,
    )

    if decision == "ACCEPT":
        registry.promote(trial_id=trial.trial_id, env=trial.challenger_env,
                         dev_acc=trial.dev_score, paired_n=trial.paired_n,
                         tier=hypothesis.tier, describe=hypothesis.describe())
    memory.record(hypothesis_key=hypothesis.key, decision=decision,
                  delta_pp=trial.delta_pp, mcnemar_p=trial.mcnemar_p,
                  failure_class=hypothesis.failure_class.value, tier=hypothesis.tier,
                  note=trial.reason)
    append_trial(trials_path, trial)
    return trial


def append_trial(trials_path: Path, trial: ResearchTrial) -> None:
    row = trial.to_row()
    problems = validate_trial_row(row)
    if problems:
        raise RuntimeError(f"refusing to append an invalid trial row: {problems}")
    trials_path = Path(trials_path)
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trials_path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def load_trials(trials_path: Path) -> list[dict]:
    trials_path = Path(trials_path)
    if not trials_path.exists():
        return []
    return [json.loads(l) for l in trials_path.read_text().splitlines() if l.strip()]
