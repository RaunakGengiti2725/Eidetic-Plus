"""ChampionRegistry: the currently-promoted mind config + append-only promotion history.

champion.json is the single source of truth for "what mind is live"; promotions.jsonl
is the immutable history (who won, when, by how much, from which trial). Applying a
champion delegates to OptimizerDaemon.swap_config -- same env+settings-cache contract,
same rebuild-knob refusal -- plus the proof-DNA wall on top.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..models import now
from .space import assert_hypothesis_env_legal


class ChampionRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.champion_path = self.root / "champion.json"
        self.promotions_path = self.root / "promotions.jsonl"

    def load(self) -> dict:
        if not self.champion_path.exists():
            return {"champion_id": "baseline", "env": {}, "dev_acc": None,
                    "paired_n": 0, "promoted_at": None, "trial_id": None}
        try:
            return json.loads(self.champion_path.read_text())
        except (ValueError, OSError):
            return {"champion_id": "baseline", "env": {}, "dev_acc": None,
                    "paired_n": 0, "promoted_at": None, "trial_id": None}

    @property
    def champion_id(self) -> str:
        return str(self.load().get("champion_id", "baseline"))

    def promote(self, *, trial_id: str, env: dict, dev_acc: float, paired_n: int,
                tier: str, describe: dict) -> dict:
        """Write the new champion + append the promotion. The env overlay is validated
        against the proof-DNA wall FIRST -- an illegal promotion cannot be persisted."""
        assert_hypothesis_env_legal(env)
        champion = {
            "champion_id": trial_id,
            "env": {k: str(v) for k, v in env.items()},
            "dev_acc": float(dev_acc),
            "paired_n": int(paired_n),
            "promoted_at": now(),
            "trial_id": trial_id,
            "tier": tier,
            "hypothesis": describe,
        }
        self.champion_path.write_text(json.dumps(champion, indent=2))
        with open(self.promotions_path, "a") as fh:
            fh.write(json.dumps(champion) + "\n")
        return champion

    def promotions(self) -> list[dict]:
        if not self.promotions_path.exists():
            return []
        return [json.loads(l) for l in self.promotions_path.read_text().splitlines()
                if l.strip()]

    def apply(self, *, apply_env: bool = True) -> dict:
        """Apply the champion env (swap_config semantics: set env + drop the Settings
        cache so the NEXT engine reads it). Returns what was applied/refused."""
        champ = self.load()
        env = dict(champ.get("env") or {})
        if not env:
            return {"applied": {}, "refused_rebuild_knobs": {}, "champion_id":
                    champ.get("champion_id", "baseline")}
        import tempfile
        from ..optim.daemon import OptimizerDaemon
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"best_env": env}, f)
            path = f.name
        out = OptimizerDaemon.swap_config(path, apply=apply_env)
        out["champion_id"] = champ.get("champion_id", "baseline")
        return out


class ResearchMemory:
    """Compact lessons from past trials (the memory-in-the-loop pattern): the loop
    must never rediscover a dead end. Append-only jsonl; block_repeat() is the
    proposer's gate."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, hypothesis_key: str, decision: str, delta_pp: float,
               mcnemar_p: Optional[float], failure_class: str, tier: str,
               note: str = "") -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps({
                "ts": now(), "hypothesis_key": hypothesis_key, "tier": tier,
                "decision": decision, "delta_pp": round(float(delta_pp), 3),
                "mcnemar_p": mcnemar_p, "failure_class": failure_class,
                "note": note[:200],
            }) + "\n")

    def lessons(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def block_repeat(self, hypothesis_key: str) -> bool:
        """True when this exact hypothesis was already tried (either verdict): a
        REJECT must not be retried against the same champion, and an ACCEPT is
        already IN the champion."""
        return any(l.get("hypothesis_key") == hypothesis_key for l in self.lessons())

    def rejected_keys(self) -> set[str]:
        return {l["hypothesis_key"] for l in self.lessons() if l.get("decision") == "REJECT"}
