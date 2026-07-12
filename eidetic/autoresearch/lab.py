"""The DEV laboratory: a frozen witness the ratchet experiments against.

`ingest_once()` builds the lab store from DEV-split samples exactly the way the
neutral harness would (same adapter, same session granularity, same consolidate),
into a DEDICATED DATA_DIR under the lab root (store-isolation compliant), then
freezes it: a manifest pins the sample ids, per-group namespaces, and the store
file SHA. Every subsequent eval asserts the SHA is unchanged -- a trial that
mutated the witness is invalid by construction.

`eval_config(env_overlay, label)` answers the SAME dev questions ANSWER-ONLY (no
reset, no re-ingest) under the overlay, judged by the pinned bench judge, and
writes harness-format rows + a dev-split manifest -- so `bench.guard.run_guard`
consumes lab evals with zero adaptation. Mind knobs never touch ingest, which is
exactly why answer-only trials are honest for this search space (space.py refuses
ingest-side knobs for the same reason).

Integrity wall: every sample is asserted dev-split via bench.datasets.split_of
before ingest AND before each eval; namespaces use the harness benchmark pattern so
feedback from lab runs stays audit-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from ..models import now
from .space import assert_hypothesis_env_legal
from .operators import expected_stages

_LAB_SYSTEM = "eidetic-plus-full"


def _store_sha(data_dir: Path) -> str:
    """SHA over the lab store's sqlite + substrate bytes (WAL checkpointed first)."""
    h = hashlib.sha256()
    db = Path(data_dir) / "eidetic.sqlite"
    if db.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(db)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        except sqlite3.Error:
            pass
        h.update(db.read_bytes())
    substrate = Path(data_dir) / "substrate"
    if substrate.exists():
        for p in sorted(substrate.rglob("*")):
            if p.is_file():
                h.update(p.name.encode())
                h.update(p.read_bytes())
    return h.hexdigest()


class DevLab:
    def __init__(self, root: Path, *, dataset: str = "locomo", subset: int = 24,
                 variant: str = "longmemeval_s", sample_offset: int = 0):
        self.root = Path(root)
        self.dataset = dataset
        self.subset = int(subset)
        self.variant = variant
        self.sample_offset = int(sample_offset)
        self.data_dir = self.root / "data"
        self.manifest_path = self.root / "lab_manifest.json"
        self.evals_dir = self.root / "evals"

    # ---- environment plumbing ------------------------------------------------
    def _apply_env(self, overlay: dict[str, str]) -> dict[str, Optional[str]]:
        """Set DATA_DIR + overlay; return the previous values for restore."""
        from ..config import get_settings
        keys = set(overlay) | {"DATA_DIR"}
        previous = {k: os.environ.get(k) for k in keys}
        os.environ["DATA_DIR"] = str(self.data_dir)
        for k, v in overlay.items():
            if k == "EXPECT_STAGES":
                continue
            os.environ[k] = str(v)
        get_settings.cache_clear()
        return previous

    def _restore_env(self, previous: dict[str, Optional[str]]) -> None:
        from ..config import get_settings
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_settings.cache_clear()

    def _load_dev_samples(self):
        from bench.datasets import split_of
        from bench.run import load_samples
        samples = load_samples(self.dataset, self.subset, self.variant,
                               self.sample_offset, split="dev")
        bad = [s.sample_id for s in samples if split_of(s.sample_id) != "dev"]
        if bad:
            raise RuntimeError(f"integrity wall: non-dev samples in lab load: {bad[:5]}")
        if not samples:
            raise RuntimeError("lab loaded zero dev samples")
        return samples

    # ---- one-time witness build -----------------------------------------------
    def ingest_once(self, *, force: bool = False) -> dict:
        if self.manifest_path.exists() and not force:
            return json.loads(self.manifest_path.read_text())
        from bench.harness import _group_by_sessions
        from bench.run import make_system
        samples = self._load_dev_samples()
        groups = _group_by_sessions(samples)
        previous = self._apply_env({})
        try:
            system = make_system("eidetic-full")
            group_rows = []
            for gi, (sessions, qs) in enumerate(groups):
                ns = f"{_LAB_SYSTEM}-{self.dataset}-g{gi}-r0"     # harness pattern: audit-only
                system.reset(ns)
                for sess in sessions:
                    turns = [{"role": t.role, "content": t.content, "timestamp": t.timestamp}
                             for t in sess.turns]
                    system.ingest_session(ns, sess.session_id, turns, sess.session_time)
                system.consolidate(ns)
                group_rows.append({"namespace": ns,
                                   "sample_ids": [s.sample_id for s in qs],
                                   "sessions": len(sessions)})
        finally:
            self._restore_env(previous)
        manifest = {
            "dataset": self.dataset, "variant": self.variant, "subset": self.subset,
            "sample_offset": self.sample_offset, "split": "dev",
            "sample_ids": [s.sample_id for s in samples],
            "groups": group_rows,
            "store_sha": _store_sha(self.data_dir),
            "ingested_at": now(),
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=1))
        return manifest

    def assert_frozen(self) -> str:
        manifest = json.loads(self.manifest_path.read_text())
        sha = _store_sha(self.data_dir)
        if sha != manifest["store_sha"]:
            raise RuntimeError(
                f"lab witness mutated: store sha {sha[:12]} != frozen "
                f"{manifest['store_sha'][:12]} -- trials over a moved witness are void")
        return sha

    # ---- per-config evaluation --------------------------------------------------
    def eval_config(self, env_overlay: dict[str, str], label: str, *,
                    judge=None, overwrite: bool = False) -> Path:
        """Answer-only eval of the frozen lab under `env_overlay`. Returns the eval
        dir (guard-consumable). Resumable: an existing complete eval is returned
        as-is unless `overwrite`."""
        assert_hypothesis_env_legal(
            {k: v for k, v in env_overlay.items() if k != "EXPECT_STAGES"})
        out_dir = self.evals_dir / label
        log_path = out_dir / f"{_LAB_SYSTEM}__run0.jsonl"
        manifest = json.loads(self.manifest_path.read_text())
        if log_path.exists() and not overwrite:
            rows = [l for l in log_path.read_text().splitlines() if l.strip()]
            if len(rows) == len(manifest["sample_ids"]):
                return out_dir
        from bench.harness import QResult, _as_of_time, _judge_sample
        from bench.judge import Judge
        from bench.run import make_system
        from dataclasses import asdict
        self.assert_frozen()
        judge = judge or Judge()
        samples = self._load_dev_samples()
        by_id = {s.sample_id: s for s in samples}
        ns_of = {sid: g["namespace"] for g in manifest["groups"]
                 for sid in g["sample_ids"]}
        declared = expected_stages(env_overlay)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay = dict(env_overlay)
        if declared:
            overlay["RECALL_TRACE"] = "1"       # executed-stage honesty needs the trace
        previous = self._apply_env(overlay)
        stage_hits: dict[str, int] = {s: 0 for s in declared}
        try:
            system = make_system("eidetic-full")
            with open(log_path, "w") as fh:
                for sid in manifest["sample_ids"]:
                    s = by_id.get(sid)
                    if s is None:
                        raise RuntimeError(f"lab sample vanished from the dev load: {sid}")
                    ns = ns_of[sid]
                    t0 = time.perf_counter()
                    try:
                        ar = system.answer(ns, s.question, as_of=_as_of_time(s))
                        correct = _judge_sample(judge, s, ar.answer)
                        error = ""
                    except Exception as e:      # per-question resilience, like the harness
                        ar = None
                        correct = False
                        error = f"{type(e).__name__}: {str(e)[:200]}"
                    if declared and ar is not None:
                        trace = getattr(system.engine.retriever, "last_trace", None)
                        chans = set(getattr(trace, "enabled_channels", []) or [])
                        for st in declared:
                            if st in chans or (st == "dense" and chans):
                                stage_hits[st] += 1
                    qr = QResult(
                        system=_LAB_SYSTEM, dataset=s.dataset, category=s.category,
                        sample_id=s.sample_id, question=s.question, gold=s.gold,
                        predicted=(ar.answer if ar else ""),
                        correct=bool(correct and not error),
                        write_tokens=0,
                        query_tokens=(ar.context_tokens if ar else 0),
                        search_ms=(ar.search_ms if ar else 0.0),
                        e2e_ms=(ar.e2e_ms if ar else (time.perf_counter() - t0) * 1000.0),
                        abstained=bool(ar.abstained) if ar else False,
                        run_idx=0, n_sessions=len(s.sessions),
                        extra=(ar.extra if ar else {}), error=error,
                    )
                    fh.write(json.dumps(asdict(qr)) + "\n")
                    fh.flush()
        finally:
            self._restore_env(previous)
        if declared:
            dead = [s for s, n in stage_hits.items() if n == 0 and s != "dense"]
            if dead:
                raise RuntimeError(
                    f"pipeline declared stages {declared} but {dead} never appeared in "
                    "any recall trace -- a silently no-op pipeline cannot be trialed")
        (out_dir / "run_manifest.json").write_text(json.dumps({
            "split": "dev", "render_only": False, "systems": _LAB_SYSTEM,
            "dataset": self.dataset, "variant": self.variant,
            "sample_count": len(manifest["sample_ids"]),
            "samples_file": str(self.manifest_path),
            "env_overlay": {k: v for k, v in env_overlay.items()},
            "store_sha": manifest["store_sha"],
            "judge": (judge.describe() if hasattr(judge, "describe") else {}),
            "ts": now(),
        }, indent=1))
        return out_dir
