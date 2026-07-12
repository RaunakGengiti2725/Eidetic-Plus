"""Before/after replay: prove a promotion moved a REAL remembered failure --
on the SAME store, with NO new ingest, citing bytes that existed BEFORE the trial.

For each replayed query the report records status before (from the task) and after
(a fresh governed ask under the current champion), plus three mechanical honesty
checks a judge can re-run:

  same_witness      -- the store's record count and content-hash set are unchanged
  no_new_ingest     -- no memory's created_at postdates the replay start
  citations_preexist -- every citation hash on a new VERIFIED answer maps to a
                        record created BEFORE the promotion timestamp
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..models import Scope, now


def _hash_set(store, scope: Scope) -> set[str]:
    return {r.content_hash for r in store.all_records(scope)}


def replay_offline(engine, queries: list[dict], *, promotion_ts: Optional[float],
                   out_path: Optional[Path] = None) -> dict:
    """`queries`: [{query, namespace, agent_id?, project_id?, prior_status?}].
    Returns the replay report; writes it to `out_path` when given."""
    started = now()
    rows = []
    flips = 0
    for item in queries:
        scope = Scope(namespace=item.get("namespace", "default"),
                      agent_id=item.get("agent_id"), project_id=item.get("project_id"))
        before_hashes = _hash_set(engine.store, scope)
        ans = engine.ask(item["query"], scope=scope, verify=True, use_cache=False)
        after_hashes = _hash_set(engine.store, scope)
        new_records = [r for r in engine.store.all_records(scope)
                       if r.created_at >= started]
        citations_preexist = True
        if ans.verified and promotion_ts is not None:
            for cit in ans.citations:
                rec = engine.store.get_record(cit.memory_id)
                if rec is None or rec.created_at >= promotion_ts:
                    citations_preexist = False
        prior = item.get("prior_status", "ABSTAINED")
        flipped = prior != "VERIFIED" and ans.status.value == "VERIFIED"
        flips += int(flipped)
        rows.append({
            "query": item["query"][:200],
            "namespace": scope.namespace,
            "prior_status": prior,
            "status": ans.status.value,
            "flipped_to_verified": flipped,
            "citations": [{"memory_id": c.memory_id, "content_hash": c.content_hash}
                          for c in ans.citations][:6],
            "same_witness": before_hashes == after_hashes,
            "no_new_ingest": not new_records,
            "citations_preexist_promotion": citations_preexist,
            "note": (ans.note or "")[:160],
        })
    report = {
        "ts": started,
        "promotion_ts": promotion_ts,
        "n": len(rows),
        "flipped_to_verified": flips,
        "all_same_witness": all(r["same_witness"] for r in rows),
        "all_no_new_ingest": all(r["no_new_ingest"] for r in rows),
        "all_citations_preexist": all(r["citations_preexist_promotion"] for r in rows),
        "rows": rows,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=1))
    return report
