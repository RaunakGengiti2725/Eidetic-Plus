"""Retrieval-recall probe: does the record containing the GOLD answer enter the retriever's
top-k? Measured per category on already-scored (burned) windows -- reader-free, judge-free,
zero quota (query embeddings are cache-hits from the original runs). This isolates the
RETRIEVAL half of every accuracy gap: a question whose gold never reaches the reader can
never be answered, whatever the reader is.

Also measures the GRAPH-EXPANSION CEILING for multi-hop: recall@k when dense top-k is
augmented with records reachable via claim-graph edges from the dense candidates' entities
(one hop of spreading activation). A large ceiling gap = evidence the spreading-activation
channel is worth wiring; no gap = it is not, however good it sounds.

Usage (one window per process -- DATA_DIR is read at import):
  DATA_DIR=artifacts/holdout_rotation_r9_codex/data \
    .venv/bin/python -m bench.retrieval_recall_probe \
    artifacts/holdout_rotation_r9_codex --k 10 --out .../retrieval_recall.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _gold_parts(gold: str) -> list[str]:
    """Split a multi-part gold ('painting, hiking, and cooking') into containment parts;
    single-part golds return themselves. Parts under 3 chars are dropped (un-checkable)."""
    parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", str(gold or ""))
             if p.strip()]
    parts = [_norm(p) for p in parts if len(_norm(p)) >= 3]
    return parts or ([_norm(gold)] if len(_norm(gold)) >= 3 else [])


def _covered(parts: list[str], texts: list[str]) -> bool:
    """Every gold part appears in the UNION of the texts."""
    blob = " ".join(_norm(t) for t in texts)
    return bool(parts) and all(p in blob for p in parts)


def _entities_of(rec) -> set[str]:
    return {e.lower() for e in (getattr(rec, "entities", None) or []) if e}


def run_window(window: Path, *, k: int) -> dict:
    from eidetic.engine import Engine
    from eidetic.models import Scope

    rows_path = None
    for cand in sorted(window.glob("eidetic-plus-full__run0.jsonl")):
        rows_path = cand
    if rows_path is None:
        raise SystemExit(f"no eidetic-plus-full__run0.jsonl in {window}")
    rows = [json.loads(l) for l in open(rows_path) if l.strip()]

    eng = Engine()
    # namespace per row via the committed candidate ids (deterministic, no guessing)
    id_to_ns: dict[str, str] = {}
    ns_records: dict[str, list] = {}
    import sqlite3
    con = sqlite3.connect(str(Path(eng.settings.data_dir) / "eidetic.sqlite"))
    namespaces = [row[0] for row in con.execute("SELECT DISTINCT namespace FROM memories")]
    con.close()
    for ns in namespaces:
        recs = eng.store.all_records(Scope(namespace=ns))
        ns_records[ns] = recs
        for r in recs:
            id_to_ns[r.memory_id] = ns

    per_cat = defaultdict(lambda: {"answerable": 0, "dense_hit": 0, "expand_hit": 0,
                                   "rows": 0, "no_namespace": 0, "not_in_store": 0})
    details = []
    for row in rows:
        cat = str(row.get("category") or "?")
        stats = per_cat[cat]
        stats["rows"] += 1
        q = row.get("question") or ""
        gold = str(row.get("gold") or "")
        extra = row.get("extra") or {}
        cand_ids = [c for c in (extra.get("candidate_memory_ids") or []) if c]
        ns = next((id_to_ns[c] for c in cand_ids if c in id_to_ns), None)
        if ns is None:
            stats["no_namespace"] += 1
            continue
        recs = ns_records[ns]
        parts = _gold_parts(gold)
        # answerable-by-containment: the union of ALL records in the namespace covers the gold
        if not _covered(parts, [r.text or "" for r in recs]):
            stats["not_in_store"] += 1
            continue
        stats["answerable"] += 1
        scope = Scope(namespace=ns)
        cands = eng.retriever.retrieve(q, at=None, scope=scope)
        dense_top = [c.record for c in cands[:k] if getattr(c, "record", None) is not None]
        dense_hit = _covered(parts, [r.text or "" for r in dense_top])
        if dense_hit:
            stats["dense_hit"] += 1
        # one-hop graph expansion ceiling: entities of dense top-k -> edges -> records
        expand = list(dense_top)
        if not dense_hit:
            seed_entities: set[str] = set()
            for r in dense_top:
                seed_entities |= _entities_of(r)
            if seed_entities:
                edges = eng.store.all_edges(scope, include_inferred=False)
                linked_ids: set[str] = set()
                for e in edges:
                    src = str(e.src or "").lower()
                    dst = str(e.dst or "").lower()
                    if src in seed_entities or dst in seed_entities:
                        if e.source_memory_id:
                            linked_ids.add(e.source_memory_id)
                by_id = {r.memory_id: r for r in recs}
                expand += [by_id[i] for i in linked_ids
                           if i in by_id and by_id[i] not in expand]
        expand_hit = dense_hit or _covered(parts, [r.text or "" for r in expand])
        if expand_hit:
            stats["expand_hit"] += 1
        if not dense_hit:
            details.append({"sample_id": row.get("sample_id"), "category": cat,
                            "gold": gold[:60], "recovered_by_expansion": bool(expand_hit),
                            "expansion_pool": len(expand)})

    totals = {"answerable": 0, "dense_hit": 0, "expand_hit": 0, "rows": 0,
              "no_namespace": 0, "not_in_store": 0}
    for c in per_cat.values():
        for key in totals:
            totals[key] += c[key]
    return {
        "window": str(window),
        "k": k,
        "per_category": {c: dict(v) for c, v in sorted(per_cat.items())},
        "totals": totals,
        "dense_recall": round(totals["dense_hit"] / totals["answerable"], 4)
        if totals["answerable"] else None,
        "expansion_ceiling_recall": round(totals["expand_hit"] / totals["answerable"], 4)
        if totals["answerable"] else None,
        "misses": details[:40],
        "method": ("gold-part containment over top-k record texts; answerable = gold parts "
                   "covered by the union of the question's own namespace records. Expansion "
                   "ceiling = dense top-k plus records linked by one claim-graph hop from the "
                   "dense candidates' entities (upper bound for a spreading-activation channel; "
                   "NOT a product path). Reader-free, judge-free."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("window")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rep = run_window(Path(args.window), k=args.k)
    text = json.dumps(rep, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
