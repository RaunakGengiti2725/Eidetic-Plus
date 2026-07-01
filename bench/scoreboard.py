"""Render the per-category accuracy scoreboard + cost table + latency table from the raw
run logs. RENDERS ONLY FROM REAL LOGS -- if there are no logs it writes an explicit
"pending run" placeholder, never invented numbers (a number that does not reproduce does
not exist).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Optional

import numpy as np

from .fingerprints import log_fingerprint
from .harness import load_logs

_LME_ORDER = ["single-session-user", "single-session-assistant", "single-session-preference",
              "multi-session", "knowledge-update", "temporal-reasoning"]
_LOCOMO_ORDER = ["single-hop", "multi-hop", "temporal", "open-domain"]
_MAB_ORDER = ["factconsolidation", "fact-consolidation", "eventqa", "event-qa"]

_BEAM_NOTE = (
    "> **Honest frontier note.** Cross-session contradiction resolution at BEAM scale "
    "(1M->10M tokens) is unsolved across the field (best public BEAM-1M contradiction "
    "~0.357). Eidetic-Plus targets the LongMemEval + LoCoMo categories above and the two "
    "categorical wins no competitor has (flat recall-vs-age, verified recall with a citable "
    "immutable source); BEAM-10M contradiction is presented as the frontier, not a solved box."
)


def _wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _mcnemar_pvalue(a_only: int, b_only: int) -> float:
    total = a_only + b_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(0, min(a_only, b_only) + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


_CONSOLIDATION_FIELDS = (
    "pending_processed",
    "facts_extracted",
    "events_indexed",
    "extraction_timed_out",
    "extraction_deferred",
    "extraction_windows_planned",
    "extraction_windows_submitted",
    "extraction_window_cap_per_record",
    "extraction_call_budget",
    "extraction_raw_only_bounded",
    "record_raw_only_bounded",
    "extraction_partial_bounded",
    "long_haystack_bounded",
    "long_haystack_raw_only",
)


def _consolidation_group_key(row: dict) -> str:
    sample_id = str(row.get("sample_id", "unknown"))
    return f"{row.get('dataset', 'unknown')}:{sample_id.split('_q')[0]}:{row.get('run_idx', 0)}"


def _consolidation_field(report: dict, field: str) -> int:
    src = report.get("consolidate_pending", report)
    if not isinstance(src, dict):
        return 0
    try:
        return int(src.get(field, 0) or 0)
    except (TypeError, ValueError):
        return 0


def consolidation_rollup(rows: Iterable[dict], *, system: str | None = None) -> dict[str, dict]:
    """Roll up logged consolidation reports once per conversation/run.

    The harness attaches the same consolidation report to every question in a grouped
    conversation. De-duping by dataset/sample-prefix/run keeps the health metrics honest.
    """
    by_group: dict = defaultdict(dict)
    for row in rows:
        sysname = row.get("system")
        if not sysname or (system is not None and sysname != system):
            continue
        extra = row.get("extra") or {}
        report = extra.get("consolidate")
        if not isinstance(report, dict):
            continue
        by_group[sysname][_consolidation_group_key(row)] = report

    out: dict[str, dict] = {}
    for sysname, reports_by_group in by_group.items():
        reports = list(reports_by_group.values())
        out[sysname] = {"groups": len(reports)}
        for field in _CONSOLIDATION_FIELDS:
            out[sysname][field] = sum(_consolidation_field(r, field) for r in reports)
    return out


def aggregate(rows: list[dict]) -> dict:
    # accuracy per (system, dataset, category) averaged within a run, then mean+/-std across runs
    by_run: dict = defaultdict(lambda: defaultdict(list))   # (sys,ds,cat) -> run_idx -> [correct]
    pooled: dict = defaultdict(lambda: [0, 0])               # (sys,ds,cat) -> [successes, n]
    write_by_group: dict = defaultdict(dict)                 # sys -> group(sample-prefix) -> tokens
    qtok: dict = defaultdict(list)
    search: dict = defaultdict(list)
    e2e: dict = defaultdict(list)
    paired: dict = defaultdict(dict)                         # (ds,cat,sample,run) -> sys -> correct
    # Integrity rollup: verified recall vs fabrication, read straight off the logged verify/abstain
    # flags. Honesty differentiators no accuracy column shows -- does an emitted answer carry an
    # entailment proof, or is it an unproven claim. Pooled across runs (no per-run std needed).
    integrity: dict = defaultdict(lambda: {"n": 0, "verified_correct": 0, "answered": 0,
                                           "unverified_answered": 0, "abstained": 0,
                                           "has_verify": False})
    systems, cats_by_ds = set(), defaultdict(set)
    for r in rows:
        if r.get("error"):       # transport/runtime error on this question -> excluded from accuracy
            continue
        sys, ds, cat = r["system"], r["dataset"], r["category"]
        systems.add(sys)
        ig = integrity[sys]
        ig["n"] += 1
        _abst = bool(r.get("abstained"))
        extra = r.get("extra") or {}
        _ver = bool(extra.get("verified"))
        if ("verified" in extra) or _abst:
            ig["has_verify"] = True              # this system has a verify/abstain step
        if _abst:
            ig["abstained"] += 1
        else:
            ig["answered"] += 1
            if not _ver:
                ig["unverified_answered"] += 1   # emitted without an entailment proof
        if r.get("correct") and _ver:
            ig["verified_correct"] += 1
        cats_by_ds[ds].add(cat)
        ok = 1.0 if r["correct"] else 0.0
        by_run[(sys, ds, cat)].setdefault(r["run_idx"], []).append(ok)
        pooled[(sys, ds, cat)][0] += int(bool(r["correct"]))
        pooled[(sys, ds, cat)][1] += 1
        paired[(ds, cat, r["sample_id"], r["run_idx"])][sys] = bool(r["correct"])
        qtok[sys].append(r.get("query_tokens", 0))
        search[sys].append(r.get("search_ms", 0.0))
        e2e[sys].append(r.get("e2e_ms", 0.0))
        # one write-cost per (system, group, run); group encoded in sample namespace prefix
        group_key = f"{ds}:{r['sample_id'].split('_q')[0]}:{r['run_idx']}"
        write_by_group[sys][group_key] = r.get("write_tokens", 0)

    acc: dict = {}
    acc_ci: dict = {}
    n_by_cat: dict = defaultdict(int)                        # (ds,cat) -> questions per run
    run_acc: dict = {}
    for key, runs in by_run.items():
        per_run = [mean(v) for v in runs.values() if v]
        acc[key] = (mean(per_run) if per_run else 0.0, pstdev(per_run) if len(per_run) > 1 else 0.0)
        run_acc[key] = {run_idx: mean(vals) for run_idx, vals in runs.items() if vals}
        successes, n = pooled[key]
        acc_ci[key] = (*_wilson_ci(successes, n), successes, n)
        first = next(iter(runs.values()), [])
        n_by_cat[(key[1], key[2])] = max(n_by_cat[(key[1], key[2])], len(first))

    def pct(xs, p):
        return float(np.percentile(xs, p)) if xs else 0.0

    cost = {s: {"tokens_per_write": (mean(list(write_by_group[s].values())) if write_by_group[s] else 0.0),
                "tokens_per_query": (mean(qtok[s]) if qtok[s] else 0.0)} for s in systems}
    latency = {s: {"search_p50": pct(search[s], 50), "search_p95": pct(search[s], 95),
                   "e2e_p50": pct(e2e[s], 50), "e2e_p95": pct(e2e[s], 95)} for s in systems}
    consolidation = consolidation_rollup(rows)
    head_to_head: dict = {}
    for ds_cat in sorted({(r["dataset"], r["category"]) for r in rows}):
        ds, cat = ds_cat
        present = sorted(s for s in systems if (s, ds, cat) in acc)
        for a, b in combinations(present, 2):
            a_only = b_only = both = neither = paired_n = 0
            for (pds, pcat, _sid, _run), vals in paired.items():
                if pds != ds or pcat != cat or a not in vals or b not in vals:
                    continue
                paired_n += 1
                av, bv = vals[a], vals[b]
                if av and bv:
                    both += 1
                elif av and not bv:
                    a_only += 1
                elif bv and not av:
                    b_only += 1
                else:
                    neither += 1
            if paired_n:
                head_to_head[(a, b, ds, cat)] = {
                    "n": paired_n, "a_only": a_only, "b_only": b_only,
                    "both": both, "neither": neither,
                    "p_mcnemar": _mcnemar_pvalue(a_only, b_only),
                }

    survival: dict = {}
    for (a, b, ds, cat), stats in head_to_head.items():
        a_runs = run_acc.get((a, ds, cat), {})
        b_runs = run_acc.get((b, ds, cat), {})
        common = sorted(set(a_runs) & set(b_runs))
        if len(common) < 2:
            status = "needs-2-runs"
        else:
            diffs = [a_runs[i] - b_runs[i] for i in common]
            status = "survives" if all(d > 0 for d in diffs) else "does-not-survive"
        survival[(a, b, ds, cat)] = {"runs": common, "status": status}
    return {"systems": sorted(systems), "acc": acc, "cats_by_ds": {k: sorted(v) for k, v in cats_by_ds.items()},
            "cost": cost, "latency": latency, "n_by_cat": dict(n_by_cat),
            "acc_ci": acc_ci, "head_to_head": head_to_head, "survival": survival,
            "integrity": {s: dict(v) for s, v in integrity.items()},
            "consolidation": consolidation}


def _acc_cell(acc, sys, ds, cat) -> str:
    if (sys, ds, cat) in acc:
        m, sd = acc[(sys, ds, cat)]
        return f"{m * 100:.1f}±{sd * 100:.1f}"
    return "-"


def _ci_cell(acc_ci, sys, ds, cat) -> str:
    if (sys, ds, cat) not in acc_ci:
        return "-"
    lo, hi, successes, n = acc_ci[(sys, ds, cat)]
    return f"{successes}/{n}, {lo * 100:.1f}-{hi * 100:.1f}"


def render(out_dir: Path, judge_desc: Optional[dict] = None) -> Path:
    out_dir = Path(out_dir)
    rows = load_logs(out_dir)
    fingerprint = log_fingerprint(out_dir)
    md = out_dir / "scoreboard.md"
    if not rows:
        md.write_text("# Eidetic-Plus benchmark scoreboard\n\n"
                      "**Pending run.** No result logs found in this directory. Run the harness "
                      "with a funded key (and the baselines) to populate this scoreboard:\n\n"
                      "```bash\nbash bench/reproduce.sh\n```\n\n"
                      "Numbers appear here ONLY from real runs -- never fabricated.\n")
        (out_dir / "scoreboard.json").write_text(json.dumps({
            "status": "pending",
            "reason": "no logs",
            "log_fingerprint": fingerprint,
        }, indent=2))
        return md

    agg = aggregate(rows)
    systems = agg["systems"]
    lines = ["# Eidetic-Plus benchmark scoreboard", ""]
    if judge_desc:
        lines.append(f"_Judge: **{judge_desc.get('judge_model')}** "
                     f"({judge_desc.get('judge_backend')}), one fixed judge + one fixed reader "
                     f"across all systems. Per-category accuracy = mean±std over runs; "
                     f"CI = Wilson 95% interval over logged questions._\n")

    dataset_orders = {"longmemeval": _LME_ORDER, "locomo": _LOCOMO_ORDER,
                      "memoryagentbench": _MAB_ORDER, "beam": []}
    ordered_datasets = [d for d in dataset_orders if d in agg["cats_by_ds"]]
    ordered_datasets += [d for d in sorted(agg["cats_by_ds"]) if d not in dataset_orders]

    for ds in ordered_datasets:
        order = dataset_orders.get(ds, [])
        cats = [c for c in order if c in agg["cats_by_ds"].get(ds, [])]
        cats += [c for c in agg["cats_by_ds"].get(ds, []) if c not in order]
        if not cats:
            continue
        lines.append(f"## {ds} - accuracy by category (%), mean±std; n = questions/run\n")
        lines.append("| category (n) | " + " | ".join(systems) + " |")
        lines.append("|" + "---|" * (len(systems) + 1))
        for cat in cats:
            n = agg.get("n_by_cat", {}).get((ds, cat), 0)
            lines.append(f"| {cat} (n={n}) | "
                         + " | ".join(_acc_cell(agg["acc"], s, ds, cat) for s in systems) + " |")
        lines.append("")
        lines.append(f"## {ds} - Wilson 95% CI by category\n")
        lines.append("| category | " + " | ".join(systems) + " |")
        lines.append("|" + "---|" * (len(systems) + 1))
        for cat in cats:
            lines.append(f"| {cat} | "
                         + " | ".join(_ci_cell(agg["acc_ci"], s, ds, cat) for s in systems) + " |")
        lines.append("")

    if agg["head_to_head"]:
        lines.append("## Head-to-head paired tests\n")
        lines.append("| pair | category | n | a-only | b-only | McNemar p | CI-clear win | slice survival |")
        lines.append("|---|---|---:|---:|---:|---:|---|---|")
        for key, stats in sorted(agg["head_to_head"].items()):
            a, b, ds, cat = key
            aci = agg["acc_ci"].get((a, ds, cat))
            bci = agg["acc_ci"].get((b, ds, cat))
            if aci and bci and aci[0] > bci[1]:
                clear = a
            elif aci and bci and bci[0] > aci[1]:
                clear = b
            else:
                clear = "no"
            surv = agg["survival"].get(key, {}).get("status", "unknown")
            lines.append(f"| {a} vs {b} | {ds}/{cat} | {stats['n']} | "
                         f"{stats['a_only']} | {stats['b_only']} | "
                         f"{stats['p_mcnemar']:.4f} | {clear} | {surv} |")
        lines.append("")

    lines.append("## Cost (approx tokens, uniform ~4 chars/token across all systems)\n")
    lines.append("| system | tokens / write (per conversation) | tokens / query |")
    lines.append("|---|---|---|")
    for s in systems:
        c = agg["cost"][s]
        lines.append(f"| {s} | {c['tokens_per_write']:.0f} | {c['tokens_per_query']:.0f} |")
    lines.append("")

    lines.append("## Latency (ms)\n")
    lines.append("| system | search p50 | search p95 | e2e p50 | e2e p95 |")
    lines.append("|---|---|---|---|---|")
    for s in systems:
        la = agg["latency"][s]
        lines.append(f"| {s} | {la['search_p50']:.1f} | {la['search_p95']:.1f} | "
                     f"{la['e2e_p50']:.1f} | {la['e2e_p95']:.1f} |")
    lines.append("")

    con = agg.get("consolidation", {})
    if con:
        lines.append("## Consolidation Health\n")
        lines.append("_Counts are logged once per ingested conversation/run. Timeouts mean the "
                     "record stayed searchable as raw memory, but did not finish fact/event "
                     "extraction within the configured sleep deadline._\n")
        lines.append("| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | windows planned | windows submitted | raw-only bounded | record raw-only | partial bounded | long-haystack bounded | long-haystack raw-only |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in sorted(con):
            c = con[s]
            lines.append(f"| {s} | {c.get('groups', 0)} | {c.get('pending_processed', 0)} | "
                         f"{c.get('facts_extracted', 0)} | {c.get('events_indexed', 0)} | "
                         f"{c.get('extraction_timed_out', 0)} | {c.get('extraction_deferred', 0)} | "
                         f"{c.get('extraction_windows_planned', 0)} | "
                         f"{c.get('extraction_windows_submitted', 0)} | "
                         f"{c.get('extraction_raw_only_bounded', 0)} | "
                         f"{c.get('record_raw_only_bounded', 0)} | "
                         f"{c.get('extraction_partial_bounded', 0)} | "
                         f"{c.get('long_haystack_bounded', 0)} | "
                         f"{c.get('long_haystack_raw_only', 0)} |")
        lines.append("")

    integ = agg.get("integrity", {})
    if integ:
        lines.append("## Integrity (verified recall) - from logged verify/abstain flags\n")
        lines.append("_verified accuracy = correct AND entailment-proven, over ALL questions (so "
                     "abstentions and unverifiable categories depress it -- it is a recall metric, "
                     "not comparable to judge accuracy). unproven-answer rate = answered WITHOUT an "
                     "entailment proof (NOT a fabrication count: an unproven answer can still be "
                     "correct). abstention rate = declined for lack of evidence. Systems without a "
                     "verify step (rag-full / rag-vector / mem0) show N/A -- they emit no proofs by "
                     "construction, which is not the same as fabricating._\n")
        lines.append("| system | n | verified accuracy (/n) | unproven-answer rate | abstention rate |")
        lines.append("|---|---:|---:|---:|---:|")
        for s in systems:
            ig = integ.get(s)
            if not ig or ig["n"] == 0:
                continue
            n = ig["n"]
            ar = ig["abstained"] / n
            if ig.get("has_verify"):
                va = f"{ig['verified_correct'] / n * 100:.1f}%"
                up = f"{ig['unverified_answered'] / n * 100:.1f}%"
            else:
                va = up = "N/A (no verify step)"
            lines.append(f"| {s} | {n} | {va} | {up} | {ar * 100:.1f}% |")
        lines.append("")

    lines.append(_BEAM_NOTE)
    md.write_text("\n".join(lines) + "\n")
    (out_dir / "scoreboard.json").write_text(json.dumps({
        "systems": systems,
        "accuracy": {
            f"{k[0]}|{k[1]}|{k[2]}": {
                "mean": v[0], "std": v[1],
                "ci95": list(agg["acc_ci"].get(k, (0.0, 0.0))[:2]),
                "successes": agg["acc_ci"].get(k, (0.0, 0.0, 0, 0))[2],
                "n": agg["acc_ci"].get(k, (0.0, 0.0, 0, 0))[3],
            } for k, v in agg["acc"].items()
        },
        "head_to_head": {
            f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in agg["head_to_head"].items()
        },
        "survival": {
            f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in agg["survival"].items()
        },
        "cost": agg["cost"], "latency": agg["latency"], "judge": judge_desc,
        "integrity": agg.get("integrity", {}),
        "consolidation": agg.get("consolidation", {}),
        "log_fingerprint": fingerprint,
    }, indent=2))
    return md
