"""Benchmark Theater report (Track 5.4): one bundled report.md + report.json over the raw run
logs. On top of the scoreboard's accuracy/cost/latency/paired-significance, it adds the PRODUCT
metrics that are Eidetic's actual claims -- verified recall rate, abstention rate + accuracy on
answered, and the age-flatness slope (recall vs memory age; ~0 = the signature age-independent
recall). Renders ONLY from real logs; a number that does not reproduce is never written.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from . import scoreboard
from .harness import load_logs


def _slope_per_year(pairs: list[tuple[float, float]]) -> Optional[float]:
    """Linear slope of correctness (y in {0,1}) on memory age (x in days), scaled to per-year.
    ~0 means recall does not depend on age. None when there are <2 distinct ages to fit."""
    xs = [x for x, _ in pairs]
    if len(set(xs)) < 2:
        return None
    m = float(np.polyfit(np.asarray(xs, float), np.asarray([y for _, y in pairs], float), 1)[0])
    return m * 365.0


def product_metrics(rows: list[dict]) -> dict:
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    out: dict[str, dict] = {}
    for sysname, rs in by_sys.items():
        n = len(rs)
        answered = [r for r in rs if not r.get("abstained")]
        n_correct = sum(1 for r in rs if r.get("correct"))
        n_abstained = sum(1 for r in rs if r.get("abstained"))
        n_verified = sum(1 for r in rs if (r.get("extra") or {}).get("verified"))
        ans_correct = sum(1 for r in answered if r.get("correct"))
        ages = [(float(r["age_days"]), 1.0 if r.get("correct") else 0.0)
                for r in rs if r.get("age_days") is not None]
        out[sysname] = {
            "n": n,
            "accuracy": (n_correct / n) if n else 0.0,
            "abstention_rate": (n_abstained / n) if n else 0.0,
            "accuracy_on_answered": (ans_correct / len(answered)) if answered else 0.0,
            "verified_rate": (n_verified / n) if n else 0.0,
            "age_flatness_slope_per_year": _slope_per_year(ages),
        }
    return out


def _smqe_note_parts(extra: dict) -> tuple[str, str, str]:
    note = str(
        extra.get("smqe_policy")
        or extra.get("policy")
        or extra.get("note")
        or ""
    )
    parts = note.split(":")
    if len(parts) >= 3 and parts[0] == "smqe":
        return note, parts[1], parts[2]
    return note, str(extra.get("smqe_operator") or ""), str(extra.get("smqe_backend") or "")


def smqe_metrics(rows: list[dict]) -> dict:
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            by_sys[r["system"]].append(r)
    out: dict[str, dict] = {}
    forbidden_policy_bits = (
        "source-scan",
        "long" + "memeval-direct",
        "locomo-" + "direct",
        "direct-fact",
    )
    for sysname, rs in by_sys.items():
        operators: dict[str, int] = defaultdict(int)
        row = {
            "n": len(rs),
            "structured": 0,
            "claim": 0,
            "record": 0,
            "fallback": 0,
            "legacy_policy_rows": 0,
            "operators": operators,
        }
        for r in rs:
            extra = r.get("extra") or {}
            note, op, backend = _smqe_note_parts(extra)
            structured = bool(extra.get("structured_recall")) or note.startswith("smqe:")
            if structured:
                row["structured"] += 1
                if backend:
                    row[backend] = int(row.get(backend, 0)) + 1
                if op:
                    operators[op] += 1
            else:
                row["fallback"] += 1
            policy_text = " ".join(str(extra.get(k, "")) for k in ("policy", "note", "smqe_policy")).lower()
            if any(bit in policy_text for bit in forbidden_policy_bits):
                row["legacy_policy_rows"] += 1
        row["operators"] = dict(sorted(operators.items()))
        out[sysname] = row
    return out


def _region_list_values(extra: dict, field: str) -> tuple[list[str], bool]:
    value = extra.get(field, [])
    if not isinstance(value, list):
        return [], True
    return [str(item) for item in value if item not in (None, "")], False


def region_metrics(rows: list[dict]) -> dict:
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            by_sys[r["system"]].append(r)

    out: dict[str, dict] = {}
    for sysname, rs in by_sys.items():
        n = len(rs)
        telemetry_rows = 0
        hint_rows = 0
        total_hints = 0
        malformed_samples: list[str] = []
        region_ids: set[str] = set()
        member_ids: set[str] = set()
        for r in rs:
            extra = r.get("extra") or {}
            if not isinstance(extra, dict) or "region_hint_count" not in extra:
                continue
            telemetry_rows += 1
            row_malformed = False
            count = 0
            try:
                count = int(extra.get("region_hint_count", 0) or 0)
            except (TypeError, ValueError):
                row_malformed = True
            if count < 0:
                row_malformed = True

            ids, bad_ids = _region_list_values(extra, "region_ids")
            members, bad_members = _region_list_values(extra, "region_member_ids")
            row_malformed = row_malformed or bad_ids or bad_members
            if row_malformed:
                malformed_samples.append(str(r.get("sample_id", "<sample>")))
                continue

            if count > 0:
                hint_rows += 1
                total_hints += count
            region_ids.update(ids)
            member_ids.update(members)

        out[sysname] = {
            "rows": n,
            "telemetry_rows": telemetry_rows,
            "telemetry_rate": round((telemetry_rows / n) if n else 0.0, 4),
            "missing_rows": n - telemetry_rows,
            "hint_rows": hint_rows,
            "hint_row_rate": round((hint_rows / n) if n else 0.0, 4),
            "total_hints": total_hints,
            "unique_region_ids": len(region_ids),
            "unique_region_member_ids": len(member_ids),
            "malformed_rows": len(malformed_samples),
            "malformed_samples": malformed_samples[:20],
        }
    return out


def _pending(out_dir: Path) -> Path:
    md = out_dir / "report.md"
    md.write_text("# Eidetic-Plus benchmark report\n\n"
                  "**Pending run.** No result logs in this directory. Run the harness with a "
                  "funded key (and baselines) to populate this report:\n\n"
                  "```bash\nbash bench/reproduce.sh\n```\n\n"
                  "Numbers appear here ONLY from real runs -- never fabricated.\n")
    return md


def build_report(out_dir, judge_desc: Optional[dict] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_logs(out_dir)
    if not rows:
        return _pending(out_dir)

    agg = scoreboard.aggregate(rows)
    pm = product_metrics(rows)
    smqe = smqe_metrics(rows)
    regions = region_metrics(rows)
    systems = agg["systems"]
    has_manifest = (out_dir / "run_manifest.json").exists()

    lines = ["# Eidetic-Plus benchmark report", ""]
    if judge_desc:
        lines.append(f"_Judge: **{judge_desc.get('judge_model')}** "
                     f"({judge_desc.get('judge_backend')}); one fixed judge + one fixed reader "
                     f"across all systems._\n")

    lines.append("## Product metrics (the claims)\n")
    lines.append("| system | n | accuracy | acc on answered | abstention rate | verified rate "
                 "| age-flatness slope/yr |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in systems:
        m = pm.get(s, {})
        slope = m.get("age_flatness_slope_per_year")
        slope_s = "n/a" if slope is None else f"{slope:+.4f}"
        lines.append(f"| {s} | {m.get('n', 0)} | {m.get('accuracy', 0.0) * 100:.1f} "
                     f"| {m.get('accuracy_on_answered', 0.0) * 100:.1f} "
                     f"| {m.get('abstention_rate', 0.0) * 100:.1f} "
                     f"| {m.get('verified_rate', 0.0) * 100:.1f} | {slope_s} |")
    lines.append("")
    lines.append("> **verified rate** = fraction of answers grounded by an NLI-entailed immutable "
                 "source (only the eidetic-plus-full product row carries this). **abstention rate** "
                 "= fraction the system honestly declined. **age-flatness slope/yr** ~ 0 is the "
                 "signature age-independent recall. **false-premise correctness** requires a "
                 "labeled false-premise set (e.g. HaluMem-style) and is reported only when such a "
                 "split is loaded -- not invented here.\n")

    lines.append("## Accuracy by category (%), mean±std\n")
    for ds in sorted(agg["cats_by_ds"]):
        cats = agg["cats_by_ds"][ds]
        if not cats:
            continue
        lines.append(f"### {ds}\n")
        lines.append("| category | " + " | ".join(systems) + " |")
        lines.append("|" + "---|" * (len(systems) + 1))
        for cat in cats:
            lines.append(f"| {cat} | "
                         + " | ".join(scoreboard._acc_cell(agg["acc"], s, ds, cat) for s in systems)
                         + " |")
        lines.append("")

    if agg["head_to_head"]:
        lines.append("## Paired significance (McNemar) + slice survival\n")
        lines.append("| pair | category | n | a-only | b-only | McNemar p | survival |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for key, st in sorted(agg["head_to_head"].items()):
            a, b, ds, cat = key
            surv = agg["survival"].get(key, {}).get("status", "unknown")
            lines.append(f"| {a} vs {b} | {ds}/{cat} | {st['n']} | {st['a_only']} | "
                         f"{st['b_only']} | {st['p_mcnemar']:.4f} | {surv} |")
        lines.append("")

    lines.append("## Cost + latency\n")
    lines.append("| system | tokens/write | tokens/query | search p50 | search p95 | e2e p50 | e2e p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in systems:
        c, la = agg["cost"][s], agg["latency"][s]
        lines.append(f"| {s} | {c['tokens_per_write']:.0f} | {c['tokens_per_query']:.0f} "
                     f"| {la['search_p50']:.1f} | {la['search_p95']:.1f} "
                     f"| {la['e2e_p50']:.1f} | {la['e2e_p95']:.1f} |")
    lines.append("")

    con = agg.get("consolidation", {})
    if con:
        lines.append("## Consolidation health\n")
        lines.append("| system | groups | pending processed | facts | events | extraction timed out | extraction deferred | record raw-only |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for s, c in sorted(con.items()):
            lines.append(f"| {s} | {c.get('groups', 0)} | {c.get('pending_processed', 0)} | "
                         f"{c.get('facts_extracted', 0)} | {c.get('events_indexed', 0)} | "
                         f"{c.get('extraction_timed_out', 0)} | {c.get('extraction_deferred', 0)} | "
                         f"{c.get('record_raw_only_bounded', 0)} |")
        lines.append("")

    if smqe:
        lines.append("## SMQE Backend Mix\n")
        lines.append("| system | rows | structured | claim | record | fallback | top operators | legacy policy rows |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---:|")
        for s in systems:
            m = smqe.get(s, {})
            ops = m.get("operators", {}) or {}
            top_ops = ", ".join(f"{k}:{v}" for k, v in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))[:4])
            lines.append(f"| {s} | {m.get('n', 0)} | {m.get('structured', 0)} | "
                         f"{m.get('claim', 0)} | {m.get('record', 0)} | "
                         f"{m.get('fallback', 0)} | {top_ops or '-'} | "
                         f"{m.get('legacy_policy_rows', 0)} |")
        lines.append("")

    if regions:
        lines.append("## Region Routing Telemetry\n")
        lines.append("| system | rows | telemetry rows | hint rows | hint rows % | total hints | unique regions | unique members | missing | malformed |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in systems:
            m = regions.get(s, {})
            lines.append(f"| {s} | {m.get('rows', 0)} | {m.get('telemetry_rows', 0)} | "
                         f"{m.get('hint_rows', 0)} | {m.get('hint_row_rate', 0.0) * 100:.1f} | "
                         f"{m.get('total_hints', 0)} | {m.get('unique_region_ids', 0)} | "
                         f"{m.get('unique_region_member_ids', 0)} | "
                         f"{m.get('missing_rows', 0)} | {m.get('malformed_rows', 0)} |")
        lines.append("")
        lines.append("> Region telemetry is read from answer-path logs only. It is an observability "
                     "signal for whether cocoon/region routing contributed context, not an "
                     "independent accuracy claim.\n")

    lines.append("## Reproduction\n")
    lines.append(f"- Raw logs: `{out_dir}/<system>__run<N>.jsonl` (one JSON line per question).")
    lines.append(f"- Manifest: `{out_dir}/run_manifest.json`"
                 + ("" if has_manifest else " (missing -- run via bench.run to record it)") + ".")
    lines.append("- Scoreboard + curves: `bash bench/reproduce.sh` (full >=10-run test split).\n")

    md = out_dir / "report.md"
    md.write_text("\n".join(lines) + "\n")
    (out_dir / "report.json").write_text(json.dumps({
        "systems": systems,
        "product_metrics": pm,
        "smqe": smqe,
        "region_telemetry": regions,
        "accuracy": {f"{k[0]}|{k[1]}|{k[2]}": {"mean": v[0], "std": v[1]} for k, v in agg["acc"].items()},
        "cost": agg["cost"], "latency": agg["latency"],
        "consolidation": agg.get("consolidation", {}),
        "head_to_head": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in agg["head_to_head"].items()},
        "judge": judge_desc, "has_manifest": has_manifest,
    }, indent=2))
    return md
