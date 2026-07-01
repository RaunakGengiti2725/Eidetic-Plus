"""The two signature curves on ALL THREE systems, from real logs: recall-vs-age and
(e2e) latency-vs-age. The thesis to show visually: Eidetic-Plus stays flat where Mem0
degrades with evidence age and Graphiti's latency grows with store size. Renders only
from real logs; writes a note if age data is unavailable.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .harness import load_logs


def render(out_dir: Path, nbins: int = 8) -> dict:
    out_dir = Path(out_dir)
    rows = [r for r in load_logs(out_dir) if r.get("age_days") is not None]
    if not rows:
        note = out_dir / "curves_NOTE.txt"
        note.write_text("No age-resolved logs yet (run the harness on a dataset with session "
                        "timestamps, e.g. LoCoMo/LongMemEval). Curves render from real runs only.\n")
        return {"ok": False, "note": str(note)}

    by_sys: dict = defaultdict(lambda: {"age": [], "correct": [], "e2e": []})
    for r in rows:
        d = by_sys[r["system"]]
        d["age"].append(r["age_days"] / 365.25)        # years
        d["correct"].append(1.0 if r["correct"] else 0.0)
        d["e2e"].append(r.get("e2e_ms", 0.0))

    all_age = np.concatenate([np.array(d["age"]) for d in by_sys.values()])
    if len(all_age) < 2 or float(all_age.min()) == float(all_age.max()):
        edges = np.linspace(float(all_age.min()), float(all_age.min()) + 1e-9, nbins + 1)
        note = "Need at least two distinct evidence ages to fit age-slope curves."
    else:
        edges = np.linspace(all_age.min(), all_age.max() + 1e-9, nbins + 1)
        note = ""
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(nbins)]

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    summary = {}
    plotted = False
    for sysname, d in sorted(by_sys.items()):
        age = np.array(d["age"]); cor = np.array(d["correct"]); e2e = np.array(d["e2e"])
        rc, lc, cx = [], [], []
        for i in range(nbins):
            m = (age >= edges[i]) & (age < edges[i + 1])
            if m.sum() == 0:
                continue
            cx.append(centers[i]); rc.append(float(cor[m].mean())); lc.append(float(np.percentile(e2e[m], 95)))
        if len(cx) > 1:
            ax1.plot(cx, rc, "o-", label=sysname)
            ax2.plot(cx, lc, "s-", label=sysname)
            plotted = True
            summary[sysname] = {"recall_slope_per_year": float(np.polyfit(cx, rc, 1)[0]),
                                "latency_slope_ms_per_year": float(np.polyfit(cx, lc, 1)[0])}

    ax1.set_title("Recall vs evidence age (flat = age-independent)")
    ax1.set_xlabel("evidence age (years)"); ax1.set_ylabel("accuracy"); ax1.set_ylim(0, 1.05)
    if note:
        ax1.text(0.5, 0.5, note, transform=ax1.transAxes, ha="center", va="center")
    ax1.grid(alpha=0.3)
    if plotted:
        ax1.legend()
    ax2.set_title("p95 end-to-end latency vs evidence age")
    ax2.set_xlabel("evidence age (years)"); ax2.set_ylabel("p95 latency (ms)")
    if note:
        ax2.text(0.5, 0.5, note, transform=ax2.transAxes, ha="center", va="center")
    ax2.grid(alpha=0.3)
    if plotted:
        ax2.legend()

    p1 = out_dir / "recall_vs_age.png"; p2 = out_dir / "latency_vs_age.png"
    fig1.tight_layout(); fig1.savefig(p1, dpi=130); plt.close(fig1)
    fig2.tight_layout(); fig2.savefig(p2, dpi=130); plt.close(fig2)
    res = {"ok": True, "recall_png": str(p1), "latency_png": str(p2), "slopes": summary}
    if note:
        res["note"] = note
    return res
