"""python -m eidetic.autoresearch.leaderboard -- render the research ledger.

Reads trials.jsonl + promotions.jsonl + improve_ticks.jsonl and prints the ratchet's
history: every hypothesis, verdict, delta, and the epistemic-map movement per tick.
Read-only; a judge can run this from a fresh clone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def render(root: Path) -> str:
    trials = _load(root / "trials.jsonl")
    promotions = _load(root / "promotions.jsonl")
    ticks = _load(root / "improve_ticks.jsonl")
    lines = ["# Autoresearch leaderboard", ""]
    lines.append(f"trials: {len(trials)}  |  promotions: {len(promotions)}  |  "
                 f"improve ticks: {len(ticks)}")
    lines.append("")
    if trials:
        lines.append("| trial | tier | hypothesis | delta_pp | McNemar p | n | verdict |")
        lines.append("|---|---|---|---:|---:|---:|---|")
        for t in trials:
            hyp = t.get("hypothesis", {})
            if t.get("tier") == "A":
                desc = f"{hyp.get('knob')}={hyp.get('value')}"
            elif t.get("tier") == "B":
                read = (hyp.get("pipeline") or {}).get("read", [])
                desc = f"pipeline read={read}"
            else:
                desc = f"law {hyp.get('law_id', '')[:32]}"
            p = t.get("mcnemar_p")
            lines.append(
                f"| {t.get('trial_id', '')[:22]} | {t.get('tier')} | {desc[:44]} "
                f"| {t.get('delta_pp', 0.0):+.2f} | {p if p is not None else '-'} "
                f"| {t.get('paired_n', 0)} | {t.get('decision')} |")
        lines.append("")
    if ticks:
        lines.append("## Epistemic map per improve tick")
        lines.append("| ts | unknown | contested | known | Δu | Δc | Δk |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for tk in ticks[-20:]:
            b, a, d = tk.get("map_before", {}), tk.get("map_after", {}), tk.get("map_delta", {})
            lines.append(
                f"| {int(tk.get('ts', 0))} | {a.get('unknown_n', '-')} "
                f"| {a.get('contested_n', '-')} | {a.get('known_n', '-')} "
                f"| {d.get('unknown', 0):+d} | {d.get('contested', 0):+d} "
                f"| {d.get('known', 0):+d} |")
        lines.append("")
    if promotions:
        lines.append("## Promotions (append-only)")
        for p in promotions:
            lines.append(f"- `{p.get('trial_id')}` tier {p.get('tier')} dev_acc="
                         f"{p.get('dev_acc'):.3f} n={p.get('paired_n')} at {int(p.get('promoted_at', 0))}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="artifacts/autoresearch")
    args = ap.parse_args()
    print(render(Path(args.root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
