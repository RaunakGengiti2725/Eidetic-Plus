"""The EvolveMem auto-revert Guard: never let a tuned config degrade the dev score.

The PDF (3e) calls this "the genuinely valuable, transferable idea." It is the gate between the
sweep's `best_config.json` (the challenger) and applying it: the challenger is accepted ONLY if
it beats the current champion on the DEV split by a minimum margin AND that gain is significant
(paired McNemar on the SAME dev items). Otherwise the champion's config is kept.

Honest scope (the whole point of the idea is honesty):
  * Prevents a per-swap DEV-PROXY regression. Dev is a proxy; this does NOT guarantee a test-set
    win, and repeated guarded swaps overfit dev (multiple comparisons). Report on held-out TEST.
  * "Revert" = do-not-promote-the-challenger (keep the champion artifact). There is no hot-swap
    (Settings are frozen at construction; applying a config = write .env + restart).
  * Reads ONLY the dev split (the champion/challenger logs must be dev-split runs). It never
    reads, scores, or compares a test item.

`guard_decision` and `pooled_guard_inputs` are pure, offline-unit-testable logic; producing the
dev logs they consume is LLM-gated (a real benchmark run).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .compare import compare_dirs
from .scoreboard import _mcnemar_pvalue


def guard_decision(champion_acc: float, challenger_acc: float, mcnemar_p: Optional[float],
                   paired_n: int, *, min_delta_pp: float = 1.0, alpha: float = 0.05,
                   require_significance: bool = True) -> dict:
    """Accept the challenger only if it beats the champion by >= min_delta_pp accuracy points
    AND (optionally) the paired McNemar p < alpha, over a non-empty paired dev set."""
    delta_pp = (challenger_acc - champion_acc) * 100.0
    base = {"delta_pp": delta_pp, "mcnemar_p": mcnemar_p, "paired_n": paired_n}
    if paired_n <= 0:
        return {**base, "accept": False, "reason": "no paired dev items (cannot compare)"}
    if delta_pp < min_delta_pp:
        return {**base, "accept": False,
                "reason": f"delta {delta_pp:.2f}pp < required {min_delta_pp:.2f}pp"}
    if require_significance and (mcnemar_p is None or mcnemar_p >= alpha):
        return {**base, "accept": False,
                "reason": f"not significant (McNemar p={mcnemar_p}; need < {alpha})"}
    return {**base, "accept": True,
            "reason": f"accepted: +{delta_pp:.2f}pp, McNemar p={mcnemar_p}"}


def pooled_guard_inputs(compare_result: dict) -> dict:
    """Pool a compare_dirs() result into one OVERALL guard input: summed accuracy each side,
    pooled McNemar discordant counts (b = champion-right/challenger-wrong, c = the reverse),
    paired_n, and total unpaired rows. unpaired > 0 means the two runs did NOT score the same
    dev items, so the McNemar test is invalid and the guard must refuse."""
    champ_correct = champ_n = chal_correct = chal_n = 0
    b = c = paired_n = unpaired = 0
    incomparable = []
    for label, item in compare_result.get("comparisons", {}).items():
        if item.get("status") not in ("compared", "unpaired"):
            incomparable.append(label)
            continue
        champ_correct += item["control"]["correct"]
        champ_n += item["control"]["n"]
        chal_correct += item["experiment"]["correct"]
        chal_n += item["experiment"]["n"]
        p = item["paired"]
        b += p["control_only"]
        c += p["experiment_only"]
        paired_n += p["paired_n"]
        unpaired += p["unpaired_control_rows"] + p["unpaired_experiment_rows"]
    return {
        "champion_acc": champ_correct / champ_n if champ_n else 0.0,
        "challenger_acc": chal_correct / chal_n if chal_n else 0.0,
        "mcnemar_p": _mcnemar_pvalue(b, c) if paired_n else None,
        "paired_n": paired_n, "unpaired": unpaired,
        "discordant_b": b, "discordant_c": c, "incomparable": incomparable,
    }


# ---- champion registry (the persisted dev-score floor) ----------------------
def load_champion(path: Path) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def save_champion(path: Path, *, env: dict, dev_acc: float, n: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"env": env, "dev_acc": dev_acc, "n": n}, indent=2))


def run_guard(champion_dir: Path, challenger_dir: Path, *, system: str = "eidetic-plus",
              min_delta_pp: float = 1.0, alpha: float = 0.05) -> dict:
    """Compare champion vs challenger DEV-split log dirs and decide. Refuses if the two runs did
    not score the same dev items (unpaired > 0) -- McNemar would be invalid otherwise."""
    cmp = compare_dirs(champion_dir, challenger_dir, system=system)
    inp = pooled_guard_inputs(cmp)
    if inp["unpaired"] > 0:
        return {"accept": False, "reason": f"unpaired dev items ({inp['unpaired']}): champion and "
                "challenger must score the SAME dev set for a valid paired test", **inp}
    decision = guard_decision(inp["champion_acc"], inp["challenger_acc"], inp["mcnemar_p"],
                              inp["paired_n"], min_delta_pp=min_delta_pp, alpha=alpha)
    return {**decision, **inp}


def main() -> int:
    ap = argparse.ArgumentParser(description="EvolveMem auto-revert guard (dev-split only)")
    ap.add_argument("--champion", required=True, help="champion DEV-split log dir")
    ap.add_argument("--challenger", required=True, help="challenger DEV-split log dir")
    ap.add_argument("--system", default="eidetic-plus")
    ap.add_argument("--min-delta-pp", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--best-config", help="challenger best_config.json (promoted only on accept)")
    ap.add_argument("--champion-out", default="artifacts/guard/champion.json")
    args = ap.parse_args()

    res = run_guard(Path(args.champion), Path(args.challenger), system=args.system,
                    min_delta_pp=args.min_delta_pp, alpha=args.alpha)
    verdict = "ACCEPT challenger" if res["accept"] else "KEEP champion"
    print(f"Guard: {verdict} — {res['reason']}")
    print(f"  champion dev acc={res['champion_acc']:.4f}  challenger dev acc={res['challenger_acc']:.4f}"
          f"  paired_n={res['paired_n']}  McNemar p={res['mcnemar_p']}")
    if res["accept"] and args.best_config:
        data = json.loads(Path(args.best_config).read_text())
        save_champion(Path(args.champion_out), env=data.get("best_env", {}),
                      dev_acc=res["challenger_acc"], n=res["paired_n"])
        print(f"  promoted -> {args.champion_out}. Apply by writing best_env to .env + restart.")
    return 0 if res["accept"] else 3      # nonzero on KEEP so a cron/operator notices a no-promote


if __name__ == "__main__":
    raise SystemExit(main())
