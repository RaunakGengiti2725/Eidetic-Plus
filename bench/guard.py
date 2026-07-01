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

from .compare import _load_logs_strict, compare_dirs
from .datasets import split_of
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
    if p.suffix == ".env":
        lines = [
            "# Champion config promoted by bench.guard from DEV-split logs.",
            f"# dev_acc={dev_acc:.6f}",
            f"# paired_n={n}",
        ]
        for k in sorted(env):
            lines.append(f"{k}={env[k]}")
        p.write_text("\n".join(lines) + "\n")
        return
    p.write_text(json.dumps({"env": env, "dev_acc": dev_acc, "n": n}, indent=2))


def _reject(reason: str, **extra) -> dict:
    return {
        "accept": False,
        "reason": reason,
        "champion_acc": 0.0,
        "challenger_acc": 0.0,
        "mcnemar_p": None,
        "paired_n": 0,
        "unpaired": 0,
        **extra,
    }


def _load_manifest(path: Path) -> tuple[dict, str]:
    manifest_path = Path(path) / "run_manifest.json"
    if not manifest_path.exists():
        return {}, "missing run_manifest.json"
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as e:
        return {}, f"{type(e).__name__}: {e}"
    if not isinstance(data, dict):
        return {}, "run_manifest.json is not a JSON object"
    return data, ""


def _dev_artifact_checks(path: Path, label: str, *, required_split: str = "dev") -> list[dict]:
    path = Path(path)
    manifest, error = _load_manifest(path)
    checks: list[dict] = []
    checks.append({
        "name": f"{label}:manifest_valid",
        "pass": not error,
        "detail": "valid" if not error else error,
    })
    checks.append({
        "name": f"{label}:split",
        "pass": manifest.get("split") == required_split,
        "detail": f"{manifest.get('split', '<missing>')} (expected {required_split})",
    })
    checks.append({
        "name": f"{label}:not_render_only",
        "pass": not bool(manifest.get("render_only", True)),
        "detail": f"render_only={manifest.get('render_only', '<missing>')}",
    })
    if error:
        return checks

    rows = _load_logs_strict(path).rows
    bad = [
        str(row.get("sample_id", ""))
        for row in rows
        if split_of(str(row.get("sample_id", ""))) != required_split
    ]
    checks.append({
        "name": f"{label}:rows_dev_split",
        "pass": not bad,
        "detail": (
            "all dev rows" if not bad
            else f"{len(bad)} rows outside {required_split}: {', '.join(bad[:5])}"
        ),
    })
    return checks


def run_guard(champion_dir: Path, challenger_dir: Path, *, system: str = "eidetic-plus",
              min_delta_pp: float = 1.0, alpha: float = 0.05) -> dict:
    """Compare champion vs challenger DEV-split log dirs and decide. Refuses if the two runs did
    not score the same dev items (unpaired > 0) -- McNemar would be invalid otherwise."""
    artifact_checks = (
        _dev_artifact_checks(champion_dir, "champion")
        + _dev_artifact_checks(challenger_dir, "challenger")
    )
    failed_artifact_checks = [c for c in artifact_checks if not c["pass"]]
    if failed_artifact_checks:
        return _reject(
            "invalid dev artifacts: " + "; ".join(
                f"{c['name']}={c['detail']}" for c in failed_artifact_checks
            ),
            artifact_checks=artifact_checks,
            failed_artifact_checks=failed_artifact_checks,
        )

    cmp = compare_dirs(champion_dir, challenger_dir, system=system)
    inp = pooled_guard_inputs(cmp)
    if inp["unpaired"] > 0:
        return {"accept": False, "reason": f"unpaired dev items ({inp['unpaired']}): champion and "
                "challenger must score the SAME dev set for a valid paired test",
                "artifact_checks": artifact_checks, "failed_artifact_checks": [], **inp}
    decision = guard_decision(inp["champion_acc"], inp["challenger_acc"], inp["mcnemar_p"],
                              inp["paired_n"], min_delta_pp=min_delta_pp, alpha=alpha)
    return {**decision, "artifact_checks": artifact_checks, "failed_artifact_checks": [], **inp}


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
    print(f"Guard: {verdict} -- {res['reason']}")
    print(f"  champion dev acc={res['champion_acc']:.4f}  challenger dev acc={res['challenger_acc']:.4f}"
          f"  paired_n={res['paired_n']}  McNemar p={res['mcnemar_p']}")
    if res["accept"] and args.best_config:
        data = json.loads(Path(args.best_config).read_text())
        best_env = dict(data.get("best_env", {}))
        tau_path = Path(args.challenger) / "abstention_v2_tau.json"
        if tau_path.exists():
            try:
                tau = json.loads(tau_path.read_text()).get("tau")
                if tau is not None:
                    best_env["ABSTENTION_V2_TAU"] = str(tau)
            except (ValueError, OSError):
                pass
        save_champion(Path(args.champion_out), env=best_env,
                      dev_acc=res["challenger_acc"], n=res["paired_n"])
        print(f"  promoted -> {args.champion_out}. Apply by writing best_env to .env + restart.")
    return 0 if res["accept"] else 3      # nonzero on KEEP so a cron/operator notices a no-promote


if __name__ == "__main__":
    raise SystemExit(main())
