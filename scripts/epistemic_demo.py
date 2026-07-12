"""The falsifiable overnight demo: a personal-memory corpus with PLANTED gaps,
conflicts, temporal holes, and law-shaped regularities.

  day0: build the corpus in its own DATA_DIR, sleep (token-free consolidation +
        map rebuild), snapshot the map, ask the Day-0 question set (real asks),
        write day0_report.json.
  night: run overnight waves (curiosity + contested programs + law verification),
        each wave logged; optionally guarded trials via the DevLab (separate flag).
  day1: re-snapshot, re-ask the SAME questions, write day1_report.json +
        map_delta.json -- the before/after a skeptic can diff.

Every answer ships through the REAL prove path. The planted design means the
expected motion is KNOWN in advance and checkable:
  - contested cells (Dana's two phone numbers) should resolve or hold honestly
  - unknown query cells (asked-then-unanswerable questions) stay unknown unless
    evidence exists somewhere in the corpus
  - law predictions (every project's repo has a CI pipeline) should verify
  - abstained day0 questions whose evidence IS in the corpus are the read-recovery
    targets (claim_select trial fodder)

Usage:
  python scripts/epistemic_demo.py day0   [--root artifacts/epistemic_demo]
  python scripts/epistemic_demo.py night  [--waves 3] [--probes 8]
  python scripts/epistemic_demo.py day1
Needs DASHSCOPE_API_KEY for asks/probes; the corpus build is embed-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

NS = "demo"

# ---- the corpus (planted structure, human-shaped notes) --------------------------
DAY = 86400.0
T0 = time.time() - 120 * DAY


def _t(days: float) -> float:
    return T0 + days * DAY

CORPUS: list[tuple[str, float]] = [
    # -- stable biography ------------------------------------------------------
    ("Maya lives in Lisbon and works as a freelance data engineer.", _t(0)),
    ("Maya's sister Dana lives in Porto with her dog Biscuit.", _t(1)),
    ("Maya keeps a small sailboat named Gaivota at the Doca de Belem marina.", _t(2)),
    # -- supersession chain with a CURRENT value (KNOWN current employer) -------
    ("Maya signed a six-month contract with Fintor, a payments startup.", _t(5)),
    ("Maya's Fintor contract ended; she started a new engagement with Oceanic Labs.", _t(60)),
    # -- superseded WITHOUT a current value (g1 target: gym membership) ---------
    ("Maya joined the Alvalade climbing gym on a monthly plan.", _t(10)),
    ("Maya cancelled her Alvalade climbing gym membership.", _t(45)),
    # -- CONTESTED: two active phone numbers for Dana (c1 target) ---------------
    ("Dana's phone number is +351 912 111 222.", _t(20)),
    ("Dana's phone number is +351 933 444 555.", _t(20)),
    # -- temporal hole (g4 target): apartment history with a gap ----------------
    ("Maya moved out of her Alfama apartment at the end of March.", _t(15)),
    ("Maya signed the lease on her new Graca apartment in June.", _t(75)),
    # -- events with and without dates (g3 target) ------------------------------
    ("Maya presented her data pipeline talk at the Lisbon Data Meetup on the first Tuesday of last month.", _t(80)),
    ("Maya once visited a tiny jazz bar in Barcelona she can never remember the name of.", _t(30)),
    # -- law-shaped regularity (laws target): project => repo => CI -------------
    ("Maya's project Albatross has its repo on GitHub under maya/albatross.", _t(35)),
    ("The Albatross repo runs CI on GitHub Actions.", _t(36)),
    ("Maya's project Barnacle has its repo on GitHub under maya/barnacle.", _t(40)),
    ("The Barnacle repo runs CI on GitHub Actions.", _t(41)),
    ("Maya's project Cormorant has its repo on GitHub under maya/cormorant.", _t(85)),
    # (Cormorant CI never stated -> the law PREDICTS it; unwitnessed)
    # -- read-recovery targets: answers present but phrased obliquely -----------
    ("Over coffee Maya said what makes her trust a dataset is knowing exactly who "
     "collected it and why they collected it.", _t(50)),
    ("Maya mentioned her sailing instructor, an older gentleman called Rui, taught "
     "her the trick of reading gusts from the water's texture.", _t(55)),
    ("Maya's favourite recovery meal after climbing was always the lentil soup from "
     "the corner tasca near the gym.", _t(12)),
    # -- distractors -------------------------------------------------------------
    ("Maya is rereading The Left Hand of Darkness for book club.", _t(90)),
    ("Dana adopted Biscuit from a shelter in Braga three years ago.", _t(3)),
    ("Maya's preferred espresso place near the marina closes on Mondays.", _t(8)),
]

# ---- day0/day1 question set --------------------------------------------------------
QUESTIONS: list[dict] = [
    # answerable, stable
    {"q": "Where does Maya live?", "expect": "verified"},
    {"q": "What is the name of Maya's sailboat?", "expect": "verified"},
    {"q": "Who is Maya currently working with?", "expect": "verified"},
    # read-recovery targets (evidence present, oblique phrasing)
    {"q": "What makes Maya trust a dataset?", "expect": "read_recovery"},
    {"q": "What trick did Maya's sailing instructor teach her?", "expect": "read_recovery"},
    {"q": "What was Maya's favourite meal after climbing?", "expect": "read_recovery"},
    # contested
    {"q": "What is Dana's phone number?", "expect": "contested_abstain"},
    # unknown (gap is real: never stated)
    {"q": "Does Maya still have a gym membership?", "expect": "unknown_current"},
    {"q": "What is the name of the jazz bar Maya visited in Barcelona?",
     "expect": "honest_abstain"},
    {"q": "Which CI system does the Cormorant repo use?", "expect": "law_prediction"},
    # temporal hole
    {"q": "Where was Maya living in May?", "expect": "temporal_hole"},
    # distractor control
    {"q": "What book is Maya rereading?", "expect": "verified"},
]


def _engine(root: Path):
    os.environ["DATA_DIR"] = str(root / "data")
    os.environ.setdefault("EPISTEMIC_MAP", "1")
    os.environ.setdefault("AUTORESEARCH", "1")
    os.environ["AUTORESEARCH_DIR"] = str(root / "autoresearch")
    from eidetic.config import get_settings
    get_settings.cache_clear()
    from eidetic.engine import Engine
    return Engine()


def _scope():
    from eidetic.models import Scope
    return Scope(namespace=NS)


def _ask_all(engine, label: str, root: Path) -> dict:
    scope = _scope()
    rows = []
    for item in QUESTIONS:
        t0 = time.perf_counter()
        try:
            ans = engine.ask(item["q"], scope=scope, verify=True, use_cache=False)
            rows.append({
                "q": item["q"], "expect": item["expect"],
                "status": ans.status.value,
                "answer": (ans.answer or "")[:200],
                "note": (ans.note or "")[:160],
                "citations": len(ans.citations),
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            })
        except Exception as e:
            rows.append({"q": item["q"], "expect": item["expect"],
                         "error": f"{type(e).__name__}: {str(e)[:160]}"})
        print(f"  [{rows[-1].get('status', 'ERR'):>9}] {item['q'][:60]}")
    verified = sum(1 for r in rows if r.get("status") == "VERIFIED")
    report = {"label": label, "ts": time.time(), "verified": verified,
              "total": len(rows), "rows": rows}
    out = root / f"{label}_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"{label}: {verified}/{len(rows)} verified -> {out}")
    return report


def day0(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    engine = _engine(root)
    scope = _scope()
    print("ingesting corpus...")
    for text, valid_at in CORPUS:
        engine.ingest_text(text, source="demo-notes", valid_at=valid_at, scope=scope,
                           consolidate_now=False)
    print("sleep (consolidate + dream + map rebuild)...")
    report = engine.lifecycle.sleep(scope=scope)
    print(json.dumps({k: v for k, v in report.get("epistemic_map", {}).items()},
                     indent=1))
    # law candidates minted token-free at day0 (verification happens overnight)
    from eidetic.epistemic.laws import LawBook
    book = LawBook(engine.knowledge_map_store)
    print("laws:", book.mine_candidates(engine.store, scope))
    engine.knowledge_map_store.rebuild(engine.store, scope)   # law predictions -> cells
    snap = engine.knowledge_map_store.snapshot(scope, root / "map_day0.json",
                                               label="day0")
    print(f"map day0: known={snap['known_n']} unknown={snap['unknown_n']} "
          f"contested={snap['contested_n']}")
    _ask_all(engine, "day0", root)
    engine.close()


def night(root: Path, *, waves: int, probes: int, trials: int) -> None:
    engine = _engine(root)
    scope = _scope()
    log = root / "night_log.jsonl"
    lab = None
    if trials > 0:
        from eidetic.autoresearch.lab import DevLab
        lab = DevLab(root / "lab")
        print("lab ingest (one-time, dev split)...")
        lab.ingest_once()
    for wave in range(waves):
        print(f"--- wave {wave + 1}/{waves} ---")
        entry: dict = {"ts": time.time(), "wave": wave + 1}
        entry["improve"] = engine.improve(scope=scope, max_trials=trials,
                                          max_probes=probes, lab=lab)
        from eidetic.epistemic.contested import contested_wave
        entry["contested"] = contested_wave(engine, scope, max_programs=3)
        from eidetic.epistemic.laws import law_verification_wave
        entry["laws"] = law_verification_wave(engine, scope, max_probes=4)
        with open(log, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        counts = engine.knowledge_map_store.counts(scope)
        print(f"map after wave: {counts}")
    engine.close()


def day1(root: Path) -> None:
    engine = _engine(root)
    scope = _scope()
    engine.knowledge_map_store.rebuild(engine.store, scope)
    snap = engine.knowledge_map_store.snapshot(scope, root / "map_day1.json",
                                               label="day1")
    print(f"map day1: known={snap['known_n']} unknown={snap['unknown_n']} "
          f"contested={snap['contested_n']}")
    _ask_all(engine, "day1", root)
    from eidetic.epistemic.map import KnowledgeMap
    before = json.loads((root / "map_day0.json").read_text())
    delta = KnowledgeMap.delta(before, snap)
    # attach the WHY for every closed cell from the transition ledger
    ledger = engine.knowledge_map_store.transitions_since(scope, before["ts"])
    delta["transitions"] = ledger
    (root / "map_delta.json").write_text(json.dumps(delta, indent=1))
    print(json.dumps({k: delta[k] for k in
                      ("unknown_delta", "contested_delta", "known_delta")}, indent=1))
    print(f"delta -> {root / 'map_delta.json'}")
    engine.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["day0", "night", "day1"])
    ap.add_argument("--root", default="artifacts/epistemic_demo")
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--trials", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root)
    if args.phase == "day0":
        day0(root)
    elif args.phase == "night":
        night(root, waves=args.waves, probes=args.probes, trials=args.trials)
    else:
        day1(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
