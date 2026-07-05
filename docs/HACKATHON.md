# Hackathon demo: shot list + judge walkthrough

Rule for every shot: nothing staged, nothing pre-baked. Each command is one the judge
can re-run from a fresh clone. Numbers on screen must match the committed artifacts.

## Shot 1 -- the honest opening (30s)

Open [docs/claims.md](claims.md). Read the first paragraph out loud: every number ties
to a `run_manifest.json` or is labeled not-yet-measured. Then open
[docs/PUBLIC_CLAIMS.md](PUBLIC_CLAIMS.md) -- the claim is scoped ("best we can
*measure*"), with the refused claims listed right below it.

## Shot 2 -- five-minute verification, offline (2min)

```bash
bash scripts/judge_quickstart.sh
```

On screen, in order: leakage audit PASS (1,670 needles, fails closed), the war-room
demo answering "why did we decide X" with a revision-backed citation in milliseconds,
the rolling never-touched holdout table recomputed live from raw jsonl, snap-back
fidelity 100%.

## Shot 3 -- slice 7, the freshest never-touched window (1min)

```bash
cat artifacts/holdout_rotation_r8_codex/launch_A.log        # pinned SHA + profile
head -c 600 artifacts/holdout_rotation_r8_codex/run_manifest.json  # every score-affecting flag
cat artifacts/holdout_rotation_r8_codex/scoreboard.md
```

Say the hard part out loud: window 7 (r8) is the strongest new-build window; the rolling n=320 table (six wins) -- the build missed its own
internal bars there, and the ledger says so in plain text
([bench/DOMINANCE_PROGRESS.md](../bench/DOMINANCE_PROGRESS.md), "SLICE 7"). The
evidence unit is the rolling n=280 table across seven disjoint windows, not any single
window. That is what honest benchmarking looks like.

## Shot 4 -- live memory with proof (2min)

In a Claude Code session with the MCP plugin mounted:

1. `remember` a fact with a backdated `valid_at`.
2. `recall` it -- show the citation: content hash, validity window, entailment label.
3. Ask something the store cannot support -- show the explicit abstention (no
   confabulation, ever).
4. `truth_ledger` -- the raw-bytes-to-current-truth chain, with supersession.

## Shot 5 -- the cost story, with its caveat (1min)

Open [bench/COST_AB.md](../bench/COST_AB.md), the COST BLITZ table: dev-40 median
query tokens **83** with verify-or-abstain intact vs Mem0's 382 with zero verified
answers. Then the caveat, unprompted: that is a dev-split number; the holdout window's
structured coverage (13/40 on r7) did not reach the dev mix's crossing point, so the
holdout median stays on the reader plateau. Both numbers are on screen, labeled.

## Shot 6 -- forgetting that never lies (30s)

```bash
DATA_DIR=artifacts/holdout_rotation_r8_codex/data \
  .venv/bin/python scripts/snap_back_audit.py --min-records 100
```

272/272 byte-identical after a full benchmark ingest with forgetting on: forgetting is
index-priority only, the substrate never mutates.

## Closing line

"Every number you just saw is in a committed artifact with a pinned SHA. The claims
file lists what we refuse to say. Clone it and check."
