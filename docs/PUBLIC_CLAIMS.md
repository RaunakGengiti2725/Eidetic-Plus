# Public claims — with the evidence path for each

Scope: **limited** (`artifacts/public_ship/claim_scope.json`). The claim we make is
"the best *governed, verified* long-horizon memory agent **we can measure today**" —
measured against the baselines in our own harness under a shared fixed reader and
judge. We explicitly do NOT claim SOTA or best-in-world; see
[claims.md](claims.md) and "What we refuse to claim" below.

Every number here is recomputed from committed raw per-row logs
(`bench/rolling_holdout_table.py`, `bench/cost_report.py`) whose runs are pinned to a
git SHA in their launch logs and whose score-affecting flags are recorded in
`run_manifest.json`.

## Claim 1 — verified answers, not vibes

Across seven disjoint, never-touched LoCoMo holdout windows (n=280), every eidetic
answer is verify-or-abstain: NLI-checked against immutable stored sources with
citations, or an explicit abstention. **Verified answers: eidetic 241 vs Mem0 0.**
Mem0 (and the RAG baselines) return unverified text through the same reader.

Evidence: `artifacts/holdout_rotation_r1..r7_codex/*__run0.jsonl` (`extra.verified`
per row); recompute with `bench/rolling_holdout_table.py`.

## Claim 2 — more correct on rolling never-touched holdout

Rolling seven-window table (each window drawn from a rotation state, ingested fresh,
never used for tuning; window 7 measured with the promoted product_cost stack):

| window | eidetic correct | mem0 correct | margin |
|---|---|---|---|
| r1 | 23/40 | 22/40 | +1 |
| r2 | 17/40 | 18/40 | -1 |
| r3 | 27/40 | 23/40 | +4 |
| r4 | 23/40 | 21/40 | +2 |
| r5 | 24/40 | 17/40 | +7 |
| r6 | 25/40 | 18/40 | +7 |
| r7 | 20/40 | (Phase B rendering) | — |
| rolling | **159/280** | — | — |

Windows swing ±5pp at n=40 — the rolling total is the evidence unit, not any single
window. r7 was a hard draw and missed our internal bars; the ledger says so
(`bench/DOMINANCE_PROGRESS.md`, "SLICE 7 PHASE A").

Temporal-reasoning questions across r1–r7: eidetic 22/51 vs Mem0 3/43 (r1–r6) — the
write-time event-identity path generalized on holdout (2/8 pre-P2 ingest → 6/9 on r6).

## Claim 3 — the structured path is radically cheaper, verification included

On the dev-40 split with the product_cost stack (`bench/COST_AB.md`, COST BLITZ):
**median query cost 83 tokens with full verification vs Mem0's 382 unverified** —
structured claim-backed rows cost 6–85 tokens each, ~60–330× below reader rows.

Honest limit, stated up front: that median is a dev-split number. On holdout r7 the
structured path covered 13/40 rows (dev mix: 21/40), so the holdout median stays on
the reader plateau (4,029). The claim plateau itself transferred (r7 structured rows:
6–55 tokens); the coverage did not. Write-side cost DID transfer, holdout to holdout:
r7 with the stack ingested at 527 write calls / 516,036 real write tokens vs r6
without it at 910 / 887,824 — **−42% write calls, −42% write tokens** on same-mechanics
rotation windows (`bench/cost_report.py` over both artifact dirs).

## Claim 4 — forgetting never destroys, and provably so

Snap-back fidelity over the full r7 benchmark corpus: **272/272 records byte-identical**
(sha256(raw) == content hash) after ingest+consolidation+forgetting
(`artifacts/public_ship/snap_back_audit.json`). Forgetting is index-priority only; the
substrate refuses deletion by design (`tests/test_no_delete_on_forget.py`,
`tests/test_write_once.py`).

## Claim 5 — no leakage, enforced by a failing-closed audit

`python -m bench.audit_no_holdout_leakage`: 1,670 needles (holdout IDs, questions,
answers, banned rescue-policy symbols) scanned over `eidetic/ bench/ tests/ docs/`,
fails closed on empty registry. Current status: PASS.

## What we refuse to claim

- **SOTA / best-in-world.** Requires `bench/reproduce.sh` (≥10 runs, held-out test
  split) plus structured external evidence for Chronos, Mastra, ByteRover, Hindsight.
  Not run. `python -m bench.release_gate` fails closed on exactly this, and its honest
  FAIL report ships in `artifacts/public_ship/`.
- **Any single-window accuracy claim.** ±5pp swings at n=40; r2 and r7 are in the
  table above precisely because we count the losses.
- **The dev-split cost median as a holdout number.** Claim 3's caveat is the
  measurement, not a footnote.
- **Inferred edges as facts.** Dream-inferred connections are labeled
  (`Edge.inferred=True`) and never verify as ground truth.
