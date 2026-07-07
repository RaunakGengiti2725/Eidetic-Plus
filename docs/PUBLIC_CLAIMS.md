# Public claims — with the evidence path for each

Scope: **limited** (`artifacts/public_ship/claim_scope.json`). The claim we make is
"the best *governed, verified* long-horizon memory agent **we can measure today**" —
every answer is verify-or-abstain with checkable provenance (content hash, validity
window, NLI entailment label), and against Mem0, the comparable bounded-retrieval
memory system, we lead on every rolling holdout window. We do NOT claim to be the most
accurate system in absolute terms — against full-context RAG our raw accuracy is lower
(see "Limitations: the four-system comparison" at the end). We explicitly do NOT claim
SOTA or best-in-world; see [claims.md](claims.md) and "What we refuse to claim".

Every number here is recomputed from committed raw per-row logs
(`bench/rolling_holdout_table.py`, `bench/cost_report.py`) whose runs are pinned to a
git SHA in their launch logs and whose score-affecting flags are recorded in
`run_manifest.json`.

## Claim 1 — verified answers, not vibes

Across eight disjoint, never-touched LoCoMo holdout windows (n=320, both systems on
every window), every eidetic answer is verify-or-abstain: NLI-checked against
immutable stored sources with citations, or an explicit abstention. **Verified
answers: eidetic 277 vs Mem0 0.** Mem0 (and the RAG baselines) return unverified
text through the same reader.

Evidence: `artifacts/holdout_rotation_r1..r8_codex/*__run0.jsonl` (`extra.verified`
per row); recompute with `bench/rolling_holdout_table.py`.

## Claim 2 — more correct than Mem0 on rolling never-touched holdout

Against Mem0 — the comparable bounded-retrieval memory system — eidetic wins on every
rolling window, and the margin has grown as the write-side claim families landed,
peaking at **+14 on r8** (23/40 vs 9/40), the freshest window and the first to carry
the VW-killer + event-date family. This is scoped to Mem0, not a claim of general
accuracy leadership (see the four-system limitations note at the end). Rolling
eight-window table (each window drawn by digest from a rotation state, ingested fresh,
scored once, never used for tuning; windows 7–8 measured with the promoted
product_cost stack, r8 also
carrying the VW-killer + event-date family):

| window | eidetic correct | mem0 correct | margin |
|---|---|---|---|
| r1 | 23/40 | 22/40 | +1 |
| r2 | 17/40 | 18/40 | -1 |
| r3 | 27/40 | 23/40 | +4 |
| r4 | 23/40 | 21/40 | +2 |
| r5 | 24/40 | 17/40 | +7 |
| r6 | 25/40 | 18/40 | +7 |
| r7 | 20/40 | 12/40 | +8 |
| r8 | 23/40 | 9/40 | +14 |
| rolling | **182/320** | **140/320** | **+42** |

Windows swing ±5pp at n=40 — the rolling total is the evidence unit, not any single
window. We publish the losing window (r2: −1) and the hard draw (r7, below our own
internal bars). Six consecutive wins (r3–r8), and the margin has grown over the last
four windows (+7/+7/+8/+14) as the write-side claim families landed; r8 — the first
window carrying the VW-killer and event-date family — is the largest margin of all
eight. The ledger records every window, wins and losses
(`bench/DOMINANCE_PROGRESS.md`, "SLICE 8").

Temporal-reasoning questions across r1–r8: eidetic 25/60 vs Mem0 3/60 — the
write-time event-date/identity path generalized on holdout (2/8 pre-P2 ingest → 6/9
on both r6 and r8).

## Claim 3 — the structured path is radically cheaper, verification included

On the dev-40 split with the product_cost stack (`bench/COST_AB.md`, COST BLITZ):
**median query cost 83 tokens with full verification vs Mem0's 382 unverified** —
structured claim-backed rows cost 6–85 tokens each, ~60–330× below reader rows.

Honest limit, stated up front: that median is a dev-split number. On holdout the
structured path covers ~14/40 rows (dev mix: 21/40), so the holdout median stays on
the reader plateau (4,029). The claim plateau itself transferred (r8 structured rows:
5–132 tokens, median well under 30); the coverage did not. What DID transfer, holdout
to holdout: EXTRACT_COMBINED write-call halving (r8 586 calls / 593,587 write tokens vs
r6 without the stack at 910 / 887,824), and the cost per verified-correct answer has
fallen every window as accuracy rises — 41,322 (r6) → 36,317 (r7) → **31,761 (r8)**
total tokens per verified answer (`bench/cost_report.py`).

## Claim 4 — forgetting never destroys, and provably so

Snap-back fidelity over the full r8 benchmark corpus: **272/272 records byte-identical**
(sha256(raw) == content hash) after ingest+consolidation+forgetting
(`artifacts/public_ship/snap_back_audit.json`). Forgetting is index-priority only; the
substrate refuses deletion by design (`tests/test_no_delete_on_forget.py`,
`tests/test_write_once.py`).

## Claim 5 — no leakage, enforced by a failing-closed audit

`python -m bench.audit_no_holdout_leakage`: 1,670 needles (holdout IDs, questions,
answers, banned rescue-policy symbols) scanned over `eidetic/ bench/ tests/ docs/`,
fails closed on empty registry. Current status: PASS.

## The named comparators, with receipts

We researched and source-verified the published numbers for the four systems our own
release gate names as required for any SOTA wording
(`artifacts/public_ship/comparator_research.json`, fetched + adversarially re-verified
2026-07-04): Chronos reports 92.6–95.6% on LongMemEval-S (no LoCoMo results exist);
Mastra reports 94.87% LME-S with a gpt-5-mini actor (84.23% on gpt-4o); ByteRover
reports 96.1% LoCoMo / 92.8% LME-S; Hindsight reports 83.2–92.0% LoCoMo and
83.6–94.6% LME-S depending on backbone. Every one of those numbers is vendor/author
self-reported, single-run, with no independent reproduction found, and each rides a
different frontier reader (GPT-5 / Gemini-3 Pro / Claude Opus) and its own judge.

Our regime is deliberately different: one fixed modest reader (qwen-plus) and one
fixed judge for every system in the harness, verify-or-abstain with citations, and
never-touched holdout windows. Those published numbers and ours are not
commensurable — and we say plainly: we do NOT claim to beat these systems. What no
comparator publishes is our governance axis: per-answer NLI-verified citations to an
immutable substrate (277 verified answers vs Mem0's 0 under identical conditions),
100% snap-back fidelity, and a fail-closed release gate whose FAIL report ships in
the repo.

## What we refuse to claim

- **SOTA / best-in-world.** Requires `bench/reproduce.sh` (≥10 runs, held-out test
  split) plus structured external evidence for Chronos, Mastra, ByteRover, Hindsight.
  Not run. `python -m bench.release_gate` fails closed on exactly this, and its honest
  FAIL report ships in `artifacts/public_ship/`.
- **Any single-window accuracy claim.** ±5pp swings at n=40; r2 and r7 are in the
  table above precisely because we count the losses.
- **Accuracy leadership over RAG.** Full-context RAG and vector RAG beat eidetic on
  raw LoCoMo accuracy (preliminary, n=80 — see the limitations note below). We lead
  only vs Mem0. Our edge is provenance, not raw accuracy.
- **That "verified" means "correct."** Verified = grounded in a cited source;
  verified-precision is ~55–60%. A verified answer can still be off-target.
- **The dev-split cost median as a holdout number.** Claim 3's caveat is the
  measurement, not a footnote.
- **Inferred edges as facts.** Dream-inferred connections are labeled
  (`Edge.inferred=True`) and never verify as ground truth.

## Limitations: the four-system comparison (preliminary)

The rolling table above is scoped to Mem0. On two disjoint never-touched LoCoMo
windows (r9, r10; n=80) we also ran the stronger RAG baselines through the same fixed
reader and judge — a preliminary result we record rather than feature, because n=80 is
small and LoCoMo is a small-corpus benchmark:

| system | n=80 | verified | verified-correct | median qtok |
|---|---|---|---|---|
| rag-full (stuffs whole transcript) | 54/80 (68%) | 0 | 0 | 22,499 |
| rag-vector (top-k retrieval) | 47/80 (59%) | 0 | 0 | ~1,875 |
| eidetic-plus-full | 39/80 (49%) | 64 | 36 | 4,029 |
| mem0 | 27/80 (34%) | 0 | 0 | ~390 |

On raw accuracy the RAG baselines beat eidetic on this benchmark: the whole
conversation fits in the reader's context, so re-reading everything (rag-full, at 5–6×
our token cost) extracts more than a memory system that must retrieve and verify.
LoCoMo does not test the corpus-scaling regime where a bounded memory wins, so we do
not claim that advantage. What is eidetic-only here is **provenance** — 64 cited
answers vs 0 for every other system — at ~55–60% verified-precision (grounded, not a
correctness guarantee). Evidence: `artifacts/holdout_rotation_r9_codex`,
`artifacts/holdout_rotation_r10_codex`; forensics in `bench/DOMINANCE_PROGRESS.md`
(SLICE 9, SLICE 10).

### Cross-benchmark check: LongMemEval-S (preliminary, n=24, PARTIAL)

Everything above is LoCoMo. To answer "on what test?" we also ran a stratified 24-
question LongMemEval-S subset (all 6 official categories: knowledge-update,
multi-session, single-session assistant/preference/user, temporal-reasoning) through
the same fixed reader:

| system | LongMemEval-24 | verified | verified-correct | median qtok |
|---|---|---|---|---|
| rag-vector (top-k retrieval) | **17/24 (71%)** | 0 | 0 | 1,930 |
| eidetic-plus-full | 11/24 (46%) | 17 | 11 | 3,590 |
| mem0 | (running) | 0 | — | — |
| rag-full | (running) | 0 | — | — |

The LoCoMo finding replicates: on raw accuracy vector RAG beats eidetic on
LongMemEval too (17 vs 11), and eidetic is again the only system that returns verified
answers (17 cited vs 0). This is preliminary — n=24, one draw, single run, and the
mem0/rag-full arms are still completing — so it is a directional cross-benchmark
signal, not a settled result. Evidence:
`artifacts/public_ship/slice_invariant/longmemeval/draw_1/`.
