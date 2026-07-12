# Public claims — with the evidence path for each

Scope: **limited** (`artifacts/public_ship/claim_scope.json`). The current runtime is a
governed, verified long-horizon memory agent: every public factual answer is
verify-or-abstain with checkable provenance (content hash, validity window, NLI
entailment label). Against Mem0, the comparable bounded-retrieval memory system, the
committed rolling aggregate leads while individual windows include a loss. We do NOT
claim to be the most accurate system in absolute terms — against full-context RAG our raw accuracy is lower
(see "Limitations: the four-system comparison" at the end). We explicitly do NOT claim
SOTA or best-in-world; see [claims.md](claims.md) and "What we refuse to claim".

Every number here is recomputed from committed raw per-row logs
(`bench/rolling_holdout_table.py`, `bench/cost_report.py`) whose runs are pinned to a
git SHA in their launch logs and whose score-affecting flags are recorded in
`run_manifest.json`.

## 2026-07-10 proof-boundary correction

The current runtime has one public factual answer contract: `VERIFIED` or `ABSTAINED`.
`Engine.ask`, HTTP, MCP, fixed-reader evaluation, and governed NotebookLM recall share
the same implementation. `verify=false` cannot bypass it. Verification requires exact
scope, query-time activity, raw-byte hash resolution, source-span resolution, no active
contradiction, and per-claim proof for sentences and list items. Image claims are checked
against pixels; PDF/audio/video claims are re-read from immutable media. Direct Gemini
output is labeled `UNTRUSTED_DRAFT` and is not a normal answer.

The burned-window mechanical replay in `artifacts/replay_phase_a_r1_r10/` corrects an
older claim below: the ten frozen r1–r10 logs contained **12 unverified delivered
answers**, not zero. Applying the current output policy converts all 12 to abstentions,
leaves **0 unverified delivered**, and preserves **212/212 verified-correct rows**. The
artifact is SHA-256-bound to all ten source logs, all outputs, and `bench/replay.py`, and
made zero provider/NLI/generation calls. It is explicitly a policy projection, not a
provider rerun.

This does **not** establish accuracy-first release readiness. The same historical logs
contain **129/341 (37.8%) verified-but-judge-wrong rows**: 46 single-hop, 40 multi-hop,
37 temporal, and 6 open-domain. `VERIFIED` means every delivered claim is grounded in
resolving evidence under the current proof rules; it does not prove task correctness or
completeness. The runtime now adds contradiction, temporal-validity, source-span,
per-item, and multimodal fail-closed guards, but fresh unseen reruns are still required.

## 2026-07-11 verified-precision guard wave (mechanical projections, no fresh window yet)

Two selection/form-side guard families landed against that 129-row verified-wrong
panel, each measured OFFLINE against frozen rows before shipping (zero provider calls;
both artifacts hash-bound to implementation bytes):

- **Pure form floors** (dangling separator tails, degenerate conjunction repetition,
  junk-stripped question echoes, preference category-object anchoring, when-type
  agreement): projected over the 341 frozen verified rows, **21 verified-wrong rows
  convert to abstention; 3 verified-correct rows are lost** (all three are the
  pre-existing first-person clean-fact shape, enumerated in the artifact).
  Evidence: `artifacts/guard_projection_r1_r10/`.
- **relative_temporal derivation boundary**: the legacy candidate loop now tags every
  shipped date `atom_derived` (the winning atom's own expression dates the event, or a
  deterministic rule resolved the contest) or `mention_selected` (score/hit ordering
  alone picked among materially conflicting dated mentions); `mention_selected` fails
  closed. Selection-replayed against ten burned windows' frozen stores:
  **12/35 verified-wrong rows convert to abstention; 26/28 verified-correct rows keep
  their exact answers; 0 atom-derived regressions attributable to the diff** (one
  pre-existing drift row and one deliberate fail-closed tie are itemized, with a
  stashed-tree baseline report shipped for the attribution).
  Evidence: `artifacts/forensics/selection_replay_20260711.json` +
  `selection_replay_baseline_20260711.json`, harness `bench/selection_replay.py`.

These are projections over burned windows. The first never-touched window under this
stack (r18, 2026-07-11) then measured live: **0 unproven deliveries on 40 fresh rows**
(the r1–r10 era had shipped 12), verified precision 64.3% vs the 62.2% historical
panel, and **none of the 12 abstentions came from these guards** (8 contradiction-gate,
4 NLI confidence floor). The same window is a raw-accuracy loss to Mem0 (18/40 vs
21/40, recorded in the Claim 2 table) — the strict boundary's coverage cost is visible
and we publish it; contradiction-gate calibration forensics is the named next step.

## 2026-07-11 age-neutral ranking, now enforced and proven on the shipped path

The age-independence claim previously covered only the raw vector index while the
shipped hybrid ranker fused a recency channel (`RRF_W_RECENCY=0.3`) — a real
contradiction, now closed: the recency fusion weight **defaults to 0.0** (the channel
and its underfill fallback are dead code unless a run explicitly opts in, which its
manifest must then show), and `Engine.prove_age_independence` probes **both** the raw
index and the full `retrieve()` fusion path (`flat` requires both recall curves flat).
Offline proof: ranking is byte-identical under age permutation of the same corpus at
the default, and provably diverges only under explicit opt-in
(`tests/test_age_independence.py`).

## Claim 1 — verified answers, not vibes

The current runtime is verify-or-abstain. Historical rolling artifacts predate that
hard boundary and must be read through the correction above rather than described as
having zero unverified deliveries. On the r1–r10 replay panel, 341/400 historical rows
were labeled verified, 47 had already abstained, and 12 unverified answers had been
delivered; the mechanical replay converts those 12 to abstentions. Mem0 and RAG rows
remain unverified comparator outputs under the shared reader.

Evidence: `artifacts/replay_phase_a_r1_r10/replay_report.json`, its manifest, and the
source `artifacts/holdout_rotation_r1_codex` through `r10_codex` JSONL logs.

## Claim 2 — aggregate lead over Mem0 on rolling never-touched holdout

Against Mem0 — the comparable bounded-retrieval memory system — eidetic leads in the
committed rolling aggregate, while the published per-window record includes a loss. The
margin grew as the write-side claim families landed,
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
| r16 | 25/40 | 23/40 | +2 |
| r18 | 18/40 | 21/40 | -3 |
| rolling | **225/400** | **184/400** | **+41** |

Windows swing ±5pp at n=40 — the rolling total is the evidence unit, not any single
window. We publish the losing windows (r2: −1; r18: −3) and the hard draw (r7, below
our own internal bars). Six consecutive wins (r3–r8) preceded r18 — the first fresh
window under the strict proof boundary, where a 30% abstention rate (12 rows, all
contradiction-gate or NLI-floor, none from the newest guards) cost raw coverage while
delivering 0 unproven answers; the ledger's SLICE 18 carries the full forensic
separation. The ledger records every window, wins and losses
(`bench/DOMINANCE_PROGRESS.md`, "SLICE 8", "SLICE 18").

Temporal-reasoning questions across r1–r8: eidetic 25/60 vs Mem0 3/60 — the
write-time event-date/identity path generalized on holdout (2/8 pre-P2 ingest → 6/9
on both r6 and r8). On r18 the temporal edge held under the new derivation boundary:
4/8 vs Mem0 1/8 and rag-vector 1/8.

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

## Claim 3b — verify-or-abstain is now fail-closed on derived numeric aggregates (2026-07-09)

The live LME-S numeric panel exposed a leak in the core promise: 5/13 computed answers
(cross-session counts/sums) shipped `verified=True` while WRONG — every cited atom was real,
but the atom SET or arithmetic was wrong, which NLI-against-sources cannot catch. Fixed by
an aggregate CITATION floor (`eidetic/smqe/verify.py`): a count or cross-session sum is
verified ONLY when a single cited source states the value; a value DERIVED by enumerating
across atoms abstains. Live panel now **0 verified-wrong** (4 verified-correct preserved,
9 abstentions). The trade is explicit and chosen: correct-or-silent over coverage.

Evidence: `bench/measure_sum_live_probe.py` against
`artifacts/lme_s_r1_codex/measure_sum_live.json`; release gate re-scoped with auditable
denominators (`artifacts/public_ship/GATE_STATUS.md`, update 2026-07-09).

## Claim 3c — the free-read product row, judged, with checkable provenance

Different-reader product row (NotebookLM/Gemini free tier reads eidetic's exported verified
sources; 0 tokens on the caller's metered model) — kept OUT of the fixed-reader benchmark
table, never merged:

- **Judged accuracy: 102/120 = 85.0%** over three disjoint LoCoMo holdout windows
  (r14 77.5% / r13 92.5% / r15 85.0%, r15 fully prospective), pinned qwen3-max judge,
  vs same-window/same-judge rag-vector 74/120 = 59.2% (paired McNemar p<1e-5).
- **The ≥10-run reproduce gate: PASS (2026-07-11).** Ten full same-window collections
  (r15), every run judged with the pinned judge: mean 82.2%, bootstrap 95% CI
  [81.3, 83.2] — the CI FLOOR clears the same-window rag-vector fixed-reader
  comparator (57.5) by 23.8pp (`artifacts/holdout_rotation_r15_codex/notebooklm_gate.json`,
  runs 0-10 committed, run3 excluded as quota-truncated). The gate's own honest note
  stands: statistical reproducibility, not by itself a best/SOTA claim.
- **Citation provenance, measured live (2026-07-09):** the quote-content citation map
  resolves Gemini's `[n]` references to immutable content hashes on **25/26 rows (96%),
  198/204 references (97%)** of the LME-S retrieval-guided panel — revalidated
  2026-07-11 after resolver hardening at **26/26 rows (100%)**
  (`provenance_citation_map_live_20260711_0000.json`)
  (`artifacts/lme_s_r1_codex/provenance_citation_map_live.json`,
  gate `bench/provenance_live_probe.py`). Attribution is conservative: ambiguous or
  unmatched quotes are returned unattributed with the reason, never guessed.
- The direct research surface remains Gemini-side and is returned as
  `output_type=UNTRUSTED_DRAFT`, never as a normal verified answer. The product
  `NotebookLMBridge.governed_recall` path submits that draft and the exact exported evidence IDs
  to `Engine.prove_external_draft`; it returns only `VERIFIED` with resolving proof or
  `ABSTAINED`. Proof-model usage is reported separately from the 0 caller-generation-token read.

## Claim 3d — the export-truncation correction, measured (2026-07-10)

A one-line packing bug (`packed[:max_sources]`) silently dropped store records past a
fixed source budget, so the free reader answered "no information" about facts the store
held. Found by a 69-agent per-miss forensics fleet (16/58 judged misses traced to it,
3/3 adversarial skeptics confirmed mechanically), fixed lossless-or-loud, then measured
paired on fresh notebooks, same window, same pinned judge:

**LME-S whole-export: 25.0% → 78.6% (McNemar exact p = 0.0003; 16 fixed / 1 lost —
the 16 are exactly the fleet's predicted cluster).** knowledge-update went 0/4 → 4/4.
**Reproduced 2026-07-11 on fresh -v3 notebooks: 25/30 (83.3%)** — two independent
collections both far above the 25.0% baseline; no longer single-run directional. Evidence:
`artifacts/lme_s_r1_codex/notebooklm_freetier_v2pack.*`,
`artifacts/forensics/miss_forensics_fleet_20260709.json`.

## Claim 3e — same-version hardened-judge head-to-head (2026-07-10)

Judge v2 (5 fleet-confirmed LoCoMo gold defects quarantined symmetrically + a
deterministic gold-containment pre-check, replay-validated at zero false positives)
applied to BOTH sides across the three disjoint judged holdout windows, 115 paired rows:

**eidetic NBL product row 104/115 (90.4%) vs rag-vector 71/115 (61.7%) — discordant
34:1, McNemar exact p = 2.1×10⁻⁹.** The hardening helped the comparator too (59.2 →
61.7), confirming symmetry. v1 numbers remain the published baseline; v2 sidecars ship
alongside (`*.judged_v2.json`). Single collection run per window for both systems;
product-row variance bounded by the 7-run same-window band (84.2–86.8% under v2).

## Claim 3f — first raw-accuracy parity window with vector RAG (2026-07-10)

On r16 — a never-touched window whose fresh ingest carried the write-path fix stack —
the neutral fixed-reader row TIED vector RAG on raw accuracy for the first time
(25/40 = 25/40; the r9–r14 aggregate had eidetic 5.5pp behind), while keeping the
provenance column eidetic-only (25 verified answers vs 0). One window, ±5pp swings:
parity is directional until repeated. Honest non-effects on the same window: temporal
2/8 (unmoved), structured coverage 10/40. Evidence: `artifacts/holdout_rotation_r16_codex/`.

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

(We measured the fair apples-to-apples pair — eidetic and vector RAG, both
bounded-retrieval — through the same reader. The mem0 and rag-full arms were deferred:
mem0 is a weak baseline and rag-full's full-history stuffing is prohibitively slow on
LongMemEval's long multi-session inputs; neither changes the comparison that matters.)

The LoCoMo finding replicates: on raw accuracy vector RAG beats eidetic on
LongMemEval too (17 vs 11), and eidetic is the only system that returns verified
answers (17 cited vs 0). This is preliminary — n=24, one stratified draw, single run —
a directional cross-benchmark signal, not a settled result. Per-category eidetic:
single-session-user 4/4, knowledge-update 3/4, temporal-reasoning 2/4, multi-session
1/4, single-session-assistant 1/4, single-session-preference 0/4. Evidence:
`artifacts/public_ship/slice_invariant/longmemeval/draw_1/`.
