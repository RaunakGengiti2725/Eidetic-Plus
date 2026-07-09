# LongMemEval-S measurement — the honest "beat #1" gate (2026-07-09)

Chronos (PwC, arXiv 2603.16862) reports **95.6% on LongMemEval-S** — the number this goal
targets. Every prior eidetic number was LoCoMo. This is the first eidetic run on LongMemEval-S:
never-touched ledgered slice (window 0, digest 988f7ba6, n=30), shared qwen-plus reader +
qwen3-max judge (inline), fresh data dir.

| system | LME-S accuracy (judged) | median caller-tokens/q |
|---|---|---|
| rag-vector | **19/30 = 63.3%** | 1941 |
| eidetic fixed-reader (eidetic-plus-full) | **16/30 = 53.3%** | 6096 |
| Chronos (PwC, **reported, NOT reproduced**) | 95.6% | its own agentic reader (Cohere rerank + multi-LLM + ReAct); no code |

## The honest reading
- **On the target benchmark, the eidetic fixed-reader LOSES to vector RAG on BOTH accuracy
  (53.3% vs 63.3%) and cost (6096 vs 1941 tokens).** The LoCoMo finding replicates exactly:
  the neutral fixed-reader path is not an accuracy or cost winner.
- **We are ~42 points below Chronos's reported 95.6%**, and Chronos is unreproducible
  head-to-head (no code; its number comes from its own agentic reader, not the shared qwen
  reader). "Beat #1" is not close on measured evidence.
- The failure is broad, not one lever: multi-session aggregation, temporal reasoning, recall
  gaps (honest abstentions on answerable questions), and a verified-precision hole
  (10/30 verified-wrong).

## What is still true (and measured)
- eidetic's **cost** win is real ONLY on its non-fixed-reader paths: the structured tier
  (~20-146 tokens, measured on LoCoMo) and the NotebookLM free read (0 caller tokens). The
  metered fixed-reader is the expensive path and is not the product story.
- eidetic's **uniqueness** — content-hash-verifiable provenance + qwen post-hoc audit — stands.
- The only realistic path to a COMPETITIVE LongMemEval-S accuracy number is the NotebookLM
  free-read config (85% vs 55% fixed-reader on LoCoMo), which is quota-gated (task #37) and,
  honestly, would still likely land below 95.6%.

## Bottom line
This measurement is the point of measuring: it refutes a "most powerful / beats #1" claim.
On LongMemEval-S the current eidetic path is **53.3%, below rag-vector, far below Chronos.**
No SOTA claim is made or supported. The defensible, measured claims remain cost (on the free/
structured paths) and verifiable provenance — not accuracy leadership.

## Update — NotebookLM free-read on LME-S = 25.0% (judged), the export bug (2026-07-09)

The free-read config (85% on LoCoMo, the hoped competitive path) was run on the SAME LME-S
slice and judged by the pinned qwen3-max judge: **7/28 = 25.0%** (2 nlm errors). Complete
honest LME-S table:

| path | LME-S accuracy | note |
|---|---|---|
| Chronos (PwC) | 95.6% | reported, unreproduced, own agentic reader, no code |
| rag-vector | 63.3% | shared qwen reader |
| eidetic fixed-reader | 53.3% | shared qwen reader |
| **eidetic NotebookLM free-read** | **25.0%** | 0 caller tokens, but the long-session export bug |

**Every eidetic path loses to rag-vector on LME-S; the free-read collapses from its LoCoMo
85%.** Root cause (read-only store check): the answer facts ARE in the store but buried in
long LongMemEval-S records (a fact inside a multi-paragraph "write a blog post about..."
turn), so NotebookLM reads over one big block and answers "no information" (only 3/… cited
tokens confirmed). LoCoMo turns are short, so this only bites long-session benchmarks.

Fix built this session (commit 5e6f23529, flag-gated default-off): `format_source_chunks` +
`NLM_CHUNK_CHARS` split long records into per-chunk sources (provenance on each) so buried
facts surface. UNMEASURED end-to-end — a fresh chunked re-run is task #40. Honest ceiling
even if it works: a different-reader product row, still below Chronos 95.6%.

**Bottom line, unchanged and now complete: "most powerful / beat #1" is refuted on every
measured path.** Defensible measured claims remain cost (free/structured paths) + verifiable
provenance — not accuracy leadership.

## BREAKTHROUGH — retrieval-guided free-read = 78.6% on LME-S (2026-07-09)

The whole-conversation free-read failed (25%) by burying facts. Fix (this session, commit
e2921026d): eidetic's qwen retriever picks the top-k question-relevant records -> export
ONLY those to a focused per-query notebook -> NotebookLM/Gemini free-reads. Judge-scored on
the SAME never-touched LME-S slice (pinned qwen3-max, n=28, 0 errors):

| path | LME-S accuracy | caller tokens |
|---|---|---|
| Chronos (PwC, reported, **unreproduced**) | 95.6% | its own agentic reader |
| **eidetic retrieval-guided free-read** | **78.6% (22/28)** | **0** |
| rag-vector | 63.3% | 1941 |
| eidetic fixed-reader | 53.3% | 6096 |
| eidetic free-read (whole-conversation) | 25.0% | 0 |

**The retrieval-guided free-read (78.6%) BEATS rag-vector (+15.3) and the fixed-reader
(+25.3) at ZERO caller tokens** -- a +53.6-point lift over its own whole-conversation
baseline. Spot-checked: the recovered answers are genuinely correct (aggregation counts
Negroni 5->10, poems 17->23; "no mention"->Dark Souls 3 DLC), not judge leniency. Novel
shape: **qwen retrieves, Gemini free-reads a focused set, provenance rides on every source.**

### Honest boundaries (non-negotiable)
- **Still BELOW Chronos's reported 95.6% by ~17 points.** This does NOT make eidetic "#1 /
  the most powerful." No SOTA/best claim.
- **Single run, n=28, DIFFERENT reader** (Gemini, off-meter) -- a labeled product row, not
  the neutral fixed-qwen table and NOT the >=10-run reproduce gate. Variance unmeasured.
- **Provenance gap:** Gemini cited [1]-style, not the eidetic:<id> tokens (cited=0 confirmed);
  quote-grounding still applies, but content-hash citation needs the focused sources to carry
  tokens Gemini will surface -- an open item.
- Chronos remains unreproducible head-to-head (no code, agentic reader).

### What it means, honestly
On accuracy, eidetic is NOT #1 (Chronos leads). But **the retrieval-guided free-read is now
the strongest measured eidetic path on LongMemEval-S, beats the standard baselines at zero
caller cost, and is fully provenance-carrying** -- a genuine product result on the
verifiable-memory axis, discovered + built + measured this session. Not the crown; real
progress toward it, honestly bounded.

---

## Advisory-injection measurement — a WASH within noise, real per-op signal (2026-07-09)

Tested whether prepending eidetic's deterministic `structured_recall` value as a
**non-authoritative** advisory source lifts the retrieval-guided free-read. Same LME-S
window, **paired** (identical sample_ids), one fixed qwen3-max judge for both arms.

| arm | accuracy (paired, n=27) |
|---|---|
| retrieval-guided free-read | 22/27 = 81.5% |
| retrieval-guided **+ advisory-injection** | 24/27 = 88.9% |
| raw delta | **+2 questions (+7.4pp)** |

**The raw +7.4pp is NOT a real lift — it is within noise.** 6 discordant pairs, McNemar
exact two-sided p ≈ 0.69: indistinguishable from a coin flip. Per-op attribution (advisory
presence detected via `n_sources` delta — the advisory adds one source):

| flipped question | advisory injected? | honest attribution |
|---|---|---|
| bronchitis (gain) | **no** (identical sources) | Gemini run-to-run variance |
| kitchen-preference (loss) | **no** | Gemini run-to-run variance |
| "how old when Alex born?" gold=11 (gain) | yes | **judge false-positive** — injected answer literally says "do NOT contain any information regarding Alex"; judge wrongly scored it correct |
| weddings gold="three" (gain) | yes | plausible; advisory value correct (3, not the count_aggregate 3→4 overcount) |
| **"29 days" between two events (gain)** | yes | **CLEAN gain** — two-event-delta op computed 29; reader answered "Exactly 29 days … [1]" citing the advisory |
| **"total feed weight" gold=70 lb (loss)** | yes | **CLEAN loss** — `_measure_type_sum` advisory computed **1,226.3 lb** (garbage); reader trusted it ("structured recall of your claims"); the retrieval-guided baseline got 70 lb RIGHT by reading records |

**Net verified injection effect: +1 (a correct deterministic op) / −1 (a buggy one).**
Two of six flips carried no advisory at all (pure reader variance, they cancel); one
"gain" is a judge false-positive on a non-answer. The aggregate is a wash.

### What this actually shows (honest)
- Advisory-injection is **NOT a net accuracy lift** on this window. It is a per-op
  amplifier: a *correct* deterministic op (two-event day-gap) verifiably helps; a *buggy*
  op (measure-type-sum: 1,226.3 vs a true 70) verifiably **corrupts an otherwise-correct
  free-read**, and the "trust the records if they disagree" label did **not** prevent it.
- Correct posture, unchanged: **injection stays flag-OFF by default** (`inject_computed=False`).
  Off-by-default ⇒ no shipping harm; it is a research lever, not a product default.
- **Characterized bug (not fixed this session):** `_measure_type_sum_answer` summed unrelated
  quantities to 1,226.3 lb for "total weight of feed purchased in the past two months"
  (gold 70). Fixing risky numeric code post-long-session, without the ≥10-run gate, is how
  regressions land — recorded as a lever, deferred.
- **Judge artifact found:** a "no information about Alex" non-answer was scored correct vs
  gold "11". Affects both arms ~equally on aggregate; noted for the judge audit.
- Still **< Chronos 95.6%.** No #1 / SOTA / "most powerful" claim — unchanged.

Raw data: `artifacts/lme_s_r1_codex/notebooklm_rg_injected.jsonl` (28 rows, 27 ok);
paired scores: `artifacts/lme_s_r1_codex/rgi_scored.json`.

---

## CORRECTION — live-store measurement: the scope-gate fix does NOT fix the real case (2026-07-09)

Ran the deterministic `structured_recall` path against the **real built LME-S stores**
(`DATA_DIR=artifacts/lme_s_r1_codex/data`, embeddings cached) for the 13 numeric-aggregation
questions. This corrects an overstatement in commit `c64c10aee`.

**The `_measure_type_sum` scope-gate (c64c10aee) does NOT resolve bc149d6b on real data.**
The live store still returns **`1226.3 pounds`, verified=True** (gold 70) — unchanged by the
fix. My synthetic regression encoded the WRONG failure mode: unrelated-subject weights
(body/gym). The REAL over-count is *same-subject* — the 6 summed supports are all from
feed-heavy memories, so the per-atom/per-group subject-gate keeps them all. The spurious
`1226.3` comes from claim-level weight extraction (e.g. "cost per pound of layer feed"
reasoning text) and/or missing temporal scope ("past two months", "new"), NOT from summing
gym plates. The fix is a defensible general hardening but is verified ONLY on synthetic
atoms and does not touch this live case. **A verified-WRONG numeric answer still ships here.**

Live structured-path results on the 13 numeric questions (op | answer | gold):

| sample | op | answer | gold | ok |
|---|---|---|---|---|
| 603deb26 | count_aggregate | 10 | 10 | ✓ |
| gpt4_2f91af09 | multi_session_sum | one | 23 | ✗ (undercount/count-misroute) |
| gpt4_8e165409 | temporal_delta | 14 days | 14 days | ✓ |
| f0e564bc | multi_session_sum | "Apply the Dr" | $1,300 | ✗ (non-numeric garbage, verified?) |
| 5c40ec5b | count_aggregate | twice | twice | ✓ |
| gpt4_85da3956 | temporal_delta | (abstain) | 3 weeks ago | ✗ (miss) |
| bc149d6b | multi_session_sum | 1226.3 pounds | 70 pounds | ✗ (**verified-WRONG, still ships**) |
| gpt4_2f8be40d | count_aggregate | 4 weddings | three | ✗ (overcount) |
| 06db6396 | count_aggregate | 6 | 5 | ✗ (overcount) |
| a3838d2b | count_aggregate | 3 | 4 | ✗ (undercount) |
| 3fe836c9 | latest_value | $25,000 | $25,000 | ✓ |
| gpt4_74aed68e | temporal_delta | 29 days | 29 days | ✓ |
| a4996e51 | count_aggregate | (abstain) | 50 | ✗ (miss) |

**5/13 correct on the deterministic path.** The LME-S numeric-aggregation failure is broad
(count over/undercount, sum misroute, non-numeric leak, temporal miss) and NOT closed by
tonight's scope-gate. `_measure_type_sum` fix stands as narrow synthetic hardening only;
bc149d6b + the count/sum family remain OPEN (#41/#42). No accuracy progress claimed; the
honest lesson is that a suite-green synthetic fix must be validated on the live failure set
before any framing — which this measurement now enforces. Raw: `measure_sum_live.json`.
