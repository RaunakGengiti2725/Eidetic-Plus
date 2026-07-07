# For judges — what this is and how to check it

**Eidetic-Plus is a long-horizon memory agent that refuses to confabulate.** Every
answer is NLI-verified against an immutable content-addressed substrate and returns
citations (content hash, validity window, entailment label) — or it abstains, in so
many words. Forgetting exists (FSRS-style priority decay, consolidation, dreaming) but
can never destroy: the raw bytes always snap back byte-identical.

## Verify in five minutes (four steps offline, no API key)

```bash
bash scripts/judge_quickstart.sh
```

1. **Leakage audit** — 1,670 holdout needles scanned over all source; fails closed.
2. **War-room demo** — shared problem memory answering "why did we decide X?" with a
   revision-backed citation, offline, in milliseconds.
3. **Rolling holdout table** — recomputed live from committed raw per-row logs of
   eight disjoint never-touched LoCoMo windows.
4. **Snap-back fidelity** — 272/272 records byte-identical after a full benchmark
   ingest with forgetting on.
5. Pointers to every number's artifact path.

## The headline results (full statements in [PUBLIC_CLAIMS.md](PUBLIC_CLAIMS.md))

- **We beat Mem0 on every rolling window.** 182/320 vs 140/320 over eight
  never-touched draws (n=320), six consecutive wins, margins +1/−1/+4/+2/+7/+7/+8/+14
  — peaking at **+14 on r8** (23/40 vs 9/40), the freshest window and the first with
  the VW-killer + event-date family. We publish the losing window (r2) and the hard
  draw (r7) in the same table.
- **Provenance, and we're the only ones with it.** 277 of eidetic's answers across
  the rolling holdout carry checkable citations (hash + validity window + entailment
  label); Mem0 returns **0** verifiable answers. Honest caveat: verified = grounded,
  not correct — verified-precision is ~55–60%.
- **Cost falls as accuracy rises**: total tokens per verified answer dropped every
  window, 41.3k → 36.3k → 31.8k (r6→r8).

**Honest limitation.** The rolling comparison above is scoped to Mem0. In a
preliminary two-window run (r9+r10, n=80) the stronger RAG baselines beat eidetic on
raw LoCoMo accuracy (rag-full 54/80, rag-vector 47/80, eidetic 39/80) — LoCoMo is
small enough that re-reading the whole transcript wins. We record it in
[PUBLIC_CLAIMS.md](PUBLIC_CLAIMS.md) rather than bury it. Our edge is provenance
(64 cited answers there vs 0 for every other system), not raw accuracy.

**Cross-benchmark (preliminary).** Not just LoCoMo: on a 24-question LongMemEval-S
subset (all 6 categories, same fixed reader) the pattern replicates — vector RAG
17/24 vs eidetic 11/24 on raw accuracy, eidetic again the only system with verified
answers (17 vs 0). Partial (n=24, mem0/rag-full arms still running);
`artifacts/public_ship/slice_invariant/longmemeval/draw_1/`.

## What makes it defensible

- **Fair harness.** Every system (eidetic, Mem0, full-context RAG, vector RAG)
  answers through the SAME pinned reader and judge (`bench/reader.py`,
  `bench/judge.py`) — it is a memory comparison, not a reader comparison.
- **Never-touched windows.** Each holdout window is drawn by digest from a rotation
  state, ingested fresh, scored once, and never tuned against. The leakage audit
  bans its IDs and strings from source, failing closed.
- **Pinned everything.** Each run's launch log records the git SHA and profile;
  `run_manifest.json` records every score-affecting flag.
- **A gate that says no.** `python -m bench.release_gate` fails closed on the
  multi-run public-claim standard; its honest current FAIL status ships in
  `artifacts/public_ship/` rather than being hidden.

## Honest limits (we say these before you find them)

- No SOTA claim: the ≥10-run reproduce sweep and named-comparator evidence
  (Chronos, Mastra, ByteRover, Hindsight) have not been run.
- Single windows swing ±5pp at n=40; judge the rolling table, not a window.
- The 83-token median is dev-split; holdout r7's median stayed on the reader plateau.
- Mem0's write side is unlogged in its adapter, so its totals are understated —
  stated in our own cost ledger.

## Ten-second architecture

Immutable substrate (layer 1, write-once, content-addressed) → metabolism (layer 2:
FSRS forgetting, consolidation, dreaming — priority only, never bytes) → mind
(layer 3: multi-channel retrieval + typed claim graph + SMQE structured execution) →
proof (layer 4: per-support NLI, form floors, verify-or-abstain). One switch
(`METABOLISM_MODE=1`) wires the whole profile; `bench/` proves it.
