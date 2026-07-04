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
   seven disjoint never-touched LoCoMo windows.
4. **Snap-back fidelity** — 272/272 records byte-identical after a full benchmark
   ingest with forgetting on.
5. Pointers to every number's artifact path.

## The three headline results (full statements in [PUBLIC_CLAIMS.md](PUBLIC_CLAIMS.md))

- **Verified 241 vs 0.** Across n=280 rolling holdout, every one of our 241 verified
  answers carries citations to immutable sources; Mem0's answers verify nothing.
- **Rolling holdout lead.** 159/280 correct across seven never-touched windows vs
  Mem0's 119/240 on the six completed head-to-heads (window 7's Mem0 phase renders
  from the same launch script). We publish the losing windows too (r2: −1, r7: hard
  draw below our own bar).
- **Structured answers at 6–85 tokens, verification included.** Dev-40 median 83
  tokens vs Mem0's 382 unverified; on holdout the plateau transferred but coverage
  didn't (13/40), so we do not claim the dev median as a holdout number. Write-side
  −42% held holdout-to-holdout.

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
