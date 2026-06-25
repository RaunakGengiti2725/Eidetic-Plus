# Fastest, Best Memory Agent

Status: **all waves of the [plan](/Users/raunakgengiti/.cursor/plans/fastest_best_memory_agent_dee8a701.plan.md) implemented** — the safety/governance foundation, the speed waves, and the competitive (benchmark-winning) track — built in the plan's prerequisite order, committed wave-by-wave on branch `connected-brain-loop` for bisectability.

**Honest headline:** F0 fixed a **real latent corruption bug** and stands on its own. Every other wave's definition-of-done is a *measured* number (p95, throughput, 429 behavior, beats-Mem0). The DashScope quota is 403-blocked, so those waves are built **correct, tested, and flag-off** but **not demonstrated as faster or winning** here. Nothing is promoted to default-on on a benchmark basis; the speed/competitive wins are measurable the moment a funded key exists.

## Safety + governance (the prerequisite layers)

| Wave | What | Flag-gated? |
|---|---|---|
| **F0** | Engine `_write_lock` (no model call ever held under it), atomic index/BM25 saves (temp+`os.replace`), lock-free consistent reads (snapshot+clamp), `set_num_threads(1)`, thread-safe caches, **thread-local `RecallTrace`**, singleton init locks | No (fixes a bug that exists today) |
| **F1** | `RateGovernor`: token-bucket RPM + concurrency semaphore + 429 backoff (Retry-After); 429 retried, exhausted/other fail loud; every model call routes through `_governed` | `DASHSCOPE_GOVERN` (on) |
| **F2** | `KnowledgeGraph._lock` around the contradiction closure — concurrent conflicting facts leave exactly one active edge | No |

F0 is the one wave that can regress the working system (it is not flag-gated); it was deadlock-reviewed (grep confirmed **no model call under any lock**) and proven by a stress test that caught a real `kth out of bounds` race mid-build.

## Speed waves (flag-off; semantics preserved)

| Wave | What | Flags |
|---|---|---|
| **S1** | `nli_batch` (N NLI calls → 1, fail-loud parse, missing→neutral); deferred re-embed (idle/sleep drain, FSRS+verified-helpful stay sync); short-circuit verify | `BATCH_NLI`, `DEFER_REEMBED`, `FAST_VERIFY` |
| **S2** | `ingest_many` (batched bulk); `rebuild_index_from_store` (index is a derived cache → crash-recoverable); debounced save; windowed tag-and-capture | `INDEX_SAVE_DEBOUNCE` |
| **S3** | BM25 backfill moved out of `_run_bm25` → parallel channels are read-only (fixes the `PARALLEL_CHANNELS` race) | `PARALLEL_CHANNELS` |
| **S4** | persistent embedding cache keyed by `(model, dim, sha256(text))` — warm hit needs no key/no call; a model/dim change misses | `EMBED_CACHE` (on) |
| **S5** | speculative cascade (cheap tier first, escalate on grounding miss); difficulty-adaptive depth | `SPECULATIVE_CASCADE`, `DIFFICULTY_ADAPTIVE_DEPTH` |

The grounding NLI gate is never removed — only batched, deferred, or short-circuited. Streaming SSE is the one S4 piece left for later (large API change, no offline-measurable contract).

## Competitive moat (C-track)

| Wave | What |
|---|---|
| **C2** | `value_as_of(entity, relation, as_of)` — deterministic time-travel ('where did Alice work on date X'); `fact_history` — current-vs-historical superseded chain (retained, never deleted); emits `SUPERSEDED`. MCP/API surfaced. |
| **C1** | `integrity_report()` — fabrication / abstention / verified rates + conflict load, counted from BrainEvents (never fabricated); emits `INTEGRITY_CHECKED`. |
| C3/C4 | event-chain + co-activation channels and calibrated verified abstention already shipped in the prior plan; this wave surfaces them and adds the integrity rollup. |

## Integration spine (one brain, not islands)

New `BrainEventType`s — `REEMBED_DEFERRED`, `SUPERSEDED`, `RATE_LIMITED`, `CACHE_HIT`, `INTEGRITY_CHECKED` — so every wave emits to the one stream. Deferred work (re-embed) runs on the `LifecycleController` idle/sleep loop. The **anti-island test** (`test_brain_connectivity.py`) asserts each feature emits its event, its deferred work is reachable from a kernel loop, its model calls go through the governor, and its writes go through the lock.

## Invariants held (and tested)

- **Deadlock-free:** no model call under any write/graph lock (verified by grep + review); lock order is always engine → graph.
- **Baseline byte-identical** with every speed flag off; F0/F2/S2/S3/S4 preserve the single-threaded path (full suite green).
- **Concurrency-safe:** 50-way ingest+search+save stress test → zero corruption, index reloads; per-request trace thread-local; concurrent conflicting facts → one active edge.
- **Governor never fabricates:** 429 retried/bounded; exhausted-403 and other errors fail loud.
- **No stale/fake output:** embed cache keyed on (model, dim) can't return a stale-dim vector; nli_batch missing→neutral (not grounded); rebuild re-embeds for real.
- **Age-independence preserved:** no new wave puts age/FSRS into the ranking score (deferred re-embed refreshes content only; cascade swaps the reader; adaptive depth scales count by query features, not memory age).

## Tests

+~30 offline tests this plan: `test_concurrency`, `test_rate_governor`, `test_atomic_graph`, `test_read_path`, `test_write_path`, `test_parallel_channels`, `test_embed_cache`, `test_intelligence`, `test_competitive`, plus the anti-island assertion. **345 offline tests pass.** Only the 5 pre-existing live-model tests fail (HTTP 403 quota); the key-gated smoke test skips cleanly.

## Not measured (the honest gap)

No latency/throughput/benchmark number is claimed. The win the plan promises — "beats Mem0 on knowledge-update / temporal / abstention", "p95 down 80%", "5x ingest" — requires a funded key to run `bench/` head-to-head against a reproduced Mem0 baseline. The machinery is built and invariant-checked; the measured run is the next step.
