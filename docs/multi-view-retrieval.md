# Multi-view retrieval + provenance flagships (Memory Agent Upgrade)

This implements the highest-leverage slice of the Memory Agent Upgrade plan: wiring three
**dormant retrieval signals into the live fused ranking**, plus two read-only flagship functions.
Everything is behind config flags defaulting **off**, so the neutral benchmark path is unchanged.

## The honest frame (read first)

This is the fifth consecutive "make it destroy the benchmarks" task, and the honest state is the
same after each: the machinery is built and correct in isolation, the baseline is byte-for-byte
unchanged, and **it has never been run against a benchmark**, because live DashScope is still
quota-blocked (HTTP 403). The plan's own promotion rule requires a measured dev win + McNemar +
latency budget, which the 403 blocks, so by the plan's own gate nothing defaults on. The single
thing standing between you and a real benchmark number is the quota toggle you own, not more code.

## What was wired (all flag-gated, offline-tested)

`retrieve()` previously fused dense + BM25 + graph-PPR + recency. Three dormant signals are now
fusable channels (they append to the weighted fusion only when their flag is on):

| Channel | Flag | What it adds |
|---|---|---|
| Structure-code | `STRUCT_CHANNEL` | entity / role / modality similarity in structure space (`index.search_struct`) |
| Event-overlap | `EVENT_RANKING` | promotes memories whose normalized event interval matches the query's time constraint |
| Derived-gist | `GIST_CHANNEL` | a gist that matches the query boosts its **raw member** memories (with provenance) |
| Graph-vocab seeding | `GRAPH_VOCAB_SEEDING` | seeds PPR from query tokens matched against in-scope store vocabulary |

**Age-independence is preserved** (the property this project disproves the violation of). The
structure code encodes only **cyclic** time (hour-of-day, day-of-week), never absolute age, and
the query structure code has no temporal dimension; the event channel ranks by overlap with the
**query's** time constraint, not the memory's age. A regression test confirms the structure
channel ranks two memories one year apart identically at a fixed cyclic coordinate, and the
formal index-level age-flatness proof is unaffected (it uses pure content search).

The **scoped-HNSW underfill** the plan flags as a bug is already fixed in this codebase: the
exact-subset fallback returns the full in-scope allowed set. A direct hnswlib-backend regression
test now locks that behavior (the numpy test backend cannot exercise the HNSW path).

## Flagship read-only functions

- **`prove_answer(answer)`** (`eidetic/proofs.py`, engine `prove`, MCP `recall(prove=True)`): a
  machine-readable proof tree. For each cited memory it returns the immutable content hash, span,
  timestamp, NLI grounding label, and any contradictions, plus whether provenance is complete.
  This is the citable-photographic-recall differentiator: every answer can show its work.
- **`memory_health_report(scope)`** (engine + MCP `health_report`): read-only self-diagnosis of a
  scope (memories, edges, derived gists, distinct entities, contradiction load, inferred /
  low-confidence / pruned edges, orphan records, replay debt, age spread). Every figure is counted
  from the store, never fabricated. Works without a key.

## Not done (scoped out of this session, per the plan's own multi-week cadence)

The plan also describes a closed-loop controller, a memory-physics layer, a large
revolutionary-function library, and a quantum-inspired track. Much of that is already present
behind flags from prior work (bandits, FTRL/EG fusion weights, fusion variants, anomaly scoring,
the EvolveMem guard, MMR, Rocchio, the daemon, Markov prefetch, memory typing, the MemMA repair
router). The genuinely new speculative surface (amplitude fusion, QUBO context selection, density
confidence, causal linking, tensor indexing) is deferred: it is "another flag-off variant," lower
leverage than wiring the dormant channels, and unmeasurable under the 403.

## Runbook (the one thing that turns this into a measured win)

```bash
# 0. Restore paid quota (disable "use free tier only" / add billing).  <- the only blocker
# 1. Baseline on the held-out TEST split.
python -m bench.run --systems eidetic --dataset both --runs 10 --split test
# 2. A/B a channel on the DEV split only (e.g. the event channel):
EVENT_RANKING=1 python -m bench.run --systems eidetic --split dev --out artifacts/dev/event_on
python -m bench.run --systems eidetic --split dev --out artifacts/dev/event_off
# 3. Guard the change (promote only on a significant dev win):
python -m bench.guard --champion artifacts/dev/event_off --challenger artifacts/dev/event_on
# 4. Re-prove age-independence with the channel on, then re-measure the winner on TEST.
```
Promote a channel only if it beats the champion on dev AND the age-flatness proof still holds.
