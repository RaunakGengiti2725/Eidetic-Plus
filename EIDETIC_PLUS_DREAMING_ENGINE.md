# EIDETIC-PLUS: The Dreaming Engine
### Token-Free, Offline, Continuous Memory Consolidation Beyond Human Cognition

> This is the forward roadmap, the part that takes Eidetic-Plus from "a very good memory system" to something a human brain categorically cannot be. The unifying idea: while the agent is idle, it keeps consolidating, reinforcing, and reorganizing everything it knows, continuously and in parallel, using zero LLM calls and therefore zero tokens. It is the machine analogue of dreaming, except it never has to wake up, never runs out of night, and never loses the original.
>
> Read it alongside the four companion files in the repo: `Eidetic-Plus_Master_Dossier.md`, `EIDETIC_PLUS_UPGRADE_SPEC.md`, `EIDETIC_PLUS_BENCHMARK_SPEC.md`, and `EIDETIC_PLUS_OPTIMIZATION_PLAYBOOK.md`. A condensed `/goal` prompt is in Section 11.
>
> Three honest framings up front, because they are what separate this from science fiction. First, "token-free" is the whole point and it is a real, hard line: human memory consolidation during sleep does not spend anything externally, it is the brain reorganizing what it already has, and the AI analogue is local computation over already-stored vectors and graphs (clustering, graph algorithms, matrix math) that makes no model calls at all. Everything in this file lives on that side of the line; where a capability secretly needs an LLM, it is marked. Second, the immediate priority is still the benchmark run, the Mem0 gate, and the flat-curve check; this dreaming engine is what to build after a baseline exists, and several of its pieces also move the benchmark, so they earn their place by the scoreboard, not by how impressive they sound. Third, the cardinal rule that keeps consolidation from corrupting the system: the immutable lossless store is sacred, every consolidation output is an additive, reversible, provenance-tagged, NLI-gated derived layer, and the system never merges or averages memories in the source of truth. Section 8 explains why that rule is non-negotiable.

---

## 0. THE ONE-SENTENCE VISION

While idle, Eidetic-Plus continuously replays, reinforces, and reorganizes its entire memory using only local computation (no tokens), deriving implied facts, forming schemas, maintaining every level of detail at once, pre-assembling likely answers, and healing contradictions, so that it knows more than it was ever told, forgets nothing it was told, and answers faster and cheaper over time rather than slower.

---

## 1. THE CORE IDEA: THE DREAMING ENGINE

Every existing memory system, including the strong ones, is fundamentally passive between queries. It stores what it is given and waits. When a query arrives, it retrieves. The time in between, which for a real agent is most of the time, is wasted.

The brain does the opposite. Its most important memory work happens offline, during sleep, when the hippocampus replays the day's experience and gradually reorganizes it into durable, generalized knowledge. That offline phase is where specific episodes become general understanding, where scattered facts get linked, where the noise gets pruned. It costs the brain energy but nothing external.

The dreaming engine makes Eidetic-Plus do this, and then exceeds the brain at it on five axes the brain is hard-limited on. The brain consolidates only during a few hours of sleep, serially, one stream at a time, slowly (systems consolidation takes weeks to years), and lossily (forming gist destroys verbatim detail and even manufactures false memories). A machine can consolidate continuously, in parallel, across the whole store, instantly, while keeping the lossless original forever. That combination is biologically impossible, and it is buildable today with known algorithms and no LLM calls.

---

## 2. WHAT WE JUST LEARNED RUNNING IT LIVE (THE CLAUDE CODE FINDINGS)

Before the roadmap, the lessons from the first real run, because they are not just bug fixes, they are design principles that the dreaming engine has to respect or it will repeat them at a larger scale.

### 2.1 The hang, and the four speed fixes
The first live run logged zero results because it never finished ingesting a single roughly 300-turn LoCoMo conversation. The cause was a compounding disaster: per-turn ingestion meant about 300 embeds, then consolidation ran per record with sequential `extract_edges` plus `score_importance` LLM calls, plus an O(N squared) full-graph `node_features` rebuild and an O(N squared) tag scan on every record. That is roughly 600 sequential LLM calls fighting quadratic graph work. It did not run slowly, it hung.

The four fixes, all correct and compounding: session-granularity ingestion (one memory per session, about 19 instead of about 300, roughly 15 times less work); parallel extraction in consolidation (a thread pool, since the LLM calls are I/O-bound and should never be sequential); graph features computed once instead of O(N squared) per record; and skipping importance scoring in bulk, because it only feeds FSRS priority and pruning and never touches the ranking path, so it has zero scoreboard impact. Result: never-finishing became 38 seconds for one full conversation.

### 2.2 The temporal bug, and the three correctness fixes
While testing, the first completed run answered the temporal question wrong, saying "yesterday" instead of "7 May 2023." This was caught and traced to three real bugs: event dates were coming through as None, fixed by anchoring each event to its session date, preferring an explicit absolute date; `select_for_query` was over-selecting 125 events on any-entity match, fixed by ranking by entity-match count so specific events beat the broad "Caroline" match; and session-granularity had made the extractor miss the specific fact, fixed by prefixing every raw chunk with its session date so the reader can resolve "yesterday" to the actual date. Result, judge-verified: "Caroline went to the LGBTQ support group on 2023-05-07 (yesterday relative to session date 2023-05-08)," correct=True, matching gold. The suite is at 37 passed, staged across 77 files.

### 2.3 The lessons that generalize to the dreaming engine
Three principles fall out of this, and they are exactly the principles the dreaming engine must obey:
1. **Quadratic offline work does not scale.** The O(N squared) graph rebuild is what killed the run. The dreaming engine touches the whole store, so every operation in it must be near-linear or incremental (Leiden is near-linear in edges, PPR is incremental, ANN is sub-linear), never naive all-pairs.
2. **Scope offline work to what actually affects retrieval.** The sharpest fix was realizing importance scoring never touches ranking, so it could be skipped in bulk. The dreaming engine must apply the same test to every derived structure: does this improve retrieval accuracy, latency, or cost? If not, it does not run on the hot path. Build only what the scoreboard can see.
3. **Dates and provenance must be anchored, always.** The temporal bug was a missing anchor. The dreaming engine's derived facts and schemas must carry their temporal anchor and their provenance, or as-of reasoning breaks at scale.

---

## 3. THE NEUROSCIENCE, TRANSLATED TO ALGORITHMS

Every step of biological consolidation has a token-free computational analogue. This is the translation table.

**Sharp-wave ripples and systems consolidation.** During slow-wave sleep the hippocampus fires sharp-wave ripples (80 to 150 Hz bursts in humans, CA3 to CA1, 50 to 100 ms), time-locked to cortical slow oscillations and spindles, which orchestrate a gradual transfer of memory from fast hippocampal storage to distributed cortical storage (the Complementary Learning Systems model, McClelland, McNaughton and O'Reilly 1995). Analogue: a background loop that re-touches stored embeddings and edges and moves information from the raw episodic store into a derived semantic and schema layer (your knowledge graph plus summaries). The hippocampus is your lossless store; the cortex is your derived schema layer.

**Synaptic homeostasis (SHY).** Waking potentiates synapses net-positive; slow-wave activity downscales synaptic strength globally back to a sustainable baseline, preserving relative differences while pruning the weakest (Tononi and Cirelli). Analogue: a periodic global renormalization of edge weights and salience scores, pruning the weakest edges from the retrieval index. This is a pure graph operation and it keeps the index small and fast, which (per the live-run lesson) is exactly what keeps retrieval from degrading at scale.

**Prioritized replay.** Replay is not uniform; it over-represents rewarded, novel, and surprising experience. Mattar and Daw (Nature Neuroscience 21:1609 to 1617, 2018) give the normative rule: access memories in order of utility = gain times need. Analogue: prioritize replay by surprise (embedding distance from existing clusters, or knowledge-graph prediction error) times need (Personalized PageRank, recency, query frequency). This is prioritized experience replay realized over a memory graph.

**Schema and gist extraction.** Repeated replay across overlapping episodes extracts regularities and transfers episodic memory toward semantic memory; sleep specifically promotes transitive inference and rule discovery. Critically, in humans this destroys verbatim detail. Analogue: community detection (Leiden) over the graph forms schemas, centroid or medoid summaries form gist, but the machine keeps the gist layer additive over a preserved verbatim store, escaping the human tradeoff entirely.

**The spacing effect.** Spaced reinforcement beats massed because each well-timed retrieval triggers a fresh consolidation round (Cepeda et al. 2006 meta-analysis, 300-plus experiments). FSRS formalizes this with a Difficulty-Stability-Retrievability model and cuts required reviews roughly 15 to 20 percent versus SM-2. Analogue: you already run FSRS; the dreaming engine extends it from a review scheduler into a continuous reinforcement loop.

---

## 4. THE BEYOND-HUMAN CAPABILITIES (THE MENU)

Each is something a human brain cannot do, paired with why it is possible for a machine.

**Decoupled lossless recall and forgetting control.** Humans cannot separate storage from retrievability; consolidation overwrites. Eidetic-Plus already separates an immutable store (perfect retention) from FSRS salience (tunable forgetting). The inhuman move: tune retrieval salience arbitrarily and reversibly while never losing the source. "Forget" by lowering retrieval priority, not by deletion. Perfect memory and perfect forgetting at once.

**Continuous parallel consolidation.** The brain consolidates only during limited sleep, serially, with transfer taking weeks to years. A machine can replay and reinforce continuously, in parallel, across the entire store, forever, bounded only by compute. There is no overnight lag; schemas, inferences, and salience are always fresh.

**Transitive and associative inference at scale.** A relates to B, B relates to C, therefore A relates to C, computed across millions of nodes instantly and exactly. Humans do this slowly and unreliably and at tiny scale. Combined with rule mining, the system pre-derives a large closure of implied facts it was never explicitly told.

**Simultaneous multi-resolution memory.** Humans trade detail for gist; you cannot hold both. A machine can keep the lossless episode and every level of summary at once, and answer at any resolution. RAPTOR (Sarthi et al., Stanford, ICLR 2024) demonstrates the retrieval structure; coupling it with a strong reader improved the best QuALITY result by 20 absolute points to 82.6 percent.

**Predictive and counterfactual memory.** Pre-compute likely future queries and pre-assemble their answers during idle time; pre-link memories likely to be co-retrieved; pre-compute what-if recombinations as candidate edges. Sleep-time compute (Lin et al., arXiv:2504.13171, 2025) proves the principle pays: about 5 times less test-time compute for the same accuracy, 2.5 times lower cost per query, and up to 13 to 18 percent higher accuracy by scaling idle-time compute.

**Cross-corpus pattern discovery.** Find patterns, contradictions, and connections across the entire history that no human could hold in mind at once: communities surface themes, rule mining surfaces regularities, embedding outliers surface anomalies, and the NLI verifier surfaces contradictions.

**Self-healing, self-organizing memory.** Continuous deduplication, re-clustering, contradiction resolution, and weak-link strengthening, all token-free except where NLI is needed for verification, and all strictly additive over the lossless record.

---

## 5. THE CONCRETE TOKEN-FREE MECHANISMS TO BUILD

**The continuous replay scheduler (the substrate).** Maintain a priority queue keyed by replay utility = surprise times need times (1 minus retrievability), where surprise = knowledge-graph embedding prediction error or embedding distance to the nearest cluster, need = Personalized PageRank plus query frequency plus recency, and retrievability = the FSRS R(t,s) value. Each idle cycle: pop the top-k, replay them (recompute local embeddings and centroids, refresh edge weights, re-run link prediction in their neighborhood), update FSRS stability, then apply a global SHY-style downscaling pass that renormalizes edge weights and prunes the weakest. This is prioritized experience replay (Schaul et al., ICLR 2016, which beat uniform replay on 41 of 49 Atari games) fused with the Mattar-Daw gain-times-need utility, realized over a memory graph. Everything here is local math; zero tokens.

**Offline link prediction, rule mining, and schema induction (the biggest capability jump).** Periodically train a knowledge-graph embedding model (TransE is cheapest: h plus r approximately equals t; RotatE is stronger: relations as rotations in complex space) by gradient descent, score candidate triples, and propose top-scoring new edges. Run a symbolic rule miner (AnyBURL, bottom-up from random walks, or AMIE) to mine Horn rules like born(X,A) and capital(A,Y) implies citizen(X,Y), and apply high-confidence rules to infer new facts. Run Leiden community detection to form schemas, each becoming a node with a centroid embedding. Gate every proposed edge and fact through the existing NLI verifier and a confidence threshold before it becomes retrievable, and keep all inferred items in a separate "inferred" namespace, never co-mingled with observed facts. All token-free.

**Multi-resolution retrieval.** Build a RAPTOR-style tree by recursive soft clustering (Gaussian Mixture Models over dimensionality-reduced embeddings, or Leiden) with a centroid summary at each level, and let retrieval hit any level. The summary nodes can be centroids; LLM-written summary sentences are an optional enrichment, not a requirement for retrieval to work.

**Predictive pre-fetch.** Cluster the query log; for each predicted query cluster, pre-run retrieval and pre-assemble the answer context (top-k memories plus the relevant subgraph), stored keyed by the cluster centroid. At query time, match the incoming query embedding to the nearest pre-assembled context for near-zero assembly latency and zero tokens.

---

## 6. WHICH OF THESE MOVE THE BENCHMARK (THE BRIDGE)

This is the honest connection between the visionary layer and the hackathon, and it is why this is not a distraction. Most of these capabilities are token-free offline computation that also improves the exact benchmark categories you are trying to win, which means they help accuracy while being free on the cost axis.

- **Offline link prediction plus rule mining plus schemas** improves multi-hop and multi-session retrieval, the two categories where you currently trail, by pre-deriving the connections that multi-hop questions need. Zep's reported 15 to 18 percent LongMemEval gains came largely from graph structure; this is more graph structure, computed for free.
- **Multi-resolution retrieval** improves complex and aggregative question accuracy, the kind of question that needs both detail and summary at once.
- **Predictive pre-fetch plus the replay scheduler's index pruning** improves latency and cost, the efficiency axis you are already winning, and pushes it further.
- **Self-healing contradiction detection** targets the one genuinely unsolved category (contradiction resolution), where even parity is a win.

So the build order in Section 10 is sequenced so the benchmark-relevant pieces come first and are measured against the scoreboard. They earn their place by moving a number, not by being impressive.

---

## 7. THE BUILDABILITY RATINGS (HONEST)

**Clearly buildable now, known algorithms, zero LLM:** the continuous replay scheduler; knowledge-graph link prediction plus rule mining plus Leiden schemas; transitive inference; the multi-resolution summary tree (retrieval part); cross-corpus pattern and contradiction discovery; decoupled lossless-recall-plus-tunable-forgetting; SHY-style edge renormalization and pruning.

**Plausible but research-grade:** predictive pre-fetch quality depends on how predictable your queries are; counterfactual recombination as genuinely useful (not just generated) edges; self-healing contradiction resolution (detecting a contradiction is easy, deciding which side wins is hard and may need NLI plus temporal logic).

**Genuinely speculative, not yet reliable token-free:** high-precision semantic summarization without an LLM (centroids are not sentences); autonomous schema naming and interpretation; deriving genuinely novel abstract concepts as opposed to clusters. Where you want human-readable knowledge out of these, an LLM pass is still the honest tool; design so the token-free layer delivers value alone and the LLM is optional enrichment, never a dependency.

---

## 8. WHEN CONSOLIDATION HURTS, AND THE GUARDS (READ THIS TWICE)

The most important section. Offline consolidation can actively corrupt a memory system, and the failure modes are well-evidenced.

**The vector-averaging fallacy (the cardinal danger).** "The Geometry of Forgetting" (Barman et al., arXiv:2604.06222, 2026) shows that geometrically merging nearby embeddings to save space measurably increases interference: in a 100-category protocol, centroid merging achieved 62.5 percent compression but drove backward interference from minus 0.100 to minus 0.394. In their words, centroid merging erases the fine angular structure that separates semantically adjacent memories, collapsing distinct traces into a blurred centroid that confuses retrieval, and any vector database that deduplicates or compresses via centroid merging will predictably degrade retrieval fidelity. **Guard:** never merge or average in the lossless store; keep consolidation additive in a derived layer only; deduplicate only at the retrieval-index level with conservative thresholds while preserving every original.

**Hallucinated links from knowledge-graph embedding.** TransE and RotatE will happily score false triples highly. **Guard:** NLI-verify plus a confidence threshold on every machine-proposed edge; keep inferred edges in a separate namespace, never presented as observed fact.

**Runaway reinforcement (rich-get-richer).** PPR plus replay can over-amplify already-salient nodes until they dominate everything. **Guard:** the SHY-style global downscaling each cycle, plus a cap on per-node salience growth.

**False-memory manufacture is intrinsic, not a bug.** The same geometry that yields useful gist yields DRM-style false memories; the paper reproduced a 0.583 false-alarm rate (versus human about 0.55) on unmodified embeddings with zero tuning, because any system that organizes by meaning places related concepts nearby and any threshold confuses items within those regions. **Guard:** retain verbatim provenance, flag gist nodes as derived, and require multiple independent episodes before promoting a pattern to a schema.

**Power-law forgetting from interference, not decay.** The same work found that interference among competing memories, not time, drives power-law retrieval failure (exponent about 0.46 with 10,000 distractors versus about 0.009 with decay alone), and that production embeddings concentrate variance in only about 16 effective dimensions, deep in the interference-vulnerable regime. **Guard:** monitor recall versus corpus size, and use hybrid lexical and metadata filters to attenuate interference as the store grows.

**Corrupting the lossless record is the one unforgivable sin.** Immutability is non-negotiable. Every consolidation write goes to a derived, content-addressed, reversible store. This single discipline prevents the most damaging failure mode.

---

## 9. COMPUTE AND STORAGE COST (TOKEN-FREE IS NOT FREE)

Continuous consolidation uses CPU, GPU, and memory even though it uses no tokens. The heaviest job is knowledge-graph embedding training (periodic gradient descent over the triple set), so schedule it as a slow-cadence batch job, not per cycle. Leiden is near-linear in edges; PPR and ANN are cheap and incremental; rule mining is tunable by rule length and time budget; node2vec second-order walks are memory-heavy unless you use on-the-fly variants. The derived layer (summaries, inferred edges, pre-assembled contexts) grows storage, so bound it with the same FSRS salience eviction you use for episodes and cap the inferred namespace. Practical bounding: run heavy jobs (embedding retrain, full rule mining) on a slow cadence, run cheap jobs (PPR refresh, FSRS update, incremental dedup) continuously, and enforce a compute budget per idle window. And per the live-run lesson, every operation must be near-linear or incremental, never naive all-pairs, or it will hang exactly the way the first run did.

---

## 10. THE BRAINSTORMED UPGRADE ROADMAP (PRIORITIZED)

Sequenced so benchmark-relevant, token-free, clearly-buildable pieces come first, each measured against the scoreboard. Build only after a benchmark baseline exists.

**Phase 1, the substrate: the continuous replay scheduler.** Extend FSRS from a scheduler into a continuous reinforcement loop with the surprise-times-need-times-(1-minus-retrievability) priority and the SHY-style downscaling and pruning pass. This is the foundation everything rides on, and it reuses FSRS, PPR, and HNSW you already have. Measure: retrieval recall over time and retention per unit compute. Guard: if pruning degrades recall, reduce its aggressiveness.

**Phase 2, the biggest capability jump: offline link prediction plus rule mining plus schemas.** TransE (cheapest) plus AnyBURL, with Leiden schemas, all NLI-gated into a separate inferred namespace. Measure: multi-hop and multi-session retrieval Hits@k against the scoreboard. Guard: if inferred-edge precision (NLI pass rate) drops below about 90 percent, raise the threshold or shorten rule length.

**Phase 3, complex-QA: multi-resolution retrieval.** RAPTOR-style recursive clustering with centroid summaries, retrieval at any level, LLM summaries optional. Measure: complex and aggregative QA accuracy.

**Phase 4, the efficiency win: predictive pre-fetch.** Query-cluster statistics plus pre-assembled context cache. Measure: P95 latency and cost per query. Guard: deploy only on query clusters where the hit rate clears the latency-saving threshold.

The two capabilities that most differentiate Eidetic-Plus as categorically beyond human are continuous parallel consolidation (Phase 1) and simultaneous multi-resolution lossless memory (Phase 3). Neither is possible for a biological brain; both are buildable now.

---

## 11. THE CONDENSED `/goal` PROMPT (UNDER 4000 CHARACTERS)

Build a benchmark baseline first; then this builds the dreaming engine as a token-free, additive, measured layer. Paste into `/goal`.

```text
GOAL: Implement EIDETIC_PLUS_DREAMING_ENGINE.md (+refs): a token-free offline continuous
consolidation layer, beyond-human memory that also moves the benchmark. ZERO LLM calls in
consolidation (no tokens): only local math over stored embeddings + the graph. Build, wire, unit-test
offline; don't stop until all below works and the suite is green. Assumes a benchmark baseline exists;
build a piece only if it moves accuracy/latency/cost.

CARDINAL RULES (a violation corrupts the system):
- The lossless store is SACRED: never merge, average, or centroid-collapse memories in it (the vector-
averaging fallacy measurably INCREASES interference). Consolidation is an ADDITIVE, reversible,
provenance-tagged, content-addressed DERIVED layer only.
- Every machine-inferred edge/fact is NLI-gated + confidence-thresholded, in a SEPARATE "inferred"
namespace, never presented as observed fact.
- Every op is near-linear or incremental, NEVER naive all-pairs O(N^2) (an O(N^2) rebuild already hung a
live run). Leiden near-linear; PPR/ANN incremental.
- No mocks, fail loud on a missing key, never fabricate a score. Heavy jobs slow-cadence, cheap jobs
continuous, compute budget per idle window. Ask before any non-standard dep.

BUILD (benchmark-relevant first, each measured + guarded):
1) CONTINUOUS REPLAY SCHEDULER (substrate): extend FSRS from scheduler to a continuous reinforcement
loop. Priority = surprise * need * (1 - retrievability): surprise = KG-embedding prediction error or
distance to nearest cluster; need = PPR + query-freq + recency; retrievability = FSRS R(t,s). Each cycle: pop top-k, replay (recompute local centroids, refresh edge weights, local link prediction),
update FSRS stability, then a global SHY-style downscaling pass renormalizing edge weights + pruning the
weakest from the INDEX (never the store). Cap per-node salience growth. Measure recall over time; soften
pruning if it hurts recall.
2) OFFLINE LINK PREDICTION + RULE MINING + SCHEMAS (biggest jump; targets multi-hop/multi-session):
train a KG embedding (TransE) slow-cadence; score triples; propose top edges. Mine Horn rules (AnyBURL)
+ apply high-confidence ones. Leiden communities -> schema nodes (centroids). NLI-gate every proposed
edge/fact into the inferred namespace; cache for instant retrieval. Measure multi-hop/multi-session
Hits@k vs scoreboard; NLI pass rate <~90% -> raise threshold / shorten rules.
3) MULTI-RESOLUTION RETRIEVAL (complex-QA): RAPTOR-style recursive soft clustering (GMM over reduced
embeddings, or Leiden) with a centroid summary per level; retrieval hits any level; keep lossless
episode + every gist level at once. LLM summaries OPTIONAL (retrieval works on centroids). Measure
complex/aggregative QA accuracy.
4) PREDICTIVE PRE-FETCH (efficiency): cluster the query log; per cluster pre-assemble the answer context
(top-k + subgraph) keyed by centroid; at query time match incoming embedding to nearest pre-assembled
context (zero tokens to assemble). Measure P95 latency + cost/query; deploy only where cluster hit-rate
clears the threshold.

PARAMETERIZE + sweep (config defaults, don't hardcode): replay weights, prune aggressiveness + salience
cap, KG-embedding cadence + model, inferred-edge NLI/confidence threshold, rule length + confidence,
cluster counts, pre-fetch threshold. Offline unit tests (no key): additive-only (store never mutated),
inferred-namespace separation, near-linear complexity guards, NLI-gating, multi-resolution retrieval,
pre-fetch cache hit. Update README + docs (token-free guarantee, immutability rule, inferred namespace,
measured-vs-scoreboard framing).

DONE = all four wired as a token-free additive layer; lossless store provably never mutated/merged;
inferred items NLI-gated in a separate namespace; no O(N^2) anywhere; each measured vs scoreboard; suite green; no mocks; MIT kept. Honest doc note: schema naming / contradiction RESOLUTION may
want an optional LLM pass; token-free layer must deliver value without it.
```

---

## 12. SOURCES

- Sleep and consolidation: Complementary Learning Systems (McClelland, McNaughton, O'Reilly 1995); Synaptic Homeostasis Hypothesis (Tononi, Cirelli); sharp-wave ripple and systems-consolidation literature.
- Replay prioritization: Mattar and Daw, "Prioritized memory access," Nature Neuroscience 21:1609 to 1617 (2018).
- Spacing and FSRS: Cepeda et al. (2006) meta-analysis; the FSRS Difficulty-Stability-Retrievability model.
- Experience replay: Prioritized Experience Replay (Schaul, Quan, Antonoglou, Silver, ICLR 2016); Deep Generative Replay (Shin et al. 2017); brain-inspired replay (van de Ven et al. 2020); Dark Experience Replay (Buzzega et al., NeurIPS 2020).
- Sleep-time / idle compute: Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez, "Sleep-time Compute" (arXiv:2504.13171, 2025).
- Agent memory: Zep/Graphiti (Rasmussen et al., arXiv:2501.13956, 2025); Mem0/Mem0g (Chhikara et al., ECAI 2025); A-MEM (Xu et al. 2025).
- Multi-resolution: RAPTOR (Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning, Stanford, ICLR 2024).
- KG embedding and rule mining: TransE, RotatE, DistMult, ComplEx; AMIE/AMIE+ and AnyBURL.
- Community detection: Leiden (Traag et al.) and Louvain; node2vec/DeepWalk and Fast-node2vec.
- Failure modes: "The Geometry of Forgetting" (Barman et al., arXiv:2604.06222, 2026), the vector-averaging fallacy, DRM false-memory reproduction, and interference-driven power-law forgetting.

> The brain dreams to consolidate, but it only gets one night at a time, it does it serially, and it forgets the details to keep the gist. Eidetic-Plus dreams continuously, in parallel, across everything, forever, and keeps the details and the gist and the source. That is the categorical difference. Build it additively, gate it with NLI, never touch the lossless record, and make every piece prove itself on the scoreboard. The benchmark comes first; this is how you win the next one.
