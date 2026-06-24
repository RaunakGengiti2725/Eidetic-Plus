# EIDETIC-PLUS: Optimization Playbook
### Topping Every LongMemEval and LoCoMo Category While Winning on Cost and Latency

> This is the optimization and gap-closing pass on the system you have already built. It is not a redesign. It is the prioritized list of the exact upgrades that move each benchmark category, with the technique, the expected direction, the failure mode to kill, and the source. Read it alongside the three companion files in the repo: `Eidetic-Plus_Master_Dossier.md` (the research foundation), `EIDETIC_PLUS_UPGRADE_SPEC.md` (the universal-plugin and engine upgrades), and `EIDETIC_PLUS_BENCHMARK_SPEC.md` (the neutral harness and per-category mechanisms). A condensed `/goal` prompt that triggers this playbook is in Section 11.
>
> Two honest framings that decide whether this works, stated up front. First, two of the highest-leverage moves here (the conformal abstention threshold and the query-adaptive fusion weights) cannot be tuned without a real run, because the correct values are computed from real scored questions. So the rule is: build all the machinery now, leave the tuned values as config parameters with safe defaults, and calibrate them on a cheap subset the moment a key is added. Second, the target numbers in this file (for example "temporal to 90 plus") are the published starting points of other systems, not yours. Treat them as direction, not as an assumed baseline. Your real baseline comes from your first run.
>
> Discipline carried from the other files: no mocks anywhere (every model call is a real DashScope call, fail loud on a missing key); never delete a raw record; never fabricate a score; and the fairness rule you already fixed stays intact, the neutral harness uses one fixed reader across all three systems, and NLI, abstention, and the cascade are product features reported in the cost and latency tables, not a neutral-accuracy answerer advantage.

---

## 0. THE ONE-SENTENCE GOAL

Lead every accuracy category on LongMemEval and LoCoMo by adding a structured event calendar, first-class preferences, a calibrated abstention gate, and tuned hybrid retrieval, while keeping roughly 80 percent of queries on the cheapest model so the cost and latency win is locked in, and proving all of it on the neutral harness rather than claiming it.

---

## 1. THE HONEST REALITY: WHAT IS WINNABLE AND WHAT IS NOT

Leadership in this field is fragmented. No single public system wins every sub-category at once. Different systems own different categories, which is good news because it means there is a winnable gap in every category and no monolithic champion. Your bi-temporal architecture already structurally leads the two categories where Zep beats Mem0 (knowledge-update and temporal). The work is to close the field-wide weak spots (multi-session aggregation and single-session-preference) and to calibrate the pieces you already have.

One category is genuinely unsolved by anyone: cross-session contradiction resolution at scale. The best public BEAM-1M score for that ability is 0.357, and the benchmark authors state that maintaining a globally consistent state remains an open problem. Be brutally honest about this. Aim for parity or a slight lead there, never dominance, and present it as the frontier. Winning the two standard benchmarks outright, cheaper and faster, with two properties no competitor has (flat recall-versus-age and verified recall with citable sources), is domination. Overclaiming the one unsolved box is the single move that makes a sharp judge doubt the rest.

---

## 2. THE CURRENT SOTA LANDSCAPE (PER-CATEGORY, LongMemEval-S, 500 questions)

Category counts: single-session-user 70, single-session-assistant 56, single-session-preference 30, knowledge-update 78, temporal-reasoning 133, multi-session 133.

| System | Overall | SS-User | SS-Asst | SS-Pref | Know-Update | Temporal | Multi-Session |
|---|---|---|---|---|---|---|---|
| Chronos (High) | 95.6 | 98.57 | 100 | 100 | 96.15 (Low) | 90.23 | 88.72 |
| Mastra Observational | 95.4 (task-avg) | n/a | n/a | 100 | 96.2 | 87.2 | 95.5 |
| ByteRover 2.1.5 (Gemini-3-Flash) | 92.8 | 98.6 | 98.2 | 96.7 | 98.7 | 91.7 | 84.2 |
| Hindsight (Gemini-3 Pro) | 91.4 | ~97 | ~96 | 80.0 | 94.9 | 91.0 | 87.2 |
| Honcho | 90.4 | ~95 | ~95 | 90.0 | 94.9 | 88.7 | 85.0 |
| Engram / Eidetic baseline (lean) | 83.6 | n/a | n/a | 73.3 | 87.5 | 81.1 | 79.3 |
| Zep / Graphiti (GPT-4o) | 71.2 | 74.11 | n/a | n/a | n/a | 79.79 | n/a |
| Mem0 | 66.9 | 67.13 | n/a | n/a | n/a | 55.51 | n/a |

Interpretation: your bi-temporal model already gives Zep-beating knowledge-update (87.5) and temporal (81.1), but trails the 90-plus leaders by roughly 12 points overall. The gap is almost entirely in five categories the leaders push into the mid-90s, and the single biggest structural lever is the Chronos dual-calendar design. Note that these numbers are harness-dependent: independent GPT-4o evaluations place Mem0 and Zep lower than their self-reports, so treat cross-source numbers as targets, not guarantees, and prioritize your own neutral re-runs.

---

## 3. PER-CATEGORY: THE EXACT TECHNIQUE TO TOP EACH, AND THE FAILURE MODE TO KILL

### 3.1 Single-session-preference (SOTA 100 Chronos/Mastra, ByteRover 96.7, Hindsight only 80, Engram 73.3)
This is hard field-wide. The winning mechanism is to treat preferences as first-class typed memories extracted at write time, not retrieved as an afterthought. ByteRover's curation step captures 29 of 30; Hindsight's general retrieval gets 24 of 30. Two reinforcing techniques: a persona or profile pass that derives preference insights after each turn and accumulates a user representation (Honcho's dialectic pass), and rubric-aware answer generation, because the official judge for this category uses a lenient rubric (correct as long as the response recalls and applies the user's preference). Prompt the shared reader to surface and apply the stored preference explicitly, without giving Eidetic a different reader than the baselines. Failure mode to eliminate: preferences buried inside generic fact chunks and never surfaced in the top-k.

### 3.2 Temporal-reasoning and knowledge-update (SOTA Chronos 90.23 temporal / 96.15 KU, ByteRover 98.7 KU)
What gets the leaders past 90 is the Chronos structured event calendar: extract subject-verb-object tuples, resolve every fuzzy time expression ("recently," "the week after," "last May") into precise ISO-8601 datetime ranges using forward and backward relative inference, attach two to four paraphrase aliases per event, and index events separately from raw turns. This converts temporal reasoning from string interpretation to structured filtering, and the event calendar alone delivered a reported plus 58.9 percent lift over a turn-only baseline. You already have bi-temporal valid_at/invalid_at plus invalidate-not-delete, which is exactly what knowledge-update needs (most-recent-wins via invalidation, 87.5). The missing pieces are multi-resolution relative-date normalization at write time and an explicit as-of timeline filter at query time. Failure modes: coarse, year-bucketed time encoding; treating relative dates as ungrounded strings; off-by-one errors (the official judge has a temporal off-by-one tolerance, so exploit it rather than fight it).

### 3.3 Multi-session reasoning and LoCoMo multi-hop (SOTA Mastra 95.5 MS, ByteRover 93.3 LoCoMo multi-hop, Hindsight leads MS at 87.2)
This is the hardest category for nearly everyone and your joint weakest (79.3). The strongest cheap technique is single-pass Personalized PageRank over the knowledge graph, seeded from query-linked phrase nodes, letting activation spread to retrieve passages that share no query words but are linked through intermediate entities. Per HippoRAG-2 (Gutierrez et al., "From RAG to Memory," arXiv:2502.14802, ICML 2025), this lifted MuSiQue multi-hop F1 from 44.8 to 51.9 and 2Wiki Recall@5 from 76.5 to 90.4, a roughly 7-point gain in associative memory over the strongest embedding baseline, with no iterative LLM calls. Pair it with lightweight query decomposition for aggregation questions ("how many times did I exercise in May" becomes an entity plus temporal-range filter over the event calendar). Avoid expensive iterative ReAct loops; the cheaper path (single-pass PPR plus structured filtering) is what keeps a Flash-tier model competitive. Failure mode: relying on dense similarity alone for cross-session counting and aggregation.

### 3.4 LoCoMo open-domain (universal weak spot, roughly 67 to 77; Hindsight 90.96, ByteRover 85.9)
This requires fusing retrieved memory with the model's parametric world knowledge. The winning approach is gated generation: detect whether the question is answerable from memory alone or needs world knowledge, and instruct the answerer to combine them rather than refuse. Hindsight's typed "world facts" network separates world knowledge from episodic memory. Failure mode: over-grounding (refusing to use parametric knowledge) or under-grounding (ignoring retrieved memory).

### 3.5 Abstention (LongMemEval _abs; BEAM-1M abstention only 0.525; Hindsight 90 on BEAM)
The SOTA is calibrated, threshold-based abstention, not asking the LLM to self-judge. Two components: a retrieval-coverage signal (if the top reranked score or the NLI-entailment score falls below a calibrated threshold, abstain), and conformal prediction for distribution-free coverage guarantees. A representative RAG operating point is a threshold giving 95 percent answer precision at a roughly 30 percent abstention rate. HALT-RAG (arXiv:2509.07475) demonstrates calibrated NLI ensembles with task-specific thresholds: a QA threshold of 0.395 yields precision 0.9838, recall 0.9735, F1 0.9786 at very low calibration error. You already have an NLI abstention gate; the upgrade is to calibrate its threshold on a held-out LongMemEval subset rather than use a fixed cutoff, trading a little coverage for a large precision gain. Failure mode: over-abstaining on answerable questions, which silently tanks the other five categories.

### 3.6 Single-hop and information-extraction (SOTA ByteRover 97.5 LoCoMo single-hop, 98.6 SS-user)
Pure recall: maximize hybrid-retrieval recall so no fact is missed. Levers: a high-recall first stage (efSearch tuned up), the BM25 lexical channel weighted up for names, dates, and IDs, and the hybrid facts-plus-raw-chunks context (facts alone lose the exact string some questions need verbatim). Failure mode: over-aggressive fact extraction that drops the exact string a single-hop question requires.

### 3.7 Contradiction-resolution (the genuinely unsolved category)
BEAM-1M contradiction_resolution is 0.357, the worst of all ten abilities; BEAM-10M is 0.325. HydraDB reports 66 versus Hindsight's 59 on BEAM-1M, still a failing grade. You cannot cleanly top this with a known technique. The best available levers are active conflict detection at write time (your invalidate-not-delete plus supersedes chain is a strong start), entity resolution before data reaches the model (canonicalizing "Acme Corp" and "ACME CORP"), and belief-revision passes during consolidation. Frame this as the one category where Eidetic-Plus aims for parity or a slight lead, not dominance.

---

## 4. RETRIEVAL-QUALITY OPTIMIZATION (THE CORE OF ACCURACY)

**Reciprocal Rank Fusion tuning.** k=60 is the well-validated default, from Cormack, Clarke, and Buttcher (SIGIR 2009) and now the default in OpenSearch, Elasticsearch, Azure AI Search, and others. Weighted RRF beats vanilla when channels have unequal trust; Engram's defaults are semantic 1.0, graph 0.8, lexical 0.6, recency 0.3, salience 0.25. Keep k=60, keep semantic-dominant weights, but make the weighting query-adaptive: boost the BM25 weight for name, date, and ID queries, boost the graph weight for multi-hop. Learned fusion only beats static RRF when you have tuning labels; for this deadline, static weighted RRF is the right call.

**Reranking.** The accuracy and latency sweet spot is rerank 50 candidates down to about 8, never more than 100 to 200 (recall degrades past about 100 due to high-scoring false positives). Cross-encoder rerank adds roughly 100 to 300ms for 30 to 50 candidates and typically delivers plus 10 to 25 percent, up to plus 33 to 40 percent on complex multi-hop queries. qwen3-rerank as the cross-encoder is appropriate. Engram ships rerank off by default, so turning it on at depth 50 is a likely gain you should A/B rather than assume. Keep listwise LLM rerankers off the hot path; they are more accurate but too slow and expensive.

**Hybrid dense-plus-sparse weighting.** Lexical matters most for names, dates, IDs, and exact-string single-hop facts; dense matters most for paraphrased and semantic queries. A dense-to-BM25 ratio around 1.0 to 0.6 is a sound start; make it query-adaptive.

**Chunking and context assembly.** The Engram finding is load-bearing: hybrid facts-plus-raw-chunks beats facts-only, because facts-only loses the verbatim detail some questions need. Engram's lean slice is up to 8 conflict-resolved facts plus the top-15 fused items plus 2 raw session chunks plus 28 session-level summaries (the reproduce flags are `--chunks 2 --topk 15 --extract-k 8 --summ-k 28`). Use turn or session granularity, not arbitrary token windows. Apply lost-in-the-middle mitigation by placing the highest-scored evidence at the edges of the context. Deduplicate aggressively. Caveat: the facts-only-versus-hybrid claim is an explicit design observation in the Engram paper, not a controlled ablation, so validate it yourself.

**Embedding optimization.** text-embedding-v4 at 1024 dimensions is the recommended default; 2048 buys only about plus 1 point on CMTEB, dropping to 512 costs about 1.4 percent but halves storage and doubles speed, and int8 quantization is negligible loss while binary quantization significantly hurts. Keep 1024. Use task instructions (Qwen reports plus 1 to 5 percent) and query-versus-document asymmetry. Embed both facts and raw text, because the hybrid path needs both indexed. Skip ColBERT-style late interaction for this deadline; it is a possible accuracy boost but adds storage and latency.

**Query understanding.** HyDE helps when queries are short and conversational and documents are dense, which is the memory setting, but only for factual or public knowledge; it fails on private or fictional content where the hypothesized answer cannot match the target. The higher-ROI moves are query decomposition for aggregation and multi-hop, and Chronos-style query-conditioned extraction (a meta-prompt parses the question for entities, temporal constraints, and the operation). Skip step-back prompting and heavy multi-call rewriting; they add cost without reliable benefit here. Caveat: decomposition can hurt when the model has a reasoning shortcut to the direct answer but lacks the intermediate facts.

---

## 5. COST AND LATENCY OPTIMIZATION (WIN THE EFFICIENCY AXIS DECISIVELY)

**Token reduction.** The Engram result is the headline: 9.6k versus 79k tokens (roughly 8 times fewer) while gaining accuracy. Layer LLMLingua-2 selectively on the raw-chunk portion of the context: per Pan et al. (ACL 2024), it reduces end-to-end latency by up to 2.9 times at 2-to-5-times compression; LongLLMLingua (arXiv:2310.06839) reports up to plus 17.1 percent over the original prompt with about 4 times fewer tokens and a 1.4-to-3.8-times speedup. Compress raw chunks only, never the structured facts.

**Latency engineering.** Run the four retrieval channels (dense, BM25, PPR, recency) in parallel; keep consolidation fully async off the hot path (write under 50ms, read under 100ms, retrieval sub-second; the roughly 60-second p50 end-to-end is dominated by the answerer's generation call, not by retrieval). For HNSW, efSearch=100 gives about 85 percent recall in about 1ms and efSearch=500 gives about 98 percent in about 5ms, so tune efSearch up to about 256 to 500 for the recall-sensitive single-hop and multi-session categories since retrieval is not your latency bottleneck; keep M=32. Semantic cache at cosine 0.9 is the standard operating point.

**Model routing.** This is where you win cost decisively. Per FrugalGPT (Chen, Zaharia, and Zou, arXiv:2305.05176), an LLM cascade can match the best individual model with up to 98 percent cost reduction, or improve accuracy by 4 percent at the same cost. Per RouteLLM (Ong et al., ICLR 2025), matrix factorization achieves 95 percent of GPT-4 performance using 26 percent GPT-4 calls (about 48 percent cheaper), and with judge-augmented data the GPT-4 calls halve to 14 percent of total (75 percent cheaper than random routing). ByteRover proves a Flash model tops the leaderboard (its Flash run at 92.8 beat its Pro run at 92.2). Target keeping roughly 80 percent of queries on qwen-flash, escalating to qwen-plus or qwen3-max only when the cheap-model confidence or verification check fails. Use the cheapest reliable difficulty classifier (a small fine-tuned scorer). Caution: a badly tuned escalation threshold either leaks errors or escalates too often and adds tail latency.

**Consolidation efficiency.** Batch fact extraction; consolidate on idle or scheduled triggers; do incremental graph updates. Critically, prune the INDEX, not the store, via FSRS salience: keep the immutable WORM store complete for provenance, but drop low-salience facts from the hot retrieval index so retrieval stays fast at scale. This directly fights the BEAM lesson that at 10M tokens retrieval degrades because similar content appears repeatedly.

---

## 6. THE NEWEST 2025-2026 SYSTEMS AND WHAT MAKES THEM WIN

- **Chronos (95.6, arXiv:2603.16862):** dual-calendar (event subject-verb-object plus ISO-8601 ranges, and raw turns), query-conditioned selective extraction (minimum sufficient abstraction to avoid context entropy), dynamic per-query prompting, dual-index hybrid retrieval. This is the single most copyable design for Eidetic-Plus.
- **ByteRover 2.1.5 (92.8 LongMemEval, 96.1 LoCoMo, 1.6s p50):** hierarchical Context Tree, first-class preference facts, Flash-tier models. Proof that architecture beats model budget.
- **Hindsight (91.4, arXiv:2512.12818):** four typed memory networks (world facts, experiences, opinions, observations) plus retain, recall, reflect, and 4-way parallel retrieval (semantic, BM25, graph, temporal) merged by RRF and a cross-encoder reranker, which is nearly identical to your read path. Leads multi-session at 87.2. With a 20B open model it reaches 83.6, plus 44.6 over full-context with the same model, proving architecture beats model size.
- **Mastra Observational Memory (95.4):** stores observations that beat the raw oracle data; 100 SSP, 95.5 multi-session.
- **Memory-R1 (arXiv:2508.19828):** GRPO reinforcement learning over ADD/UPDATE/DELETE/NOOP with only 152 training examples, plus 48 percent F1 and plus 69 percent BLEU-1 over Mem0 on LoCoMo with zero-shot transfer to LongMemEval. A possible stretch goal, but overkill for the July 9 deadline.
- **Engram (83.6, arXiv:2606.09900):** your baseline. Proves lean context beats full context (plus 10.4 points, McNemar p below 10 to the minus 6, gain CI plus 6.4 to plus 14.4) and that structured facts contribute essentially all the gain (engram_full 83.4 versus engram_lean 83.6, no difference, p=0.91, so the full history adds tokens not correctness). Published config: RRF k=60, a bge-small embedder in the published run, rerank off by default.

What separates the number-one system from the number-five system is replicable: structured, query-conditioned extraction with explicit temporal grounding (Chronos, ByteRover Context Tree), plus first-class preference handling, plus multi-strategy parallel retrieval with reranking (Hindsight). The number-one system does not use a bigger model; ByteRover at number one uses Flash. The differentiator is architecture: minimum sufficient abstraction, temporal-as-structure, and typed memories.

---

## 7. THE PARAMETERS TO TUNE (AND WHY CALIBRATION NEEDS A RUN)

Build all of this as machinery now, but leave these as config parameters with safe defaults, and tune them on a 50-to-100-question subset the moment a key is added. Hardcoding a tuned value before a run is guessing.

- **Abstention threshold (highest leverage):** a conformal threshold computed from a held-out subset to hit roughly 95 percent answer precision. There is no correct value to hardcode; it must be calibrated. Wrong value either over-abstains (tanks the other five categories) or under-abstains (fails _abs).
- **RRF channel weights:** query-adaptive (BM25 up for names and dates, graph up for multi-hop), with the base weights in config. k stays fixed at 60.
- **Rerank depth and on/off:** default depth 50 to about 8, with an on/off flag to A/B against off.
- **efSearch:** a config value around 256 to 500, since retrieval is not the latency bottleneck. M stays 32, embeddings stay 1024-dim.
- **Cascade routing threshold:** tuned to keep roughly 80 percent of queries on qwen-flash without accuracy loss.
- **Compression ratio:** the LLMLingua-2 ratio applied to raw chunks only.

Add a sweep command that grid-searches these on a small subset and reports the best config, ready to run once a key is present.

---

## 8. EVALUATION AND TUNING METHODOLOGY

- **Sweep priority order:** abstention threshold first (highest leverage), then rerank depth (50 versus 100), then RRF channel weights (query-adaptive versus static), then efSearch (256 versus 500), then the cascade routing threshold. RRF k=60 is safe to fix.
- **The judge:** qwen3-max-as-judge and GPT-4o-as-judge can differ by several points, so report both and use the official LongMemEval category-specific judge prompts (including the temporal off-by-one tolerance, the knowledge-update old-info tolerance, and the single-session-preference rubric-leniency prompt). Never let a system grade its own work; keep judge separate from answerer. Cross-evaluate with at least two judges and report variance over at least 10 runs.
- **Per-category error analysis:** read the per-question JSONL logs filtered by category; for each miss, classify it as a retrieval-miss (evidence not in the top-k, so fix efSearch, weights, or decomposition), a reasoning-miss (evidence present but the answer is wrong, so fix the prompt or temporal normalization), or an abstention-miss (wrongly abstained, so recalibrate the threshold). This maps each lost point directly to a knob.
- **The signature curves:** plot recall-versus-age and latency-versus-age, and make them convincing by showing Eidetic-Plus flat across session age while Mem0 and Graphiti degrade. The bi-temporal as-of filter and the salience-pruned index are exactly why recall stays flat. Pair every accuracy number with its token cost and latency to pre-empt the saturated-benchmark critique.

---

## 9. STAGED RECOMMENDATIONS

**Stage 1, structural wins, highest leverage, do first.**
1. Add the Chronos-style query-conditioned structured event calendar: extract subject-verb-object tuples, normalize all relative dates to ISO-8601 ranges at write time, attach paraphrase aliases, index events separately, and run a structured as-of filter at query time. Direction: temporal 81.1 toward 90 plus, multi-session 79.3 toward 88 plus.
2. Promote preferences to first-class typed memories with a persona/dialectic pass and a rubric-aware answer prompt on the shared reader. Direction: SSP 73.3 toward 90 plus.
3. Build the calibrated abstention gate as machinery, with the threshold as a config parameter and a calibrate command, leaving the actual number for post-key tuning.

**Stage 2, retrieval tuning, sweep on a subset.**
4. Turn on cross-encoder rerank at depth 50 to about 8, A/B against off.
5. Make RRF weights query-adaptive, keep k=60.
6. Raise efSearch to about 256 to 500, keep M=32 and 1024-dim embeddings.
7. Keep the hybrid facts-plus-chunks context with lost-in-the-middle edge placement.

**Stage 3, efficiency, lock the cost and latency win.**
8. Keep roughly 80 percent of queries on qwen-flash via the cascade, escalate only on low confidence.
9. Apply LLMLingua-2-style compression selectively to raw chunks only.
10. Prune the retrieval index by FSRS salience, never the WORM store.

Benchmarks that change the plan: if multi-session stays below 85 after Stage 1, add single-pass PPR query decomposition for aggregation. If _abs over-abstains and answerable accuracy drops, loosen the conformal threshold. If qwen-flash accuracy on escalated queries is within 1 point of qwen3-max, push more traffic to flash. If contradiction-resolution is in scope, accept parity (roughly 60 to 66) as the ceiling.

---

## 10. HONEST CAVEATS

- **Contradiction-resolution is unsolved** (best public BEAM-1M 0.357). Do not promise a clean win there.
- **Benchmark numbers are contested and harness-dependent.** The same system appears at 58, 66, and 92 percent across sources; Engram's own paper documents truncation bugs and home-grown-judge distortions. Use the official judge and report variance.
- **Several leader numbers are vendor-reported** (ByteRover, Hindsight, Mastra, Chronos, HydraDB) and not all independently reproduced. Treat per-category claims as targets, not guarantees.
- **Engram's facts-only-versus-hybrid claim is an unmeasured design observation,** not a controlled ablation, and is from a single-author preprint with single runs. Validate it yourself.
- **LoCoMo is widely criticized** and its short contexts (16 to 26k tokens) fit modern context windows, so strong LoCoMo numbers prove less than strong LongMemEval-S numbers. Prioritize LongMemEval-S and BEAM for credibility.

---

## 11. THE CONDENSED `/goal` PROMPT (UNDER 4000 CHARACTERS)

Build the machinery now, leave the tuned values as parameters, calibrate after the key. Paste this into `/goal`.

```text
GOAL: Implement EIDETIC_PLUS_OPTIMIZATION_PLAYBOOK.md (+ files it references) on the existing repo in
one shot: top every LongMemEval + LoCoMo category while staying cheaper and faster than Mem0 and
Graphiti. Build on existing engine (retrieval, engine, graph, models, structure_code, fsrs,
dashscope_client, config.py) + bench/. Build, wire, test offline; don't stop until all below works and
the suite is green.

RULES: these are RETRIEVAL + CONSOLIDATION upgrades; the harness keeps ONE fixed reader across all
three systems (fairness fix stays); NLI/abstention/cascade stay product features in the cost/latency
tables, NOT a neutral-accuracy answerer edge. No mocks, fail loud on a missing key, never delete a raw
record, never fabricate a score. Treat playbook targets as DIRECTION; the real baseline comes
from the first run. Ask before any non-standard dep.

BUILD (highest leverage first):
1) STRUCTURED EVENT CALENDAR (Chronos-style; biggest lever for temporal + multi-session): in async
consolidation extract subject-verb-object event tuples and normalize EVERY relative date ("recently",
"the week after") to explicit ISO-8601 ranges via forward/backward inference; attach 2-4 paraphrase
aliases; index events SEPARATELY from raw turns. At query time parse the question for entities +
temporal constraints + operation (filter/count/order) and run a structured as-of filter over the
calendar, not string matching. Keep existing bi-temporal valid/invalid + supersedes.
2) FIRST-CLASS PREFERENCES: extract preferences as a typed memory at write time (not generic facts);
maintain a per-user profile accumulating preference insights across sessions; ensure they surface in
top-k; prompt the shared reader to surface and apply the stored preference (rubric-aware), without
giving Eidetic a different reader than the baselines.
3) CALIBRATED ABSTENTION (machinery now, numbers later): keep the NLI gate but make the threshold a
CONFIG PARAMETER with a safe default + a calibrate command computing a conformal threshold from a
held-out subset (target ~95% answer precision). Don't hardcode it. Combine NLI entailment + top
reranked-score coverage.
4) RETRIEVAL TUNING (parameterize, default sane, sweep later): cross-encoder rerank via qwen3-rerank
depth 50->~8 with an on/off flag; RRF k=60 fixed; query-adaptive RRF weights (BM25 up for name/date/ID,
graph/PPR up for multi-hop) in config; HNSW efSearch config ~256-500 (retrieval isn't the bottleneck),
keep M=32 + 1024-dim; keep hybrid facts+raw-chunks context, place top-scored evidence at context EDGES
(lost-in-the-middle); dedup hard.
5) MULTI-HOP: single-pass Personalized PageRank seeded from query-linked entity nodes (no iterative LLM
loop); lightweight decomposition ONLY for aggregation/counting, routed through the calendar.
6) EFFICIENCY: keep ~80% of queries on qwen-flash via the cascade, escalate to qwen-plus/qwen3-max only
on low confidence; LLMLingua-2-style compression on raw chunks ONLY (never facts); prune the retrieval
INDEX by FSRS salience, never the WORM store.

PARAMETERIZE (config default + sweepable, don't hardcode tuned values): abstention threshold, RRF
weights, rerank depth + on/off, efSearch, cascade threshold, compression ratio. Add a sweep command
(python -m bench.sweep) grid-searching them on a 50-100 Q subset, reporting the best config, ready once
a key exists. Update bench/README + README (event calendar, typed preferences, calibrated abstention,
sweepable params), keeping neutrality wording accurate.

DONE = all six upgrades wired into engine+harness; tuned values are config params with safe defaults
(NOT hardcoded); calibrate + sweep run offline on a tiny subset with no key (real, no faked scores);
shared-reader fairness intact + documented; suite green (offline tests: event-calendar date
normalization, typed preference extraction, query-adaptive RRF, conformal-threshold plumbing,
edge-placement); no mocks; MIT kept. Ready for key: add key -> bench.sweep -> calibrate -> full run.
```

---

## 12. SOURCES

- Chronos: temporal-aware conversational agents with structured event retrieval (arXiv:2603.16862), 95.6 SOTA, dual-calendar, plus 58.9 event-calendar lift.
- ByteRover 2.1.5 vendor benchmark posts: 92.8 LongMemEval with per-category table, 96.1 LoCoMo, 1.6s p50.
- Hindsight (arXiv:2512.12818): 91.4, four typed networks, 4-way parallel retrieval with RRF and cross-encoder rerank.
- Engram (arXiv:2606.09900): the baseline; lean-context proof, exact lean-slice config, efficiency numbers.
- HippoRAG-2, "From RAG to Memory" (Gutierrez et al., arXiv:2502.14802, ICML 2025): single-pass PPR multi-hop gains.
- BEAM benchmark (arXiv:2510.27246): contradiction_resolution 0.357, ten abilities, 1M to 10M tokens.
- FrugalGPT (Chen, Zaharia, Zou, arXiv:2305.05176): cascade cost reductions up to 98 percent.
- RouteLLM (Ong et al., ICLR 2025): 95 percent of GPT-4 at 26 percent GPT-4 calls.
- LLMLingua-2 (Pan et al., ACL 2024) and LongLLMLingua (arXiv:2310.06839): selective compression, latency cuts.
- HALT-RAG (arXiv:2509.07475) and the conformal-abstention literature (Conformal-RAG, CONFLARE, C-RAG): calibrated NLI thresholds and coverage guarantees.
- LongMemEval (Wu et al., arXiv:2410.10813): category counts and official category-specific judge prompts.
- Reciprocal Rank Fusion (Cormack, Clarke, Buttcher, SIGIR 2009): the k=60 default.
- Mastra Observational Memory (95.4) and Honcho (90.4) vendor reports; Memory-R1 (arXiv:2508.19828).
- text-embedding-v4 dimension data (Qwen3-Embedding, MRL/CMTEB): 1024-dim default, quantization tradeoffs.

> Treat every benchmark number, including any this system produces, as an upper bound with methodology and variance reported. Calibrate the tunable parameters on a real subset before the full run, lead with the two wins no competitor has, and call contradiction-resolution the frontier rather than a solved box.
