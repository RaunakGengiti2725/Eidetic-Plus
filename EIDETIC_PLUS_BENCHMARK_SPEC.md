# EIDETIC-PLUS: Benchmark-Domination Spec
### How to Beat Mem0 and Graphiti on Every Category, While Being Cheaper and Faster

> This is the build brief for the part that wins the hackathon: topping every accuracy category on LongMemEval and LoCoMo while spending fewer tokens and answering faster than both Mem0 and Zep/Graphiti, and proving it with a neutral evaluation harness. Read it alongside the two companion files already in the repo: `Eidetic-Plus_Master_Dossier.md` (the research foundation) and `EIDETIC_PLUS_UPGRADE_SPEC.md` (the universal-plugin and engine upgrades). A condensed `/goal` prompt that triggers this spec, under 4000 characters, is in Section 12.
>
> Discipline carried from the other files, repeated because it is what makes the win real: no mocks anywhere (every model call is a real DashScope call, fail loud on a missing key); the immutable record is the arbiter; never delete a raw record; and the heart of this spec is that the same neutral harness runs Eidetic-Plus, Mem0, and Graphiti under one fixed judge, because a number that does not reproduce does not exist.

---

## 0. THE ONE-SENTENCE GOAL

Make Eidetic-Plus lead every accuracy category on LongMemEval and LoCoMo while spending fewer tokens per write and per query than Mem0 and far fewer than Graphiti, answering with lower p95 latency than both, and prove all of it with one neutral harness that runs all three systems under a single fixed judge with multi-run variance.

---

## 1. THE GOAL AND THE HONEST REALITY

You want to win every category, the way moving from Sonnet to Opus wins everywhere at once. That ambition is correct, and the path to it is real, but it has to be built category by category, because that is literally how a model upgrade is built underneath: hundreds of targeted fixes that together look like uniform improvement. So this spec maps every category to its failure mode to a concrete mechanism.

Two honest facts that shape what you should claim, because overclaiming is the fastest way to lose technical judges:

1. **No single public system today wins every sub-category at once.** Leadership is fragmented: different systems own different categories (Hindsight and Supermemory lead the LongMemEval long-horizon categories, ByteRover 2.0 and MemoryLake lead LoCoMo, Zep leads temporal when paired with weaker base models). That fragmentation is good news, because it means there is a winnable gap in every category and no monolithic champion to dethrone. But it means a clean sweep is a category-by-category campaign, not one knockout.

2. **The single genuinely hard target is cross-session contradiction resolution at scale.** On the BEAM-1M benchmark the best public score for that ability is only 0.357, and maintaining a globally consistent "what is true right now" state is still an open problem across the field. So the defensible plan is: win every category on LongMemEval and LoCoMo (provable, and that is the hackathon yardstick), lead with the two categorical wins no competitor has at all (flat recall-versus-age, and verified recall with a citable immutable source), and present BEAM contradiction resolution as the frontier you push on rather than a solved box you check.

This is not hedging. Winning LongMemEval and LoCoMo outright, while being cheaper and faster, with two properties the others structurally lack, is domination. Claiming a clean sweep of an unsolved 10M-token contradiction benchmark would be the one move that makes a sharp judge stop believing the rest.

---

## 2. THE CORE THESIS: WHY CHEAPER PLUS MORE ACCURATE IS NOT A CONTRADICTION

In most systems accuracy and cost trade against each other. In conversational memory they do not, and the reason is the single most important empirical result in the 2026 literature: a precisely retrieved lean slice beats the full noisy history. The published proof point is Engram (arXiv:2606.09900, June 2026): answering from a roughly 9,600-token retrieved slice scores 83.6 percent on the full 500-question LongMemEval_S under the official judge, versus 73.2 percent for stuffing the full roughly 79,000-token history into context. That is plus 10 points at one-eighth the tokens, a paired McNemar-significant gain.

The mechanism is "lost in the middle": once context grows past tens of thousands of tokens, models attend worst to the middle, so the full history actively hurts. This is the lever. Removing distractors raises accuracy and cuts cost at the same time. Mem0 is cheap because it discards most of the conversation, which loses accuracy on hard categories. Graphiti is accurate on temporal because it builds rich structure, which costs more than 600,000 tokens to ingest one conversation. The design that wins both is lossless storage (cheap on write) plus lean, structured retrieval (cheap on read, and more accurate than full context). Your existing 7-component engine is already a superset of this: lossless store, bi-temporal graph, in-app Personalized PageRank, vector ANN, FSRS, NLI verification, salience gate. You are on the right path. This spec sharpens the per-category mechanisms and builds the harness that proves the win.

---

## 3. THE EXACT TARGETS TO BEAT

Instrument against these specific numbers. These are what Mem0 and Graphiti actually cost, so these are the bars.

| System | Tokens to build (per conversation) | Tokens per query | Search latency p50 / p95 | Ingestion behavior |
|---|---|---|---|---|
| Full-context baseline | none | ~26,000 | 9.87s / 17.12s | none |
| Mem0 (2026 algorithm) | ~7,000 to 14,000 (Mem0g graph doubles it) | under 7,000 | 0.148s / 0.200s | add() 300 to 800ms, must be async |
| Zep / Graphiti | over 600,000 | top-10 nodes and edges | ~0.3s p95 | multiple LLM calls per episode, hours of post-ingestion lag |

**Eidetic-Plus targets:** build cost at or below Mem0 (near-zero, because there is no LLM call on the write path, only async consolidation tokens); query cost at or below Mem0 (roughly 5,000 to 9,000 tokens of retrieved context); search p95 under 0.2s; ingestion visible in under 1 second (no Graphiti-style lag). If you hit these, the cost-and-speed half of the triple win is done, and it is the easier half. The accuracy half is Sections 4 and 5.

---

## 4. THE UNIFIED PIPELINE (ONE ARCHITECTURE, CHEAP ON EASY, ACCURATE ON HARD)

The whole system is one dual-process pipeline. The principle: keep LLM calls off both the hot write path and the hot read path, do multi-hop in a single graph step instead of an agent loop, and escalate to expensive models only on the queries that need them.

**Write path (System 1, no LLM, target under 50ms).** Append the lossless episode to the immutable store, resolve identity across sessions and devices, embed with text-embedding-v4 (default 1024 dimensions, up to 8,192-token input), enqueue for consolidation. Build cost is near Mem0's and latency is far below Graphiti's because nothing here calls an LLM.

**Consolidation (System 2, async, seconds, off the hot path).** Extract atomic (subject, predicate, object) facts, build the bi-temporal graph, actively detect conflicts and invalidate superseded facts, normalize dates into structured attributes, and score salience with FSRS decay and reinforcement so the index stays small and fast. Use qwen-flash for extraction and escalate only genuinely ambiguous conflicts to qwen3-max.

**Read path (hybrid, target under 100ms before any LLM call).** Query understanding, then optional decomposition only for multi-hop, then four channels in parallel: dense vector recall, BM25 lexical recall, single-step Personalized PageRank over the graph, and recency/salience. Fuse with Reciprocal Rank Fusion, optionally rerank the top-k with qwen3-rerank, apply the bi-temporal as-of filter, run the NLI abstention gate, and assemble a deduplicated, provenance-tagged, token-budgeted hybrid context of facts plus raw chunks plus session summaries. The hybrid context is load-bearing: facts alone lose recall, so keep raw chunks alongside extracted facts.

**Answer cascade (route by difficulty).** qwen-flash for easy single-hop and preference queries, qwen-plus for multi-hop and temporal, qwen3-max only for contradiction adjudication and hard open-domain. Put a semantic cache in front of everything (exact-hash plus cosine at or above 0.90 to 0.95). Roughly 70 to 80 percent of queries route to the cheap model, which is where most of the cost win comes from. Default ambiguous queries to a conservative tier, because a wrong cheap answer costs more than an unnecessary escalation.

---

## 5. PER-CATEGORY WINNING MECHANISMS

Each category, the current best public number, why the incumbents lose points there, the mechanism that wins it, and the target. Most of these mechanisms already exist in your engine; this is where to sharpen them.

### 5.1 LongMemEval categories (numbers are LongMemEval_S)

| Category | Best public | Why incumbents lose | Eidetic-Plus mechanism | Target |
|---|---|---|---|---|
| single-session-user | Supermemory 97.1, Zep 92.9, Hindsight+OSS-120B 100 | Fact never extracted or indexed | Lossless episode store plus hybrid dense+BM25; never depend on extraction alone | match top |
| single-session-assistant | Hindsight 98.2, full-context 94.6, Zep only 80.4 | Assistant-stated facts treated as second-class (Zep drops 17.7 points) | Treat assistant turns as first-class memories | beat Zep decisively |
| single-session-preference | RetainDB 88, Hindsight 86.7, Zep 56.7, full-context only 20 | Preferences are diffuse, rubric-graded | Profile/identity memory plus salience-weighted preference aggregation, graded to the rubric | beat Zep by a wide margin (HARD category) |
| multi-session | Hindsight 81.2, Supermemory 71.4, Zep 57.9 | Evidence fragmented across sessions, context dilution | Single-step PPR multi-hop plus facts+chunks hybrid context | beat Zep, approach Hindsight |
| knowledge-update | Hindsight+Gemini-3 94.9, Engram 87.5, Zep 83.3 | Both old and new facts surface; no supersession chain | Bi-temporal invalidate-not-delete plus as-of filter | top tier |
| temporal-reasoning | Hindsight+Gemini-3 91.0, Zep 62.4, full-context 45.1 | Timestamps not preserved as structured attributes | Bi-temporal model plus per-entity timeline plus date normalization | beat Zep at its own strength, cheaper |

### 5.2 LoCoMo categories (the four validated categories; exclude adversarial, which lacks ground truth and both Mem0 and Zep exclude)

| Category | Mem0 | Zep (corrected) | Strong recent (ByteRover 2.0 / MemoryLake) | Eidetic-Plus mechanism |
|---|---|---|---|---|
| single-hop | 67.13 | 74.11 | 95.4 / 96.79 | Hybrid retrieval over lossless store, answered by qwen-flash |
| multi-hop | ~51 | 66.04 | 85.1 / 91.84 | Single-step PPR, decompose only when entity-linking confidence is low |
| temporal | 58.13 (Mem0g) | 79.79 | 94.4 / 91.28 | Bi-temporal plus timeline plus as-of filter, no LLM on the read path |
| open-domain | 75.71 (Mem0g) | 67.71 | 77.2 / 85.42 | Parametric-knowledge fusion plus abstention gate (the universal weak spot) |

### 5.3 The mechanisms in detail

**Temporal.** Make both time axes first-class on every fact and edge: valid_at and invalid_at for world time, created_at and expired_at for transaction time. Parse the query date, normalize relative dates ("last Tuesday," "May 2023") into structured attributes at consolidation time (this is exactly Mem0's documented 14.9-percent-of-failures blind spot), build a per-entity timeline, and apply the as-of filter at read time. This beats Graphiti at its strength because the timeline and filter are computed with no LLM call on the read path. Use the LongMemEval temporal judge's off-by-one tolerance.

**Multi-hop.** Personalized PageRank over the knowledge graph in a single retrieval step, HippoRAG-style, which does multi-hop reasoning in one step and beats iterative RAG by roughly 20 points on standard multi-hop sets while staying cheap because there are no iterative LLM calls. Your in-app PPR is the right primitive. Feed it the facts-plus-chunks hybrid context so intermediate links are not dropped (Mem0's 6.4-percent reasoning-chain-break failure).

**Knowledge-update / contradiction.** When a new fact arrives for an existing (subject, predicate) slot with a different object, resolve cheap-then-escalate: exact slot match plus embedding similarity catches the same attribute under a different predicate; content subsumption separates contradiction from elaboration; if clearly contradictory and temporally ordered, set the old fact's invalid_at to the new fact's valid_at and record new.supersedes = old.id, with no LLM call; escalate only genuinely ambiguous cases to qwen3-max. This is the cost win over Graphiti (which calls an LLM per fact) and the consistency win over Mem0 (whose ADD-only model keeps both facts). Invalidate, never delete, so both "what is true now" and historical "as of then" queries are answerable.

**Abstention.** This is where many systems fail and where your NLI verification is decisive. Run an abstention gate before answering: an NLI ensemble (for example RoBERTa-MNLI plus DeBERTa-v3) scores whether the retrieved evidence entails a candidate answer, combined with retrieval-quality signals (relevance and coverage) and a calibrated threshold (Chow's reject option or Platt scaling). Abstain when evidence is insufficient rather than guessing. Use the LongMemEval abstention judge semantics and tune the threshold to maximize correct abstention on unanswerable questions while minimizing wrongful abstention on answerable ones.

**Single-hop and preference.** Match Mem0's speed with hybrid dense plus BM25 plus entity retrieval over the lossless store (so nothing is lost to extraction), fused by Reciprocal Rank Fusion, answered by qwen-flash. For preference, maintain a profile/identity memory that aggregates salience-weighted preference facts and grade it against the preference rubric. Hybrid BM25-plus-vector reaches roughly 95 percent on single-hop-class retrieval, nearly matching pure vector.

**Open-domain.** The universal weak spot. Fuse retrieved memory with the answering model's parametric knowledge: when retrieval coverage is high, answer from memory; when memory is thin but the question is world-knowledge, allow the model to use parametric knowledge (escalate to qwen-plus or qwen3-max), gated by the abstention calibrator to avoid hallucination.

---

## 6. THE EVALUATION HARNESS (THE MOST IMPORTANT DELIVERABLE)

This is the heart of the spec. Without it, "we beat them" is a claim. With it, it is a scoreboard. Build one harness that runs all three systems through the identical answerer and judge by construction.

### 6.1 Datasets and graders
- **LongMemEval:** official repo `github.com/xiaowu0162/LongMemEval`; download `longmemeval_s` (and the oracle set) from the HuggingFace dataset `xiaowu0162/longmemeval-cleaned`; grade with the official `evaluate_qa.py`, which uses category-specific judge prompts (temporal off-by-one tolerance, knowledge-update old-info tolerance, preference rubric leniency, abstention detection); aggregate per category. The official judge model is GPT-4o.
- **LoCoMo:** official repo `github.com/snap-research/locomo`; restrict to the four validated categories (single-hop, multi-hop, temporal, open-domain) and exclude adversarial; report the LLM-as-judge J score (and F1 as a secondary, noting F1 penalizes correct paraphrases).

### 6.2 Fair self-hosted baselines through one harness
- **Mem0:** `mem0ai/mem0` (Apache 2.0) plus the `mem0ai/memory-benchmarks` runners.
- **Graphiti:** `getzep/graphiti` self-hosted on Neo4j (the engine self-hosts even though the full Zep app's community edition was deprecated).
- The closest existing single harness that already runs supermemory, mem0, and zep with swappable judges is `supermemoryai/memorybench`; use it as the template so all three systems share one answerer and one judge.
- Do not compare your fresh numbers to anyone's published numbers. Re-run every system yourself under identical conditions. Published numbers in this field are inflated and disputed.

### 6.3 The judge, honestly handled for a Qwen stack
The official LongMemEval judge is GPT-4o. If you have GPT-4o access, use it for the headline number so it is comparable to published results. If you are Qwen-only, fix qwen3-max as the judge across all three systems: the relative comparison stays valid even though the absolute number is not directly comparable to GPT-4o-judged leaderboards, and you state that explicitly. Either way, the rule is one fixed judge and one fixed reader prompt across all three systems. Report a second judge as a robustness check.

---

## 7. NEUTRAL METHODOLOGY (THIS IS WHAT MAKES THE NUMBERS BELIEVED)

The Mem0-versus-Zep dispute is the cautionary tale: the same system was reported at 84 percent, then 58.44 percent, then 75.14 percent on the same benchmark, depending on judge, prompt, category selection, and run count. To be credible:
- Fix one judge model and one reader/system prompt across all systems.
- Restrict LoCoMo to the four validated categories.
- Average at least 10 independent runs (LoCoMo) or multiple seeds (LongMemEval) and report mean and variance, not a single lucky run.
- Pre-register the configuration before looking at results (data hash, index parameters, model snapshots). Pin exact Qwen snapshots, because the `qwen-flash`/`plus`/`max` aliases rotate to newer models over time.
- Publish raw per-question logs and the exact one-line reproduce command. A number that does not reproduce does not exist.
- Run the official judge and one second judge; if they disagree by more than about 2 points on your system, report both and do not cherry-pick.

If you do this and Eidetic-Plus wins, the win is real and defensible. If you skip it and report one high number, a sharp judge will not believe it, and they will be right.

---

## 8. COST, LATENCY, AND THE SIGNATURE CURVES

Accuracy is only one third of the claim. Report all three, on identical hardware, for all three systems.
- **Cost:** tokens per write (build cost) and tokens per query, per system. Target: build at or below Mem0, query roughly 5,000 to 9,000 tokens.
- **Latency:** search p50 and p95, and end-to-end p50 and p95. Target: search p95 under 0.2s, ingestion visible under 1 second.
- **The two signature curves, run on all three systems:** recall-versus-age (accuracy as a function of how many sessions ago the evidence appeared) and latency-versus-age (and versus store size). The thesis to show visually: Eidetic-Plus stays flat where Mem0 degrades with age (extraction loss compounds) and where Graphiti's ingestion cost and latency grow with graph size. Add the BEAM 1M-to-10M scaling point as a stress test, with the honest note that everyone, including you, drops in the 10M regime.

These two curves are your strongest single piece of evidence, because flat recall-versus-age is a property the others structurally do not have, not just a score where you are a few points higher. Pair them with verified recall (you can cite the immutable source for every answer; they cannot) and you have two categorical wins on top of the per-category leads.

---

## 9. HONEST CAVEATS AND WHAT TO CLAIM

- **Claim:** Eidetic-Plus leads every category on LongMemEval and LoCoMo under a neutral fixed-judge harness, at lower token cost and lower p95 latency than Mem0 and Graphiti, with two properties neither has (flat recall-versus-age, verified recall with citable sources). This is provable and is domination.
- **Do not claim:** a clean sweep of BEAM, especially contradiction resolution at 10M tokens, which no public system has solved. Present it as the frontier.
- **Vendor numbers are contested.** Treat every published figure (including any you produce) as an upper bound tuned to a private harness. Only your own neutral re-runs count.
- **Judge and prompt drift move scores several points silently.** Standardize and disclose.
- **Some apparent wins are scorer artifacts.** Exact-match F1 penalizes correct paraphrases; the LLM-judge J score is more faithful but judge-dependent. Report both and inspect failures.

---

## 10. BUILD STAGES (TO JULY 9, 2026)

1. **Lock the harness first.** Stand up the neutral harness with the official LongMemEval judge (or fixed qwen3-max), the four-category LoCoMo J score, 10-run variance, pinned snapshots, and published logs. Reproduce Mem0 and self-hosted Graphiti baselines through it before touching your own system. Gate: if you cannot reproduce Mem0 within about 2 points of its published numbers, your harness is wrong; fix it before claiming anything.
2. **Ship the dual-process core and win the bi-temporal categories first.** LLM-free write path, async consolidation, bi-temporal invalidate-not-delete, HNSW at M=32/efSearch=128. Match Mem0 on cost and latency, then beat it on knowledge-update and temporal (where bi-temporal pays off).
3. **Win the hard categories.** Single-step PPR multi-hop, the NLI abstention gate with a calibrated threshold, profile memory for preference, and parametric-fusion plus cascade for open-domain. Gate: if any category trails the best public number, add the specific mechanism for it.
4. **Optimize the triple win.** Semantic caching, difficulty-routed model cascade (flash to plus to max), and selective compression on long retrieved slices. Tune ef_search and the rerank budget against the latency target. Produce the recall-versus-age and latency-versus-age curves on all three systems for the demo.

Thresholds that change the plan: if contradiction resolution cannot clear about 0.40 on BEAM-1M, scope the headline claim to LongMemEval and LoCoMo and present BEAM as future work. If qwen-flash multi-hop quality trails qwen-plus by more than 3 points, raise the default routing tier. If the two judges disagree by more than 2 points on your system, report both.

---

## 11. RESEARCH APPENDIX: THE NUMBERS, WITH SOURCES

- **Lean-beats-full proof:** Engram, arXiv:2606.09900 (June 2026), 83.6 vs 73.2 on LongMemEval_S at roughly one-eighth the tokens (9.6k vs 79k), McNemar-significant.
- **Retrieval efficiency:** ENGRAM-R (arXiv:2511.12987), minus 85 percent input tokens and minus 75 percent reasoning tokens vs full context, multi-hop plus 7.9 and temporal plus 7.7. Cost-Aware RAG (arXiv:2606.02581), minus 26 percent tokens and minus 34 percent latency at quality parity. Roughly 81 percent of queries route to the cheap model, saving about 85 percent of LLM spend.
- **Multi-hop:** HippoRAG (NeurIPS 2024, arXiv:2405.14831), single-step PPR, roughly plus 20 points over iterative RAG.
- **Mem0 internals and cost:** arXiv:2504.19413 (ECAI 2025); under 7,000 tokens per retrieval call (mem0.ai/research, May 2026); search p50 0.148s / p95 0.200s; documented failures 78.8 percent extraction gaps, 14.9 percent temporal blindness, 6.4 percent reasoning-chain breaks; ADD-only keeps contradictions; May 2026 update reports 92.5 LoCoMo and 94.4 LongMemEval with launch-era gains of plus 53.6 assistant recall, plus 29.6 temporal, plus 23.1 multi-hop.
- **Graphiti internals and cost:** getzep/graphiti; bi-temporal validity intervals; over 600,000 tokens to ingest one conversation; hours of post-ingestion lag; single-session-assistant regression minus 17.7 points; preference 56.7, multi-session 57.9.
- **Benchmark category counts:** LongMemEval (arXiv:2410.10813) 70 single-session-user, 56 single-session-assistant, 30 single-session-preference, 133 multi-session, 78 knowledge-update, 133 temporal, plus abstention variants. LoCoMo (Maharana et al. 2024) 1,986 QA over 10 conversations, roughly 26k tokens each.
- **BEAM:** arXiv:2510.27246; 100 conversations up to 10M tokens, 2,000 questions, ten abilities. BEAM-1M per-ability lowest is contradiction_resolution 0.357; Mem0 BEAM-1M/10M 64.1/48.6.
- **Abstention:** HALT-RAG (arXiv:2509.07475), calibrated NLI ensembles with a reject option.
- **Model cost ordering:** qwen-flash < qwen-plus < qwen3-max, roughly a 7x input-cost spread; text-embedding-v4 at 64 to 2048 dims (default 1024), 8,192-token max input; the Beijing endpoint is roughly 60 to 70 percent cheaper. Verify live rates before finalizing cost claims.

---

## 12. THE CONDENSED `/goal` PROMPT (UNDER 4000 CHARACTERS)

Paste this into `/goal`. It triggers the entire spec above.

```text
GOAL: Read EIDETIC_PLUS_BENCHMARK_SPEC.md and the files it references, then execute it in one shot on
the existing Eidetic-Plus repo. Objective: Eidetic-Plus leads every LongMemEval and LoCoMo category
while spending fewer tokens and answering faster than Mem0 and Graphiti, PROVEN by a neutral harness.
Build, wire, run it, generate the scoreboard + curves; don't stop until all below works.

RULES: no mocks (real DashScope calls, fail loud on a missing key, never fake a result or a score);
never delete a raw record; the immutable record is the arbiter; one fixed judge + one fixed reader
prompt across ALL THREE systems; a number that doesn't reproduce doesn't exist. Ask before any
non-standard dep.

BUILD, in this order (harness first; it is the deliverable that proves the win):
1) NEUTRAL HARNESS (most important): run Eidetic-Plus, Mem0 (mem0ai/mem0 + mem0ai/memory-benchmarks),
and Graphiti (getzep/graphiti self-hosted on Neo4j) through ONE harness with the SAME answerer and
judge by construction (template: supermemoryai/memorybench). LongMemEval via github.com/xiaowu0162/
LongMemEval + HF xiaowu0162/longmemeval-cleaned, official evaluate_qa.py with category-specific judge
prompts. LoCoMo via github.com/snap-research/locomo, FOUR validated categories only (single-hop,
multi-hop, temporal, open-domain), exclude adversarial, J score. Judge: GPT-4o if available, else fix
qwen3-max across all three and say so. >=10 runs, report mean +/- variance, pin exact model snapshots,
publish raw per-question logs + a one-line reproduce command. GATE: if you can't reproduce Mem0 within
~2 points of its published numbers, the harness is wrong, fix it first.
2) WIN THE BI-TEMPORAL CATEGORIES: keep all LLM calls off the hot write and read paths. Write path (no
LLM, <50ms): append lossless episode, embed (text-embedding-v4, 1024-dim). Consolidation (async):
extract (s,p,o) facts, build bi-temporal graph (valid_at/invalid_at + created_at/expired_at), ACTIVELY
detect conflicts and invalidate-not-delete (old.invalid_at=new.valid_at, new.supersedes=old.id, no LLM
unless ambiguous -> qwen3-max), normalize relative dates to structured attributes, FSRS-score salience.
As-of filter at read time. Target knowledge-update + temporal wins.
3) WIN THE HARD CATEGORIES: single-step Personalized PageRank for multi-hop (no iterative LLM loop;
decompose only on low entity-linking confidence; feed facts+raw-chunks hybrid context); NLI abstention
gate before answering (NLI ensemble + retrieval-coverage signal + calibrated threshold; abstain when
evidence is insufficient); profile/identity memory with salience-weighted preference aggregation graded
to the rubric; open-domain = parametric-knowledge fusion gated by the abstention calibrator.
4) TRIPLE-WIN OPTIMIZATION: read path is hybrid (dense + BM25 + PPR + recency) -> RRF -> optional
qwen3-rerank -> as-of filter -> NLI gate -> token-budgeted hybrid context (~5-9k tokens). Answer cascade
routed by difficulty: qwen-flash (single-hop/preference) -> qwen-plus (multi-hop/temporal) -> qwen3-max
(contradiction/hard open-domain). Semantic cache in front (exact-hash + cosine >=0.9). HNSW
M=32/efSearch=128. Targets: build tokens <= Mem0, query 5-9k tokens, search p95 <0.2s, ingestion <1s.

ALSO produce, on all three systems: a per-category accuracy scoreboard (LongMemEval + LoCoMo), a cost
table (tokens/write, tokens/query), a latency table (search + e2e p50/p95), and the two signature
curves (recall-vs-age and latency-vs-age, plus the BEAM 1M->10M scaling point). Save as committed
artifacts.

DONE = the harness runs all three systems under one fixed judge with variance; Eidetic-Plus leads every
LongMemEval and LoCoMo category in the scoreboard; cost and latency tables show it cheaper and faster
than both; the two curves show it flat where the others degrade; raw logs + reproduce command
published; honest note: BEAM contradiction resolution at 10M is the frontier, not solved.
Tests pass; README + docs updated. MIT kept.
```

---

*One file, the whole path to winning. Lean retrieval beats full history, so cheaper and more accurate are the same move. Win the bi-temporal categories with invalidate-not-delete, the hard categories with single-step PPR and a calibrated abstention gate, and the triple win with a routed cascade and a semantic cache. Then prove all of it with one neutral harness, because the scoreboard is the argument. Lead with the two wins no one else has, and call the one unsolved benchmark the frontier instead of pretending it is solved.*
