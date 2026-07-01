# Eidetic-Plus Neutral Benchmark Harness

> A number that doesn't reproduce doesn't exist.

This is **one neutral harness** that runs **Eidetic-Plus**, **RAG baselines**, **Mem0**, and **Graphiti** through the
**same fixed judge** and the **same fixed reader**, on **LongMemEval** and **LoCoMo**. Every system
is driven through an identical ingest -> retrieve -> answer -> grade loop, so the scoreboard measures
**memory quality** (what each system retrieves), not answerer quality. We never fabricate a score: the
scoreboard and curves render **only** from real per-question logs, and a missing key or dependency
**fails loud** rather than returning a mock.

For LoCoMo we restrict to the **four validated categories**: `single-hop`, `multi-hop`, `temporal`,
`open-domain`, and **exclude adversarial** (category 5), which lacks reliable ground truth and is
excluded by both Mem0 and Zep. (LongMemEval uses its own six question types; see
[Datasets](#datasets).)

---

## Quickstart (subset-smoke, cheap)

A handful of questions, one run, Eidetic-Plus only: demonstrably real for a few cents:

```bash
python -m bench.run --systems eidetic --dataset locomo --subset 10 --runs 1
```

Needs `DASHSCOPE_API_KEY` in `.env` (real Qwen calls; no mocks). This ingests 10 LoCoMo questions'
conversations into Eidetic-Plus, answers them through the shared fixed reader, grades with the fixed
judge, writes the raw log to `artifacts/bench/eidetic-plus__run0.jsonl`, and renders the scoreboard +
curves from that log.

---

## CLI flags

`python -m bench.run` accepts:

| flag | default | meaning |
|---|---|---|
| `--systems` | `eidetic` | comma list of systems to run: `eidetic`, `eidetic-full`, `eidetic-product`, `rag-full`, `rag-vector`, `mem0`, `graphiti` (any combination). |
| `--dataset` | `locomo` | which dataset(s) to load: `longmemeval`, `locomo`, or `both`. |
| `--subset` | `10` | limit samples per dataset. **`<= 0` means full** (no limit). |
| `--sample-offset` | `0` | start offset for a fresh-slice rerun without changing the loader. |
| `--runs` | `1` | independent runs for variance. Use **`>= 10`** to get a meaningful mean ± variance. |
| `--run-offset` | `0` | first run index, so reruns append `run1`, `run2`, etc. instead of overwriting `run0`. |
| `--overwrite` | off | allow replacing existing `system__runN.jsonl` files. Default is to abort. |
| `--variant` | `longmemeval_s` | LongMemEval variant file to load. |
| `--out` | `artifacts/bench` | output directory for logs, scoreboard, and curves. |
| `--render-only` | off | skip running; just re-render the scoreboard + curves from existing logs in `--out`. |

---

## Full Reproduce

The one-line command: both datasets, full sets, all public benchmark rows, `>= 10` runs for variance:

```bash
bash bench/reproduce.sh
```

This runs `--systems eidetic,eidetic-full,eidetic-product,rag-full,rag-vector,mem0,graphiti
--dataset both --subset 0 --runs 10 --split test --out artifacts/bench`.
It defaults `METABOLISM_MODE=1`, `DASHSCOPE_MAX_CONCURRENCY=2`, and `DASHSCOPE_RPM=30` unless you
set them explicitly, so LongMemEval-scale transcripts use the bounded long-haystack path instead of
burning the whole sleep deadline on auxiliary extraction. In that profile, `RAW_SPAN_AUDIT=1`
lets retrieval scan huge raw-only transcripts for exact query-supported spans and send only the
bounded supporting slice to the reader, preserving recall without paying full-history query tokens.
Use `OUT=artifacts/my_claim bash bench/reproduce.sh` to write a claim into a separate directory.
See [Honest status](#honest-status) for what it costs and what must be in place first.

Use a separate `--out` directory per claim. The scoreboard reads every `*__run*.jsonl` in that folder,
so mixing smoke runs, full runs, and fresh-slice studies will merge them.

---

## Datasets

Both datasets load **offline with no key** (network is used only to download the files once; loading
and verification need no API key).

**LoCoMo**: official schema from [`snap-research/locomo`](https://github.com/snap-research/locomo)
(`data/locomo10.json` inside that repo). The loader is **local-first**: it looks for
`data/bench/locomo/locomo10.json`, and if absent auto-downloads it from the official raw GitHub URL
(else raises with clone-and-copy instructions, no mock). The verified loader yields **1540
validated-category questions**:

| category | count |
|---|---|
| single-hop | 841 |
| multi-hop | 282 |
| temporal | 321 |
| open-domain | 96 |
| **total** | **1540** |

Adversarial (category 5) is excluded. The loader's `verify()` checks at runtime that only validated
categories are present and adversarial is excluded.

**LongMemEval**: official schema from
[`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) on
Hugging Face, into `data/bench/longmemeval/`. Local-first with the same auto-download-or-raise
behaviour. LongMemEval has its **own** six question types, and the loader's `verify()` checks the
official category counts (`single-session-user` 70, `single-session-assistant` 56,
`single-session-preference` 30, `multi-session` 133, `knowledge-update` 78, `temporal-reasoning` 133).

---

## Baselines (Mem0 + Graphiti)

The baselines are **not** needed to run Eidetic-Plus itself. They are only needed to compare against it. Install them
into the same venv:

```bash
.venv/bin/pip install -r requirements-bench.txt
```

That pins **`mem0ai==2.0.7`**, Mem0's material optional capabilities **`spacy==3.8.13`** and
**`fastembed==0.8.0`**, plus **`graphiti-core==0.29.2`** and **`neo4j==6.2.0`** (resolved cleanly on
CPython 3.14).

- **Graphiti needs a running Neo4j.** A free **Neo4j AuraDB** (cloud, **no Docker**) works; set
  `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.
- **External baselines are configured to use DashScope**: LLM `qwen-plus`, embedder `text-embedding-v4`
  via the OpenAI-compatible endpoint, so **the same model family backs every comparable row**. Each baseline
  drives its own retrieval, then hands the retrieved memories to the **one shared fixed reader**
  (`answer_with_fixed_reader`); baselines never use their own answer generator. A missing dependency,
  empty `DASHSCOPE_API_KEY`, or absent Neo4j **fails loud**.

---

## The judge

One **fixed judge** grades every system identically. That is the neutrality guarantee.

- **Default:** `qwen3-max` (the Qwen stack, from `GEN_MODEL`), via DashScope.
- **Leaderboard-comparable headline:** configure a **GPT-4o** judge by setting `JUDGE_BASE_URL`
  (e.g. `https://api.openai.com/v1`), `JUDGE_API_KEY`, and `JUDGE_MODEL=gpt-4o`. The harness **records
  which judge was used** in `scoreboard.md`/`.json`.
- **Robustness:** running a **second judge** as a cross-check is recommended.

Judging is category-aware: LongMemEval applies temporal off-by-one tolerance, knowledge-update
old-info tolerance, preference-rubric leniency, and abstention detection; LoCoMo uses the LLM-as-judge
J score. The judge is **distinct from the reader**. The shared fixed reader is `qwen-plus`
(`READER_MODEL`), used to turn each system's retrieved context into an answer.

---

## Methodology / neutrality

The discipline that makes the comparison fair and reproducible:

1. **One fixed judge + one fixed reader** across every benchmark row, by construction.
2. **Four LoCoMo categories** only (`single-hop`, `multi-hop`, `temporal`, `open-domain`); adversarial
   excluded.
3. **`>= 10` runs**, reported as **mean ± variance** per category.
4. **Pin Qwen snapshots** via the `*_MODEL` env vars in `.env`, because Qwen aliases rotate. Pin them
   so they don't change mid-study.
5. **Publish the raw per-question logs** (`artifacts/bench/*__run*.jsonl`) plus this reproduce command,
   so every number reproduces from the logs.

**GATE:** if **Mem0 cannot be reproduced within ~2 points of its published numbers**, the harness is
wrong. **Fix the harness first** before trusting any other number.

Run the gate from real logs only:

```bash
python -m bench.gate \
  --out artifacts/bench \
  --expected data/bench/mem0_locomo_published.json \
  --report-out artifacts/bench/mem0_gate.md
```

The expected file is explicit so the repo never bakes in a vendor number:

```json
{
  "min_n": 50,
  "tolerance_points": 2.0,
  "categories": {
    "single-hop": 0.0,
    "multi-hop": 0.0,
    "temporal": 0.0,
    "open-domain": 0.0
  }
}
```

Replace the zeroes with the published Mem0 LoCoMo reference values before running the gate. The command
prints per-category `n` first and fails if a category is missing, the slice is too small, or the deltas
exceed tolerance. It writes `mem0_gate.md` plus `mem0_gate.json`; the public release gate requires
that JSON report to pass and to carry the same raw-log fingerprint as the current JSONL files.

**PUBLIC RELEASE GATE:** after a full reproduce run, this command is the fail-closed check for public
claims:

```bash
python -m bench.release_gate --out artifacts/bench --report-out artifacts/guard/release_gate.md
```

The gate also requires a real dev-split ablation sidecar. Generate the three comparable dev artifacts
and the sidecar in one command:

```bash
python -m bench.run_dev_ablation \
  --out-root artifacts/dev_ablation \
  --dataset both \
  --subset 50 \
  --runs 1 \
  --system-under-test eidetic-plus-full \
  --report-out artifacts/bench/ablation_report.json
```

`bench.run_dev_ablation` writes one dev samples file, runs full memory, metabolism/consolidation off,
and forgetting-pruning off into isolated directories, then calls `bench.build_ablation_report`.
Before spending on subprocesses, it checks the effective `SALIENCE_PRUNE_THRESHOLD` and
`DREAM_PRUNE_PERCENTILE` profile, including inherited shell env and CLI overrides; the full run must
be at least one pruning knob stronger than forgetting-off, and forgetting-off must not prune more on
any knob. The sidecar records those effective profiles in `forgetting_cost_profiles`.
If you already have the three directories, call `python -m bench.build_ablation_report` directly.
The builder reads strict JSONL logs, requires matching sample/run keys, computes sample-clustered
verified accuracy plus query-token cost, and binds all three source artifacts by log fingerprint. It
fails if the evidence is from the test split, lacks verified metadata, has unpaired rows, or does not
show memory/consolidation earning accuracy and forgetting earning cost without meaningful accuracy
regression.

`bash bench/reproduce.sh` now runs that audit bundle automatically after the benchmark: it writes
the holdout-leakage audit, `forensics.md`, `snap_back_audit.json`, `claim_scope.json`, validates an
existing `ablation_report.json` or builds one from `ABLATION_FULL_DIR`,
`ABLATION_METABOLISM_OFF_DIR`, `ABLATION_REGIONS_OFF_DIR`, `ABLATION_FORGETTING_OFF_DIR`, and
`ABLATION_AFFECT_OFF_DIR`,
`affect_salience_invariant.json`, `scratchpad_invariant.json`,
`region_routing_invariant.json`, `reflex_recall_invariant.json`,
`smqe_planner_invariant.json`, `smqe_synthetic_invariant.json`,
`smqe_claim_coverage.json`, `smqe_fullpath_invariant.json`,
`smqe_paraphrase_invariant.json`, `smqe_conflict_invariant.json`,
`smqe_composition_invariant.json`, `smqe_relative_phrase_invariant.json`,
`smqe_temporal_window_invariant.json`, `smqe_attribution_invariant.json`,
`smqe_abstention_invariant.json`,
`smqe_scope_invariant.json`, `smqe_subscope_invariant.json`,
`smqe_time_invariant.json`, `smqe_invalidation_invariant.json`, `slice_invariant.json`, optional `mem0_gate.md/.json` when the published
Mem0 reference file exists, and `release_gate.md/.json` inside the same `OUT` directory.
Manual `bench.release_gate` runs are still useful when inspecting or re-rendering an existing
artifact.
Invariant sidecar failures continue through to `release_gate.md/.json` so failed public-claim
evidence is inspectable instead of vanishing as an early shell abort.

It reads raw JSONL plus `run_manifest.json` and fails unless the artifact proves held-out `--split test`,
at least 10 runs, required systems and datasets, every canonical LongMemEval/LoCoMo category,
headline accuracy thresholds, narrow enough sample-clustered Wilson confidence intervals, paired
dominance with McNemar significance, enough independent sample-level discordants, CI-clear dominance
against each baseline, operating-point budgets (median query tokens, search p95, e2e p50, token
efficiency vs `rag-full`, age-flat recall), verified-recall integrity, passing rotating affect,
scratchpad, region-routing, reflex-recall, SMQE planner and synthetic invariants, claim-backed SMQE
coverage, benchmark logs where at least 80% of integrity rows use
structured recall and at least 80% of structured rows are claim-backed tier-1 recall,
full adapter-path SMQE recall with zero fixed-reader calls,
paraphrase robustness on both record and claim backends,
temporal-conflict robustness for changed current values, unsupported-question abstention on both
record and claim backends, multi-record composition for shared values and event chains,
source-relative phrase normalization, temporal-window aggregate filtering, namespace and
actor/speaker attribution under negated same-target distractors,
agent/project sub-scope isolation under conflicting same-question evidence,
as-of time isolation under future-dated conflicting evidence,
invalidated-memory exclusion, executed multi-draw
slice-invariant sidecars with distinct sample sets, 100% snap-back fidelity over the benchmark store, a dev-split calibration report for
`ABSTENTION_V2_TAU` whenever `ABSTENTION_V2=1`, a passing Mem0 reproduction report (`mem0_gate.json`),
raw-log fingerprints that prove rendered reports are fresh, and healthy consolidation with no hidden
timeout or deferred extraction debt. SOTA or "best memory agent" wording additionally requires
structured evidence for the named top comparators (`chronos`, `mastra`, `byterover`, `hindsight`)
through `claim_scope.json`.
A failed release gate means the artifact is still engineering evidence, not a public claim.

---

## Outputs

All written to `--out` (default `artifacts/bench/`), and all rendered **only from real logs**:

- `scoreboard.md` (+ `scoreboard.json`): per-category accuracy (mean ± std), cost table
  (tokens/write per conversation, tokens/query), latency table (search/e2e p50/p95),
  verified-recall integrity, and consolidation health (pending processed, facts/events extracted,
  extraction timeouts, deferred records). Records the judge used and the raw-log fingerprint.
- `run_manifest.json`: systems, dataset slice, run offsets, judge, and feature-flag environment.
- `stage_ab.md` (+ `stage_ab.json`) inside each non-control sweep trial directory: raw-log A/B deltas
  against that stage's control value, including n, tokens, latency, paired rows, McNemar p-values,
  and consolidation health deltas.
- `release_gate.md` (+ `release_gate.json`): final public-claim eligibility check over real logs,
  manifest, paired dominance, operating budgets, integrity, snap-back fidelity, and consolidation
  health. Includes the raw-log fingerprint it checked.
- `bench.audit_no_holdout_leakage`: fail-closed source audit that rejects leaked holdout strings,
  legacy benchmark-rescue symbols/policies, fixed benchmark answer literals, and empty holdout
  registries.
- `bench.build_holdout_registry`: writes the non-answer-bearing registry consumed by the audit from
  a real holdout `--samples-file` or split. For public claims, populate `data/bench/holdout/` from a
  private holdout list before running `bench/reproduce.sh`; public fixture IDs are not evidence of
  no overfitting.
- `affect_salience_invariant.json`: cheap rotating proof that salience boosts memory priority without
  age leakage and stays bounded by the configured boost ratio.
- `scratchpad_invariant.json`: cheap rotating proof that tiny high-salience context contains only
  active, scoped, proof-linked facts and also surfaces through retrieval.
- `region_routing_invariant.json`: cheap rotating proof that cocoon/region hints recover dense misses
  while preserving active scope filters, proof links, and recall trace telemetry.
- `reflex_recall_invariant.json`: cheap rotating proof that the local reflex packet finds direct and
  coactivated memories, excludes inactive/out-of-scope distractors, preserves proof links, and stays
  inside a sub-second latency budget.
- `smqe_planner_invariant.json`: cheap rotating proof that the rules-first SMQE planner maps invented
  query shapes to generic operators, keeps terms/entities/slots question-derived, preserves synthesis
  and temporal-unit metadata, and avoids benchmark rescue policy strings.
- `smqe_synthetic_invariant.json`: cheap rotating synthetic SMQE proof that fresh latest/count/date/
  table/preference/inference/speaker/delta/sum cases answer correctly with verified citations and low
  proof-token cost.
- `smqe_claim_coverage.json`: cheap rotating proof that the same operator families are answerable
  from source-backed extracted claims, so tier-1 recall coverage stays visible.
- `smqe_fullpath_invariant.json`: cheap rotating proof that invented conversations survive the real
  `eidetic-plus-full` ingest -> consolidate -> answer path with verified SMQE policies, claim-backed
  citations, zero fixed-reader calls, and bounded proof-token cost.
- `smqe_paraphrase_invariant.json`: cheap rotating proof that paraphrased questions and evidence
  pass through both source-backed claim and record operators with verified citations.
- `smqe_conflict_invariant.json`: cheap rotating proof that changed current values choose the active
  latest support and ignore stale or future-intent proof on both backends.
- `smqe_composition_invariant.json`: cheap rotating proof that multi-record joins for shared
  values, event ordering, and event-relative clock lookup are verified on both backends.
- `smqe_relative_phrase_invariant.json`: cheap rotating proof that source-relative phrases such as
  `two weeks ago`, `a fortnight ago`, and `next month` normalize from memory timestamps on both backends.
- `smqe_temporal_window_invariant.json`: cheap rotating proof that rolling windows such as
  `recently`, `past week`, `past N days`, `past few months`, and `fortnight` constrain count/sum
  evidence before aggregation, plural list answers, most-recent object lookup, and source/location
  slot lookup, including `pick up at` / `bought at` / `purchased from` phrasing, on both backends.
- `smqe_attribution_invariant.json`: cheap rotating proof that `who recommended/gave/told/shared`
  questions return the correct actor from verified evidence and skip negated same-target distractors
  on both backends.
- `smqe_abstention_invariant.json`: cheap rotating false-positive guard that lexically close but
  unsupported questions abstain on both record-only and claims-present backends.
- `smqe_scope_invariant.json`: cheap rotating isolation proof that identical questions over
  conflicting scopes return only the requested scope's answer and proof on both backends.
- `smqe_subscope_invariant.json`: cheap rotating isolation proof that identical questions over
  conflicting `agent_id`/`project_id` sub-scopes inside one namespace do not bleed across backends.
- `smqe_time_invariant.json`: cheap rotating as-of proof that future-dated memories do not leak
  into earlier answers, while later answers can use the newer evidence on both backends.
- `smqe_invalidation_invariant.json`: cheap rotating proof that records and claims with
  `invalid_at` stop supporting answers after invalidation on both backends.
- All rotating sidecars (`affect_salience_invariant.json`, `scratchpad_invariant.json`,
  `region_routing_invariant.json`, `reflex_recall_invariant.json`, every
  `smqe_*_invariant.json`, and `smqe_claim_coverage.json`) must carry `seed_mode: "random"` for
  release eligibility. Passing a fixed `--seed` is useful for replaying a failure, but fixed-seed
  sidecars are not accepted as public evidence.
- `slice_invariant.json`: executed repeated-draw benchmark sidecar. Release eligibility requires
  `split: "test"`, every draw sample ID hashing to the test split, `holdout_profile: "holdout"`,
  enough distinct draws, distinct recorded draw seeds, `seed_mode: "random"`, enough samples per
  draw with unique IDs inside each draw, enough unique sample IDs across the union of draws
  (`draws * subset`), and perfect
  `verified_correct` scores from scoreboard integrity for the system under test on each required dataset. `--plan-only` writes sample files but records `pass: false`;
  it is not evidence. If the loaded split has fewer unique samples than `draws * subset`, the runner
  writes a nonpassing sidecar and launches no benchmark subprocesses. Replays can pin
  `SLICE_INVARIANT_SEED`, but fixed-seed sidecars are not release-eligible because they are too easy
  to tune against.
- `mem0_gate.md` (+ `mem0_gate.json`): external-baseline reproduction check against the explicit
  published Mem0 reference file, bound to the raw-log fingerprint.
- `snap_back_audit.json`: machine-readable substrate fidelity audit; release eligibility requires
  every raw-backed memory to satisfy `sha256(get_raw(content_hash)) == content_hash`.
- `recall_vs_age.png`: accuracy vs evidence age (flat = age-independent).
- `latency_vs_age.png`: p95 end-to-end latency vs evidence age.

With **no logs**, the scoreboard writes an explicit **"pending run"** placeholder and the curves write
a note, never invented numbers.

---

## Honest status

The **harness spine, all three adapters (Eidetic-Plus, Mem0, Graphiti), and both dataset loaders**
are built and **offline-verified** (LoCoMo verified at 1540 validated-category questions). The
**Mem0 and Graphiti adapters** are real and fail loud on a missing library / key / (for Graphiti)
unset `NEO4J_*`. They never mock. They have not been *run end-to-end here* because that needs the
baseline packages installed, a Neo4j instance, a funded key, and significant compute.

A **populated** scoreboard requires:

- a **funded** `DASHSCOPE_API_KEY`,
- the **baselines installed** (`requirements-bench.txt`),
- for Graphiti, a **running Neo4j** instance,
- the **datasets downloaded**, and
- **significant compute**: full LongMemEval_S + LoCoMo × 3 systems × `>= 10` runs is **millions of
  tokens / hours**.

The numbers **populate when you run it**. We never fabricate a score.

---

## Targets to beat (from the spec)

| system | build cost | query cost | search p95 | notes |
|---|---|---|---|---|
| **Mem0** | ~7–14k tok/conv | < 7k tok | 0.2s | add()-time LLM fact extraction |
| **Graphiti** | > 600k tok/conv ingest | n/a | n/a | post-ingestion lag |
| **Eidetic-Plus (target)** | **<= Mem0** (LLM-free write path) | **~5–9k tok** | **< 0.2s** | ingestion **< 1s** |

---

## Tuning: sweep + calibrate (parameterized, not hardcoded)

The optimization playbook's knobs are `eidetic/config.py` parameters with safe defaults, tuned on a
subset **after** a key is added (hardcoding a tuned value before a run is guessing):
`reader_cot_enabled`, `conflict_resolver_enabled`, `context_compress_enabled`,
`extract_light_enabled`, `temporal_rerank_enabled`, `hippo2_seeding_enabled`, `persistent_bm25_enabled`,
`rrf_w_dense/bm25/graph/recency`, `rerank_enabled`, `rerank_depth`, `hnsw_ef_search`, `compression_ratio`,
`context_token_budget`, and `salience_prune_threshold`.

```bash
# Enumerate the coordinate-descent sweep plan + token-cost estimate (OFFLINE, no scoring):
python -m bench.sweep --dry-run --subset 50

# Live sweep (needs a funded key): benchmark-visible flags first, then retrieval knobs:
python -m bench.sweep --systems eidetic --dataset locomo --subset 50 --sample-offset 0 \
  --mem0-gate-out artifacts/gate-mem0-qwen-plus \
  --mem0-gate-expected data/bench/mem0_locomo_published.json

# Fresh-slice survival check after a winning config:
python -m bench.run --systems eidetic,rag-full,rag-vector,mem0,graphiti --dataset locomo --subset 50 --sample-offset 50 --runs 1 --run-offset 1
python -m bench.run --render-only --out artifacts/bench

# Flag A/B comparison from real logs only:
python -m bench.compare --control artifacts/control --experiment artifacts/reader-cot --system eidetic-plus --out artifacts/reader-cot-vs-control.md

# Calibrate the abstention threshold from REAL dev-split scored logs (~95% precision target):
python -m bench.calibrate --logs artifacts/cal_dev --system eidetic-plus-full --method v2 --target 0.95

# Publish the calibrated threshold into the public artifact and source the exact env before scoring:
python -m bench.calibration_handoff --calibration artifacts/cal_dev/abstention_v2_tau.json --out artifacts/bench --env-out artifacts/guard/abstention.env
set -a; source artifacts/guard/abstention.env; set +a
```

`sweep --dry-run` and `calibrate` on a fixture run offline and never fabricate a score; live scoring
requires the key. The recommended order once the key lands: **`bench.sweep` (subset) → `bench.calibrate`
→ `bench.calibration_handoff` → full `bench/reproduce.sh`**. `scripts/dominance_run.sh --full`
performs the calibration handoff automatically before it launches the public reproduce run.

---

## BEAM frontier note (honest)

Cross-session **contradiction resolution at BEAM scale (1M -> 10M tokens)** is the **unsolved
frontier** across the field. The best public BEAM-1M contradiction score is **~0.357**. Eidetic-Plus
targets the LongMemEval + LoCoMo categories above; BEAM-10M contradiction is presented as the
**frontier, not a solved box**.
