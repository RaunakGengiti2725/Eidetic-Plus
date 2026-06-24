# Eidetic-Plus Neutral Benchmark Harness

> A number that doesn't reproduce doesn't exist.

This is **one neutral harness** that runs **Eidetic-Plus**, **Mem0**, and **Graphiti** through the
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
| `--systems` | `eidetic` | comma list of systems to run: `eidetic`, `mem0`, `graphiti` (any combination, e.g. `eidetic,mem0,graphiti`). |
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

## Full reproduce

The one-line command: both datasets, full sets, all three systems, `>= 10` runs for variance:

```bash
bash bench/reproduce.sh
```

This runs `--systems eidetic,mem0,graphiti --dataset both --subset 0 --runs 10 --out artifacts/bench`.
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

That pins **`mem0ai==2.0.7`**, **`graphiti-core==0.29.2`**, and **`neo4j==6.2.0`** (resolved cleanly on
CPython 3.14).

- **Graphiti needs a running Neo4j.** A free **Neo4j AuraDB** (cloud, **no Docker**) works; set
  `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.
- **Both baselines are configured to use DashScope**: LLM `qwen-plus`, embedder `text-embedding-v4`
  via the OpenAI-compatible endpoint, so **the same models back all three systems**. Each baseline
  drives its own retrieval, then hands the retrieved memories to the **one shared fixed reader**
  (`answer_with_fixed_reader`); baselines never use their own answer generator. A missing dependency,
  empty `DASHSCOPE_API_KEY`, or absent Neo4j **fails loud**.

---

## The judge

One **fixed judge** grades all three systems identically. That is the neutrality guarantee.

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

1. **One fixed judge + one fixed reader** across all three systems, by construction.
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
python -m bench.gate --out artifacts/bench50 --expected data/bench/mem0_locomo_published.json
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
exceed tolerance.

---

## Outputs

All written to `--out` (default `artifacts/bench/`), and all rendered **only from real logs**:

- `scoreboard.md` (+ `scoreboard.json`): per-category accuracy (mean ± std), cost table
  (tokens/write per conversation, tokens/query), and latency table (search/e2e p50/p95). Records the
  judge used.
- `run_manifest.json`: systems, dataset slice, run offsets, judge, and feature-flag environment.
- `stage_ab.md` (+ `stage_ab.json`) inside each non-control sweep trial directory: raw-log A/B deltas
  against that stage's control value, including n, tokens, latency, paired rows, and McNemar p-values.
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
python -m bench.run --systems eidetic,mem0,graphiti --dataset locomo --subset 50 --sample-offset 50 --runs 1 --run-offset 1
python -m bench.run --render-only --out artifacts/bench

# Flag A/B comparison from real logs only:
python -m bench.compare --control artifacts/control --experiment artifacts/reader-cot --system eidetic-plus --out artifacts/reader-cot-vs-control.md

# Calibrate the abstention threshold from REAL scored logs (conformal, ~95% precision target):
python -m bench.calibrate --logs artifacts/bench --target 0.95
```

`sweep --dry-run` and `calibrate` on a fixture run offline and never fabricate a score; live scoring
requires the key. The recommended order once the key lands: **`bench.sweep` (subset) → `bench.calibrate`
→ full `bench/reproduce.sh`**.

---

## BEAM frontier note (honest)

Cross-session **contradiction resolution at BEAM scale (1M -> 10M tokens)** is the **unsolved
frontier** across the field. The best public BEAM-1M contradiction score is **~0.357**. Eidetic-Plus
targets the LongMemEval + LoCoMo categories above; BEAM-10M contradiction is presented as the
**frontier, not a solved box**.
