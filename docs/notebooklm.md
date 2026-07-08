# NotebookLM bridge — free reads + provenance

`eidetic/integrations/notebooklm.py` connects eidetic's verified memory to Google
NotebookLM. Two modes, one idea: **eidetic supplies the trustworthy, provenance-tagged
sources; NotebookLM (Gemini) does the reading — on its free tier, that read costs zero
of *your* metered LLM tokens.**

## Why this is the token-efficient path

When you query a NotebookLM notebook, Google's Gemini generates the answer over your
uploaded sources. On a personal NotebookLM account that generation is **free** — it
does not draw down your DashScope / OpenAI / Anthropic keys. So an agent that answers
recall queries *through* NotebookLM pays **0 user LLM tokens per read**, while eidetic
keeps the part no one else has: every source carries a provenance header (content
hash + validity window + verified claims), and the answer's `eidetic:<id>` references
map back to immutable records.

Net: **free reads + provenance** — a genuinely cheap, trustworthy memory agent.

## The two modes

**1. Export (`export_namespace`)** — push a namespace's active memories into a notebook
as sources, each stamped with its eidetic provenance header:

```bash
# preview what would be exported (offline, no auth):
python -m eidetic.integrations.notebooklm preview --namespace raunak-main --limit 5

# push into a notebook (needs auth, see below):
python -m eidetic.integrations.notebooklm export \
    --namespace raunak-main --notebook-id <NOTEBOOK_ID> --backend cli
```

**2. Reader mode (`NotebookLMBridge.answer`)** — the 0-token recall path:

```python
from eidetic.integrations.notebooklm import NotebookLMBridge, CliBackend
bridge = NotebookLMBridge(engine, CliBackend())
out = bridge.answer("raunak-main", "Where did Priya move?", notebook_id)
# -> {"answer": ..., "provenance": [{content_sha256, valid_at, memory_id}...],
#     "cited_sources": {"cited": N, "confirmed_in_eidetic": M},
#     "grounding": {...deterministic quote-faithfulness + coverage...},
#     "user_llm_tokens": 0, "caveat": "..."}
```

### Deterministic grounding check (free, no model calls)

Every `answer()` runs three checks the caller can trust without spending a token:

1. **Citation existence** (`cited_sources`) — each `eidetic:<id>` token Gemini cited must
   resolve to a real immutable record; `confirmed_in_eidetic` counts the ones that do. A
   hallucinated citation is NOT confirmed.
2. **Quote faithfulness** (`grounding.quotes_*`) — each reference's `cited_text` is checked
   against the exported source bytes *rebuilt deterministically from the store*:
   whitespace-normalized substring ⇒ `verbatim`; content-token overlap ≥ 0.8 ⇒
   `high-overlap`; else `unmatched` — NotebookLM altered or fabricated the quote.
3. **Answer token coverage** (`grounding.answer_token_coverage`) — fraction of the answer's
   content tokens present in the exported text. Gemini's connective prose lowers it; read a
   low number as a flag to inspect, `quotes_unmatched > 0` as the strong signal.

**Honest label (verbatim from the code):** this is a deterministic *lexical* check — NOT
NLI and NOT eidetic's proof gate. It catches fabricated/altered quotes and alien answer
content; it cannot certify the reasoning. Live key-free demo with all fields populated:
`artifacts/public_ship/notebooklm_live_grounded_demo.md` (4/4 citations confirmed, 2/2
quotes grounded, 0 caller tokens, no metered key set in the environment).

## Preflight (`doctor`) — run this FIRST, before any live call

```bash
python -m eidetic.integrations.notebooklm doctor --backend cli         # personal free path
python -m eidetic.integrations.notebooklm doctor --backend enterprise --project-number <N>
```
`doctor` never touches your memory or the network beyond an auth check. It reports whether
the backend is reachable + logged in, and prints the **exact** commands/endpoints it will
run, so your first live attempt isn't a guess. The `cli` backend shells out to the real
`nlm` syntax: `nlm source add <notebook> --text "…"` and `nlm notebook query <notebook>
"…"` (pinned to [notebooklm-mcp-cli](https://github.com/jacob-bd/notebooklm-mcp-cli)); if
that tool changes its commands, update `CliBackend` (one place) and re-run `doctor`.

## Auth (you provide credentials; nothing is hardcoded, nothing is pasted into chat)

| backend | auth (set on YOUR machine) | account | stability |
|---|---|---|---|
| `enterprise` | GCP bearer token `NOTEBOOKLM_ACCESS_TOKEN` (`gcloud auth print-access-token`) + `NOTEBOOKLM_PROJECT_NUMBER` | NotebookLM **Enterprise** license (paid) | official, stable API |
| `cli` | `nlm login` (browser-cookie login, in your own browser) | **personal** Google account (free tier works) | undocumented internal APIs — may break; ToS-gray |

The free-read trick uses the `cli` backend (personal free account). The code never
stores or transmits your credentials anywhere except the backend you choose.

## Honest boundaries (so nobody over-claims it)

- **`user_llm_tokens: 0` means zero on *your* metered model** — Google still spends
  compute; it is free *to you* on the personal tier, not free globally, and subject to
  Google's quotas and Terms of Service.
- **A NotebookLM answer is Gemini-side and is NOT run through eidetic's verify-or-abstain
  proof gate.** It is a free, provenance-mapped answer, not a gate-verified one. For a
  cited, abstain-when-unsure answer, call `engine.recall()` — that path is what the
  benchmark measures.
- **This is a product-cost win, not a benchmark-table row.** The rotating-holdout
  benchmark compares every system through one fixed qwen reader so it measures *memory*,
  not *answerer*. NotebookLM is a different (Gemini) reader whose tokens are off that
  meter, so it cannot be an apples-to-apples line in that table. We report the free-read
  advantage as a product mode, separately and labeled.
- The `cli` backend depends on an unofficial tool; treat it as best-effort. The
  `enterprise` backend is the stable, supported path.

## Graph-native source (verified claim graph → one compact source)

Instead of pushing one source per raw memory, `build_graph_source` / `export_graph`
serialize eidetic's **verified claim graph** into a single, compact,
provenance-carrying source. It iterates raw `Edge` objects (never `build_nx`, which
drops `source_memory_id` / `valid_at` / `invalid_at` / `supersedes` and collapses
parallel edges); `node_features` is used only for hub ordering.

```bash
# offline preview: measure the compression ratio with zero network / auth
python -m eidetic.integrations.notebooklm preview-graph --namespace raunak-main
# push the graph as ONE additional source:
python -m eidetic.integrations.notebooklm export-graph \
    --namespace raunak-main --notebook-id <NOTEBOOK_ID> --backend cli
# or append it alongside per-record sources:
#   bridge.export_namespace(ns, nbk, include_graph=True)
```

### The four text regions

1. **HONESTY** — the four boundary labels verbatim (see below). Always first.
2. **PROVENANCE LEGEND** — one line per *distinct* referenced `source_memory_id`,
   hoisting the full 64-char `content_hash` once (`eidetic:<mid[:16]> sha256=… source=…
   valid_at=…`). This is the compaction lever: inline triples stay short because the
   hash lives here once. Built from `all_records`, so history tokens never dangle.
3. **ACTIVE FACTS** — entity blocks, hub-first (ordered by `degree` then `ppr` from
   `node_features`, else alphabetical). Each triple: `<relation> -> <dst>
   [eidetic:<mid[:16]> @<valid_at>]`. `--max-entities` truncates *after* ordering, so
   it drops leaves and keeps hubs.
4. **HISTORY** — superseded facts with their validity window and a correct successor
   pointer: `<src> <rel> <dst>  <valid_at>..<invalid_at>  (superseded by
   eidetic:<successor_mid[:16]>)`. `Edge.supersedes` is the *edge_id* of the closed
   predecessor and lives on the **successor**, so we resolve the pointer via a reverse
   index (`successor.supersedes == closed.edge_id`) and render the *successor's memory
   ref* — never `supersedes[:16]` (that is an edge_id → a dangling token). A closed edge
   with no successor renders bare `(superseded)`.

`_resolve_provenance` reads `all_records` (not `active_records_at`) so a Gemini answer
citing a *history* token still round-trips back to its immutable `content_hash`.

### Worked compression example (per-namespace, measured)

Synthetic 4-record fixture (verbose multi-turn bodies, 3 active triples + 1 superseded)
via `preview-graph`:

```
n_entities=1  n_relations=3  n_active=3  n_superseded=1
raw_record_chars=3984  serialized_chars=1550  compression_ratio≈2.57
```

The ratio is **measured per call and surfaced in `stats`, never asserted**. A sparse
graph (few facts over short records) can invert below 1 — read the number from
`preview-graph` for your own namespace; do not assume "always smaller."

## Router-aware answer path (`routed_answer`)

`routed_answer` routes one question across four tiers, cheapest-verified first:

| tier | path | caller LLM tokens | verified? | when |
|---|---|---|---|---|
| 0 | reflex pre-filter (no model call) | 0 | n/a | always (candidate cross-check only) |
| 1 | `structured_recall` (typed, verify-or-abstain) | ~6–85 (design-supplied) | **gate-verified** | answered+verified+immutable_proof and `confidence ≥ struct_tau` |
| 2 | free NotebookLM read | **0** | no (provenance-mapped) | *not* struct-ok **and** `require_gate_verification=False` |
| 3 | metered verified reader (`recall(prove=True)`) | ~4034 (design-supplied) | **gate-verified** | *not* struct-ok **and** `require_gate_verification=True` |

Because `structured_recall` is itself verify-or-abstain, the metered reader (Tier 3) is
reserved for `(require_gate_verification AND not struct-ok)` — a
verified-but-low-confidence answer under `require_gate_verification` **escalates** to
Tier 3 rather than falling through. Tier 2's return is labeled
`provenance_verb="provenance-mapped"`, `gate_verified=False`; Tiers 1/3 are
`"gate-verified"`, `gate_verified=True`. Every return carries the four honesty strings.

**Token math (formula only — no blended figure).** `blended = P_struct·c_struct +
P_nb·0 + P_metered·4034`, with `P_struct + P_nb + P_metered = 1`. The structured tier's
cost is now **measured** from the committed 6-window holdout logs (r9–r14, the
smqe-answered rows Tier 1 takes): `c_struct` median **20.5**, worst-case **146**, n=78 —
see `bench/notebooklm_cost.py`. The metered cost (`4034`) is the measured 6-window median.
The `P_*` hit-rate weights remain **unmeasured on an arbitrary query stream**, so we ship
per-tier costs and the formula, never a specific blended number. Consequence, stated
precisely: under free-read routing (`require_gate_verification=False`) every query costs
either a measured-band structured answer (≤146 observed) or a 0-caller-token free read —
both below mem0's measured median 381 and rag-vector's 1892. Accuracy on the free-read
tier is unmeasured. `struct_tau` is a calibration parameter validated on a dev split — not
a proven constant.

## Live free-tier collection harness (`bench/notebooklm_freetier_run.py`)

Collects, key-free, live free-tier answers for a committed holdout window: one notebook
per conversation namespace (same isolation as the benchmark), exports the verified claim
graph **plus** packed raw-record sources (the graph is compact but lossy — affect and
detail live in the records; packing respects the free tier's per-notebook source cap with
provenance headers intact inside the text), then records per question: the Gemini answer,
citation confirmation, quote grounding, latency, and a **prefix-tolerant containment
heuristic labeled NOT-the-judge**. `bench/notebooklm_freetier_report.py` aggregates it
into a labeled report and re-scores from stored answers. The jsonl is judge-ready: with a
funded key, `bench.judge` scores the same rows properly. Nothing from this harness is
merged into the benchmark scoreboard.

## Incremental content-hash sync (`IncrementalSync`)

`IncrementalSync(bridge, manifest_path).sync(namespace, notebook_id)` diffs on
`record.content_hash` (already stamped into every source header, so no read-back from
NotebookLM is needed). A per-`(namespace, notebook_id)` sidecar manifest records pushed
hashes; a re-run pushes 0 (idempotent). A changed fact is a **new** `content_hash` →
pushed as a **new** source.

**Append-only tradeoff (stated plainly):** matching eidetic's write-once ethos, the
superseded record's header already carries `invalidated_at`; we never mutate or delete a
prior source. Consequence: the NotebookLM source count grows **monotonically** — a
ceiling concern at scale. Pruning is a separate **opt-in** policy, deliberately **not**
the default.

## Honesty boundaries for the graph source (verbatim, non-negotiable)

Every graph source, router return, and sync note carries these four labels:

- **Provable + true:** ~0 tokens on the **caller's** metered model per recall (NotebookLM
  /Gemini free tier does the read), **with provenance** (content-hash-mapped). And: the
  verified claim graph is **often a more compact** source than raw turns — measured per
  call and surfaced in `stats`; it can invert below 1 on sparse graphs (see the compression
  example above), so read the number, don't assume "always smaller."
- **Must NOT imply:** free *globally* (Google spends compute); that a NotebookLM answer
  is eidetic-verify-or-abstain (it is Gemini-side, provenance-mapped, **not**
  gate-verified); that this is a **row in the fixed-qwen-reader benchmark table**
  (different, off-meter reader); any **"best / strongest / SOTA"** claim — that would
  need the ≥10-run reproduce gate + named-comparator evidence, which is not built here.
