# Live grounded NotebookLM answer — 2026-07-07 (key-free run)

Raw output: [notebooklm_live_grounded_demo.json](notebooklm_live_grounded_demo.json).
Command (no DashScope key set in the environment — the whole read path is metered-model-free):

```
DATA_DIR=$HOME/.eidetic-plus/data python -m eidetic.integrations.notebooklm answer \
  --namespace demo --backend cli --notebook-id <NB> \
  --question "Where did I work, and what changed?"
```

## What the run proves (each with its exact label)

| property | result | basis |
|---|---|---|
| caller LLM tokens | **0** | by construction — Gemini free tier did the read; no metered key was even set |
| cited sources | **4 cited, 4/4 confirmed** | each `eidetic:<id>` resolved to a real immutable record's content_sha256 |
| quote faithfulness | **2/2 grounded (high-overlap), 0 unmatched** | deterministic lexical check vs exported bytes rebuilt from the store |
| answer token coverage | 0.514 | half the answer's content tokens are Gemini's connective prose — the metric flags Gemini-side additions honestly |
| temporal reasoning | answer includes the move-to-Berlin update + July-2023 adoption | from the verified claim graph's ACTIVE FACTS / HISTORY regions |

## What it does NOT prove

- The reasoning is Gemini-side — **NOT** run through eidetic's verify-or-abstain gate.
- The grounding check is lexical (substring + token overlap) — **NOT** NLI entailment.
- No accuracy number: a single demo question, no judge, not a benchmark row.
- No "best/strongest" claim — that still requires the ≥10-run reproduce gate + named
  comparators, which is not built.

The honest statement: **a free read (0 caller tokens) whose every citation resolved to a
content-hash-addressed immutable record and whose every quoted span matched the exported
source bytes.** That combination — free + checkable — is the product point.
