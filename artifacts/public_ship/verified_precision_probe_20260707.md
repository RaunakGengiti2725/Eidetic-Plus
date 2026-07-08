# Verified-precision probe + form-gate fix — 2026-07-07

**All numbers here are from committed holdout logs + offline re-runs against each window's own
store (embeddings cached, so no live model key was needed). No fresh window was run — the
DashScope key is unset in this environment, so nothing below is a live-window confirmation.**

## 1. Honest 6-window aggregate (corrects a single-window overclaim)

Across **6 disjoint LoCoMo windows r9–r14** (n=240 each; sample_ids verified pairwise disjoint):

| system | accuracy | median caller-tokens/q | abstain |
|---|---|---|---|
| rag-full | 67.5% | 22499 | 0% |
| **rag-vector** | **58.8%** | **1892** | 0% |
| **eidetic-plus-full** | **53.3%** | **4030** | 8.3% |
| mem0 | 35.1% | 381 | 0% |

**eidetic-plus-full loses to rag-vector on both raw accuracy and caller-token cost** over the
disjoint set. The earlier "60% beats rag-vector 40%" was one window — cherry-picking. eidetic's
real differentiator is the `verified` (provenance) column, not raw accuracy.

## 2. The core-promise defect: verified-precision 56%

A verify-or-abstain agent's whole claim is "never confidently wrong." Of **199** rows eidetic
marked `verified=True` across the 6 windows, **112 were correct → verified-precision 56.3%**;
**87 shipped verified-WRONG.** Split by producer:

- **52 SMQE structured path** (deterministic, `answer_from_result`) — offline-reproducible.
- **35 fixed-reader path** — key-gated, not offline-addressable.

SMQE verified-wrong families: relative_temporal 20 (wrong-but-well-formed dates), latest_value
15, open_inference 7, preference_synth 5, count_aggregate 4, temporal_delta 1.

## 3. The fix: a general clean-fact form gate (small, safe, out-of-sample validated)

`_clean_fact_form_credible` in `eidetic/smqe/verify.py` downgrades `verified=True`→abstain for
two GENERAL malformed shapes on non-computed, non-polarity ops:

1. a first-person pronoun+verb opening carrying no factual anchor (quoted title / proper noun /
   digit) — a captured dialogue turn, not a fact;
2. a comma-list item in the `<Name>: <text>` dialogue turn-header shape — a captured turn, not a
   list value.

**Method (per the anti-overfitting discipline):** rules were designed looking ONLY at tune
windows r9–r11, then measured on holdout windows r12–r14 that were never inspected. The full set
of 26 verified-correct SMQE rows is the zero-regression guard.

| split | verified-wrong | verified-correct (must not drop) |
|---|---|---|
| tune r9–11 | 23 → 19 (**−4**) | 12 → 12 (0) |
| **holdout r12–14 (out-of-sample)** | 29 → 28 (**−1**) | 14 → 14 (**0**) |
| all 6 | 52 → 47 (−5) | 26 → 26 (0) |

An earlier `bigram_repeat` rule was **dropped**: it looked good on tune but broke legitimate
lists ("X shop and Y shop") and parallel timelines — an overfit patch, caught by the SMQE
invariant suite. Only the two rules that generalize survive.

## 4. What is NOT fixed (stated plainly)

The majority of verified-wrong is wrong-but-well-formed dates/counts/entities. Form cannot detect
these without ground truth (the correct answers include bare dates too), so they are **not**
offline-fixable — they need a live re-run through the reader, which is key-blocked here. This fix
makes the `verified` column *more* trustworthy on the malformed subset; it does not close the
raw-accuracy gap to rag-vector, and verified-wrong→abstain leaves accuracy flat while raising
abstention. For a verify-or-abstain agent that is the correct trade, but it is a narrow one.

## 5. Verification

- Leakage audit: PASS (0 findings, 1639 needles, 103k shingles).
- New synthetic-only regression test: `tests/test_smqe_clean_fact_form.py` (13 cases).
- SMQE invariant suite: green.
