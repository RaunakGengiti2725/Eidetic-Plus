# Fast measurement loop — honest numbers without 2.5h waits

All commands run from repo root with the venv python. Dev split only; the holdout
session owns `--split test`.

## Smoke (≈3 min, 5 questions)

```bash
.venv/bin/python -m bench.run --systems eidetic-full --dataset locomo \
  --subset 5 --split dev --runs 1 --out artifacts/dev_smoke_$(date +%H%M) --overwrite
```

## Regression: mixed-24 (the wave-F dev set, ≈15 min)

```bash
.venv/bin/python -m bench.run --systems eidetic-full \
  --samples-file artifacts/wave_i_mixed24_full_codex/dev_ablation.samples.json \
  --split dev --runs 1 --out artifacts/dev_mixed24_$(date +%H%M) --overwrite
```

Env: replicate the profile from `artifacts/wave_i_mixed24_full_codex/inherited_env.json`
(the launcher scripts show the pattern: load JSON, overlay os.environ, fresh DATA_DIR).

## Fix probe (2–6 rows, ≈5 min)

Write a samples-file with dev rows matching the SHAPE under test (never holdout IDs):

```bash
cat > /tmp/probe.samples.json <<'EOF'
[{"dataset": "locomo", "sample_id": "<dev-split id from bench.datasets.split_of>"}]
EOF
.venv/bin/python -m bench.run --systems eidetic-full --samples-file /tmp/probe.samples.json \
  --split dev --runs 1 --out artifacts/dev_probe_$(date +%H%M) --overwrite
```

Dev membership check: `from bench.datasets import split_of; split_of(sid) == "dev"`.

## Offline store replay (0 API cost — always FIRST)

Copy a run's `data/eidetic.sqlite` to the scratchpad, map jsonl rows → namespaces in
first-appearance order (`eidetic-plus-full-locomo-g{i}-r0`), then
`execute_plan(plan_query(q), q, records=..., claims=...)`. Execute-only (no verify) —
form floors live in verify, so junk visible here can still be refused live.

## Two-phase holdout (for the holdout session; cuts critical path ~90 min)

Phase A now: same launcher with `--systems eidetic-full` → forensics start at +40 min.
Phase B overnight: same launcher, `--systems mem0`, SAME `--out` and samples file
(per-system jsonl merge; `--render-only` rebuilds the scoreboard once both exist).
Pin the SHA in launch.log for each phase.

## Miss taxonomy (any run jsonl → forensics draft)

```bash
.venv/bin/python bench/miss_taxonomy.py artifacts/<run_dir>/eidetic-plus-full__run0.jsonl
```

Buckets: abstained / verified-wrong / unverified-wrong / error, per category, with a
markdown table ready to paste into forensics. Dev jsonl only from this branch.

## Form-floor matrix (before any verify.py change)

Replay `reader_answer_form_credible` over every past window's jsonl: kills may rise,
`flips` (correct answers refused) must print empty for EVERY window. The matrix caught
a would-be flip (quoted-name shape) before commit — run it every time.
