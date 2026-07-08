#!/usr/bin/env bash
# One-command NotebookLM free-path runner. Does EVERYTHING except the one step only you
# can do: `nlm login` (signing into YOUR Google account in YOUR browser). This script never
# sees, stores, or transmits your cookies/tokens -- nlm keeps the session locally.
#
#   scripts/notebooklm_free_setup.sh <namespace> ["Notebook name"] ["a test question"]
#
# It: (1) checks the nlm tool + your login, (2) creates/reuses a notebook, (3) exports
# eidetic's verified claim graph into it, (4) asks a question through the free Gemini read
# and prints the 0-caller-token answer + provenance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
NS="${1:-default}"
NB_NAME="${2:-eidetic-$NS}"
QUESTION="${3:-What do you remember about me?}"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

say "1/5  Is the nlm tool installed?"
# Prefer this repo's venv (Homebrew Python is externally-managed, so plain pip is blocked).
if [ -x "$ROOT/.venv/bin/nlm" ]; then NLM="$ROOT/.venv/bin/nlm"
elif command -v nlm >/dev/null 2>&1; then NLM="$(command -v nlm)"
else NLM=""; fi
if [ -z "$NLM" ]; then
  cat <<EOF
nlm is NOT installed. The PyPI package is 'notebooklm-mcp-cli' (it gives you the \`nlm\`
command). Install it into THIS repo's venv (no system pollution, no PEP-668 error):
  "$ROOT/.venv/bin/pip" install notebooklm-mcp-cli
(or globally: brew install pipx && pipx install notebooklm-mcp-cli)
Then re-run this script.
EOF
  exit 1
fi
echo "nlm found: $NLM"

say "2/5  Are you logged into your Google account?  (this is the ONLY step I can't do)"
if ! "$NLM" login --check >/dev/null 2>&1; then
  cat <<EOF
Not logged in. Run this ONE command yourself (opens YOUR browser, YOUR account):
  "$NLM" login
Then re-run this script. I never touch your credentials.
EOF
  exit 1
fi
echo "logged in."

say "3/5  Create or reuse the notebook \"$NB_NAME\""
# Robust + unkillable: recursively find any notebook id in whatever JSON shape nlm emits
# (array, {notebooks:[...]}, {notebook_id:...}, nested). set +e so a parse hiccup never
# silently aborts the script -- if it truly can't find an id we print raw output to diagnose.
set +e
_find_id() { "$PY" -m eidetic.integrations.notebooklm find-notebook-id --title "${1:-}"; }
LIST_JSON="$("$NLM" notebook list --json 2>/dev/null)"
NB_ID="$(printf '%s' "$LIST_JSON" | _find_id "$NB_NAME")"
if [ -z "${NB_ID:-}" ]; then
  CREATE_JSON="$("$NLM" notebook create "$NB_NAME" --json 2>/dev/null)"
  NB_ID="$(printf '%s' "$CREATE_JSON" | _find_id "")"
  [ -z "${NB_ID:-}" ] && NB_ID="$("$NLM" notebook list --json 2>/dev/null | _find_id "$NB_NAME")"
fi
set -e
if [ -z "${NB_ID:-}" ]; then
  echo "Could not auto-detect a notebook id. Paste me the output of this (safe, no secrets):"
  echo "    $NLM notebook list --json"
  echo "--- what it returned this run (first 400 chars) ---"
  printf '%s' "${LIST_JSON:-<empty>}" | head -c 400; echo
  exit 1
fi
echo "notebook id: $NB_ID"
export NLM_BIN="$NLM"                                   # module CliBackend uses the SAME nlm
export DATA_DIR="${DATA_DIR:-$HOME/.eidetic-plus/data}" # read your LIVE store, not an empty one
echo "reading eidetic store: $DATA_DIR"

# Optional: SEED=1 populates the namespace with sample linked facts first (real ingest +
# consolidate -> graph edges), so the loop works end-to-end even with an empty store.
if [ "${SEED:-0}" = "1" ]; then
  say "seed  Populating \"$NS\" with sample facts (uses your DASHSCOPE key)"
  "$PY" -m eidetic.integrations.notebooklm seed --namespace "$NS" --data-dir "$DATA_DIR"
fi

say "4/5  Export eidetic's VERIFIED memory into the notebook (free)"
"$PY" -m eidetic.integrations.notebooklm export-graph \
  --namespace "$NS" --notebook-id "$NB_ID" --backend cli --data-dir "$DATA_DIR"

say "5/5  Ask through the FREE Gemini read -> 0 caller tokens + provenance"
"$PY" -m eidetic.integrations.notebooklm routed-answer \
  --namespace "$NS" --notebook-id "$NB_ID" --backend cli --data-dir "$DATA_DIR" --question "$QUESTION"

cat <<EOF

Done. The answer above came from NotebookLM's free tier (0 tokens on your metered model),
grounded in eidetic's verified graph, with provenance mapped back to content hashes.
Caveat: the answer is Gemini-side (not eidetic verify-or-abstain) -- for a gate-verified
cited answer, use eidetic's own recall/ask.
EOF
