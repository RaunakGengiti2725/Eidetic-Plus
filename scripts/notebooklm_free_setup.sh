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
# Robust: `nlm notebook list --json` gives structured {id,title}; match by title. If none,
# `nlm notebook create --json` returns the new id. Parse JSON with python (portable, no jq).
resolve_id() {
  "$NLM" notebook list --json 2>/dev/null | "$PY" - "$NB_NAME" <<'PYJSON'
import json,sys
name=sys.argv[1]
try: data=json.load(sys.stdin)
except Exception: sys.exit(0)
items=data if isinstance(data,list) else (data.get("notebooks") or data.get("items") or [])
for it in items:
    if isinstance(it,dict) and str(it.get("title","")).strip()==name:
        print(it.get("id") or it.get("notebook_id") or it.get("notebookId") or ""); break
PYJSON
}
NB_ID="$(resolve_id)"
if [ -z "${NB_ID:-}" ]; then
  # create it, parse the returned id from --json
  NB_ID="$("$NLM" notebook create "$NB_NAME" --json 2>/dev/null | "$PY" -c \
    'import json,sys;d=json.load(sys.stdin);print(d.get("id") or d.get("notebook_id") or d.get("notebookId") or "")' 2>/dev/null || true)"
  [ -z "${NB_ID:-}" ] && NB_ID="$(resolve_id)"   # fallback: re-list after create
fi
if [ -z "${NB_ID:-}" ]; then
  echo "Could not resolve a notebook id. Paste the output of:  \"$NLM\" notebook list --json"
  echo "(safe -- no secrets) and I'll fix the parser."; exit 1
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
