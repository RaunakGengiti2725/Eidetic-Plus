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
if ! command -v nlm >/dev/null 2>&1; then
  cat <<EOF
nlm is NOT installed. Install it once (community tool, not a Google product):
  https://github.com/jacob-bd/notebooklm-mcp-cli   (follow its README)
Then re-run this script.
EOF
  exit 1
fi
echo "nlm found: $(command -v nlm)"

say "2/5  Are you logged into your Google account?  (this is the ONLY step I can't do)"
if ! nlm login --check >/dev/null 2>&1; then
  cat <<EOF
Not logged in. Run this ONE command yourself (opens YOUR browser, YOUR account):
  nlm login
Then re-run this script. I never touch your credentials.
EOF
  exit 1
fi
echo "logged in."

say "3/5  Create or reuse the notebook \"$NB_NAME\""
NB_ID="$(nlm notebook list 2>/dev/null | awk -v n="$NB_NAME" 'index($0,n){print $1; exit}')"
if [ -z "${NB_ID:-}" ]; then
  nlm notebook create "$NB_NAME" >/dev/null 2>&1 || true
  NB_ID="$(nlm notebook list 2>/dev/null | awk -v n="$NB_NAME" 'index($0,n){print $1; exit}')"
fi
if [ -z "${NB_ID:-}" ]; then
  echo "Could not resolve a notebook id from \`nlm notebook list\`. Paste its output to me"
  echo "(safe, no secrets) and I'll adjust the parser."; exit 1
fi
echo "notebook id: $NB_ID"

say "4/5  Export eidetic's VERIFIED claim graph into the notebook (free)"
"$PY" -m eidetic.integrations.notebooklm export-graph \
  --namespace "$NS" --notebook-id "$NB_ID" --backend cli

say "5/5  Ask through the FREE Gemini read -> 0 caller tokens + provenance"
"$PY" -m eidetic.integrations.notebooklm routed-answer \
  --namespace "$NS" --notebook-id "$NB_ID" --backend cli --question "$QUESTION"

cat <<EOF

Done. The answer above came from NotebookLM's free tier (0 tokens on your metered model),
grounded in eidetic's verified graph, with provenance mapped back to content hashes.
Caveat: the answer is Gemini-side (not eidetic verify-or-abstain) -- for a gate-verified
cited answer, use eidetic's own recall/ask.
EOF
