# Eidetic-Plus as an MCP plugin

Add persistent long-term memory to any MCP-compatible app (Claude Code, Claude Desktop, Cursor,
Cline, Windsurf, Zed, and others) with one config line. The MCP server is a thin door over the
exact same engine the rest of the project uses, so memory quality is identical to the direct
Python path and nothing is duplicated.

Honesty note, kept separate on purpose: being a plugin makes Eidetic-Plus usable everywhere. It
does not by itself make the memory more accurate than any other system. Memory quality comes
from the engine; the MCP layer is distribution, not accuracy. The server runs locally and stores
data locally unless you configure otherwise.

## Install and run (one command)

```bash
uvx eidetic-plus
# or
pipx run eidetic-plus
```

Both start the stdio MCP server with zero manual setup. You can also run it from a checkout:

```bash
pip install -e .            # editable install from the local repo
eidetic-plus                # the console entry point
python -m eidetic.mcp_server    # equivalent module path
```

The server starts and lists every tool even with no `DASHSCOPE_API_KEY`. The read-only tools
(`list_memories`, `get_raw`, `stats`) work without a key. Tools that need the model
(`remember`, `recall`) fail loud with an actionable message until you set the key. They never
fabricate a result and never silently no-op.

## Connect snippets

### Claude Code (CLI)

```bash
claude mcp add eidetic uvx eidetic-plus
```

Set the key and optional scope defaults in your environment, or with `--env`:

```bash
claude mcp add eidetic uvx eidetic-plus \
  --env DASHSCOPE_API_KEY=sk-your-key \
  --env EIDETIC_NAMESPACE=my-project
```

### Claude Desktop / Cursor / Cline (JSON config)

Add an `eidetic` server whose command is `uvx` with args `["eidetic-plus"]`:

```json
{
  "mcpServers": {
    "eidetic": {
      "command": "uvx",
      "args": ["eidetic-plus"],
      "env": {
        "DASHSCOPE_API_KEY": "sk-your-key",
        "EIDETIC_NAMESPACE": "default",
        "EIDETIC_AGENT_ID": "",
        "EIDETIC_PROJECT_ID": ""
      }
    }
  }
}
```

`DASHSCOPE_API_KEY` enables `remember` and `recall`. The three `EIDETIC_*` scope variables are
optional defaults; an explicit tool argument always wins over them.

### Any other MCP client (generic stdio)

Point the client at the command `uvx` with argument `eidetic-plus` (or `python -m
eidetic.mcp_server`) over stdio. For remote or shared use, run
`eidetic-plus --http --http-port 8765` (or the explicit form
`eidetic-plus --transport http --http-port 8765`).

## Verify it works

1. Start the server: `python -m eidetic.mcp_server` (or connect it in your app).
2. List tools: you should see `remember`, `recall`, `consolidate`, `list_memories`, `get_raw`,
   `forget`, `reawaken`, `stats`.
3. Call `remember` with `content` set to a fact, in a `namespace` you choose.
4. Call `recall` with a related `query` in the same `namespace`. The fact comes back with its
   cited immutable source. A `recall` in a different namespace returns nothing, which is the
   scope isolation guarantee.

## The tools

| Tool | What it does | Needs a key |
|---|---|---|
| `remember` | Store a durable memory in a scope (lossless, immutable). | yes (embeds the text) |
| `recall` | Verified retrieval with cited immutable sources, or an explicit abstention. | yes |
| `consolidate` | Token-free dreaming pass (replay, link inference, gist). Counts only. | no |
| `list_memories` | Paginated list of memories in a scope (read-only). | no |
| `get_raw` | The byte-identical immutable raw record by id, verbatim. Scope-filtered. | no |
| `forget` | Lower retrieval priority via FSRS. Never deletes the raw record. | no |
| `reawaken` | Re-promote a forgotten memory (the inverse of forget). | no |
| `stats` | Scope-level counts (memories, edges, vectors). | no |

## Scoping

Every tool takes `namespace` (default `"default"`) plus optional `agent_id` and `project_id`.
Scope resolves as: explicit tool argument, then the env defaults `EIDETIC_NAMESPACE` /
`EIDETIC_AGENT_ID` / `EIDETIC_PROJECT_ID`, then the safe global default. A read in namespace A
never returns a memory written in namespace B, so one server can serve many apps, users, and
projects without their memories colliding.

## Config changes need a restart

The server builds one long-lived engine at startup from config and environment. Changing engine
config (model ids, retrieval flags, and so on) means restarting the server, the same model the
rest of the codebase uses.

## Publishing (a later step you take, not done here)

This task does not publish to PyPI. To make `uvx eidetic-plus` resolve from PyPI later:

```bash
python -m build                 # builds sdist + wheel into dist/
python -m twine upload dist/*   # publish (requires a PyPI account + token)
```

Until then, install from the local checkout with `pip install -e .` and run `eidetic-plus`.
