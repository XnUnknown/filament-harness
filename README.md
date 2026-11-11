# Filament Harness

An agentic coding harness for local and hosted LLMs. Filament runs a real
tool-calling loop, keeps a bounded context window, and gives every subagent its
own session and transcript. It ships with a terminal UI, an interactive CLI, and
an Electron desktop app.

Python 3.11+.

## What it does

- **Agent loop** — think, call tools, observe results, repeat, until the task is
  finished, the agent asks a question, you cancel, or a step limit is reached.
- **Flexible tools** — read, exact edit, multi-edit, write, glob, grep, shell,
  Python syntax check, todo list, user questions, subagents, and an explicit
  `finish`.
- **Subagent sessions** — each subagent has its own message history, budget, and
  session file. It returns a summary to the parent, and the parent decides what
  to do next.
- **Bounded context** — large tool results are offloaded, old tool output is
  pruned, and old turns are compacted into a dense summary at a safe turn
  boundary.
- **Questions, not guesses** — the `ask_user` tool pauses the loop and resumes
  when the user answers, with a native question card in the UI.
- **Interrupt-safe history** — Stop abandons a partial model turn without
  running its tools, and a session interrupted mid-tool receives the missing
  tool result before the next request, so no provider sees an unanswered call.
- **Streaming** — OpenAI-compatible, Anthropic, and Ollama responses stream to
  the UI with tool-call fragments assembled across chunks.
- **Plan mode** — read-only exploration and a concrete plan before any writes.

## Quickstart

```bash
pip install -e .
filament
```

`filament` with no subcommand starts the interactive agent in the current
directory. Use `--client mock` for an offline smoke test:

```bash
filament --client mock
```

Configure a provider in `filament.toml`:

```toml
[llm]
kind = "ollama"          # auto | mock | ollama | openai | anthropic
model = "qwen2.5-coder:7b"
base_url = "http://localhost:11434"
stream = true
```

Or use environment variables: `FILAMENT_CLIENT`, `FILAMENT_MODEL`,
`FILAMENT_OLLAMA_URL`, `FILAMENT_API_KEY`, `FILAMENT_MAX_TOKENS`.

Hosted OpenAI and Anthropic endpoints require an API key. Localhost and
loopback endpoints are exempt so LM Studio, vLLM, and local Ollama work without
one.

## Interfaces

### Interactive CLI

```bash
filament
filament --mode plan
filament --shell-cwd ../my-project
```

Commands: `/new`, `/sessions`, `/load <id>`, `/mode build|plan`, `/compress`,
`/todos`, `/context`, `/help`, `/quit`.

### Terminal UI

```bash
filament tui
# or
filament-tui
```

The TUI keeps the settings menus (provider, context, execution, paths) and adds
the same agent chat screen as the CLI.

### Electron app

```bash
cd electron
npm install
npm start
```

The desktop app shows streaming assistant text, collapsible tool cards, nested
subagent transcripts, the plan, a context meter, changed files, session history,
question cards, build/plan modes, and a settings panel. It supports a dark and a
light theme.

`FILAMENT_PYTHON` selects the Python interpreter used for the backend.

### JSON backend

`python -m filament.backend_server` speaks one JSON object per line:

```json
{"id": 1, "method": "chat", "params": {"message": "fix the tests", "mode": "build"}}
{"id": 1, "ok": true, "result": {"status": "done", "content": "..."}}
```

While a turn runs, the backend streams notifications with an `event` key and no
request id:

```json
{"event": "delta", "session_id": "...", "text": "Running the suite."}
{"event": "tool_start", "session_id": "...", "id": "call_1", "name": "run_shell", "args": {"command": "pytest"}}
```

`chat` and `answer` run on worker threads, so `cancel` and other requests are
handled while the agent is working.

## Tools

| Tool | Purpose |
|---|---|
| `read_file` | Line-numbered read with offset/limit and a next-offset footer |
| `write_file` | Create a file or replace its full contents |
| `edit_file` | Exact string replacement, unique by default, whitespace-normalized fallback |
| `multi_edit` | Several exact edits to one file in a single atomic write |
| `list_dir` | Directory listing with sizes and ignored directories removed |
| `glob` | Find files by glob pattern, newest first |
| `grep` | Regex content search with context, file globs, and limits |
| `run_shell` | Command execution with timeout, allowlist, and output offload |
| `check_python` | Compile a Python file without running it |
| `todo_write` | Maintain the visible plan; one item in progress at a time |
| `ask_user` | Pause the loop and wait for a human decision |
| `spawn_agent` | Run one subagent with its own session and return its summary |
| `spawn_agents` | Run several subagents in parallel |
| `finish` | End the run with a summary of what changed and how it was verified |

Every path is confined to the workspace unless `allow_outside_target` is set.

## Subagents

A subagent is a full second `AgentLoop`:

- new session file under `.filament/sessions` with `kind = "subagent"` and a
  `parent_id`;
- its own message history and `subagent_max_tokens` budget;
- no `ask_user` tool and no further nesting by default (`max_depth = 1`);
- its final answer becomes an ordinary tool result in the parent's transcript,
  and the parent continues its loop.

The Electron UI renders the child's events inside a collapsible subagent card,
so you can watch both agents work without mixing their context.

## Context management

- Tokens are estimated as characters divided by `chars_per_token` (default 4).
- Tool results over `tool_output_limit` characters are saved under
  `.filament/runs` and replaced with a head/tail excerpt.
- Before a model call, context above `compact_threshold` is pruned and then
  compacted. Compaction only cuts at a user turn, never between an assistant
  tool call and its result, and replaces old turns with a summary.
- If `compact_with_llm` is enabled, the summary is written by the model; if that
  call fails, a deterministic digest is used.

## Configuration

Copy `filament.toml.example` to `filament.toml`.

| Setting | Default | Purpose |
|---|---|---|
| `max_tokens` | `120000` | Context budget |
| `budget_margin` | `0.05` | Reserved headroom |
| `chars_per_token` | `4` | Token estimate |
| `max_steps` | `40` | Main agent loop limit |
| `subagent_max_steps` | `24` | Subagent loop limit |
| `subagent_max_tokens` | `32000` | Subagent context budget |
| `max_depth` | `1` | Subagent nesting limit |
| `tool_output_limit` | `12000` | Chars kept inline per tool result |
| `keep_recent_messages` | `8` | Recent messages preserved during compaction |
| `compact_threshold` | `0.75` | Fraction of the budget that triggers compaction |
| `compact_with_llm` | `true` | Use the model to summarize old turns |
| `timeout_shell` | `60` | Shell timeout in seconds |
| `max_shell_output` | `20000` | Inline shell output before offload |
| `use_shell` | `true` | Use shell mode; `false` splits commands into argv |
| `allow_outside_target` | `false` | Permit paths outside the workspace |
| `atomic_writes` | `true` | Write via temp file and atomic replace |
| `offload_shell_output` | `true` | Save large output under `.filament/runs` |
| `command_allowlist` | `[]` | Optional command prefixes |
| `llm.kind` | `auto` | `auto`, `mock`, `ollama`, `openai`, `anthropic` |
| `llm.model` | `llama3.2` | Model name |
| `llm.base_url` | provider default | API base URL |
| `llm.timeout` | `120` | Request timeout in seconds |
| `llm.stream` | `true` | Stream responses |
| `llm.max_output_tokens` | `4096` | Per-response output cap |
| `llm.reasoning_effort` | unset | `low`, `medium`, or `high` where supported |

Configuration is type-checked on load. A quoted `"false"` is rejected rather
than treated as true, and every bad key is reported in one error.

## Legacy batch mode

The original one-shot task runner is still available for scripted jobs:

```bash
filament run examples/demo_tasks.json --client mock
```

It spawns one completion per task, appends the extracted code in a batch, and
optionally runs a line-level review pass. New work should use the agent loop.

## Testing

```bash
python -m unittest discover -s tests -v
```

The suite covers the agent loop, tool safety, exact editing, subagent sessions,
question pause/resume, compaction boundaries, streaming parsers for all three
providers, the backend event protocol, configuration validation, and the legacy
batch path.

## Project layout

```text
filament/
  agent_loop.py     agent state machine, subagents, events
  toolkit.py        workspace-scoped tools
  compaction.py     token estimates, pruning, summaries
  stream.py         SSE / NDJSON parsing and tool-call assembly
  llm.py            mock, Ollama, OpenAI-compatible, Anthropic clients
  chat.py           persistent chat session wrapper
  session.py        session persistence and parent links
  backend_server.py JSON backend for Electron
  console.py        terminal event renderer
  cli.py            command line entry point
  tui.py            terminal menus and chat
  config.py         configuration and validation
  orchestrator.py   legacy batch runner
electron/
  main.js           window, backend process, IPC
  preload.js        narrow contextBridge API
  renderer/         event-driven UI
tests/
```

## Security notes

- The agent can only touch paths inside the workspace unless explicitly allowed.
- Shell commands can be restricted with `command_allowlist`.
- The Electron renderer runs with `contextIsolation`, no Node integration, and a
  narrow preload API; the backend is a separate process.
- API keys are read from configuration or environment and are never written to
  session files.

## Limitations

- Token counts are character-based estimates, not tokenizer output.
- Local models vary in tool-calling quality; the JSON action fallback helps but
  is not a substitute for a model with native tool support.
- Compaction is lossy by design. Keep important decisions in the conversation or
  in files.
- The harness is not a sandbox. Review the workspace and the allowlist before
  running untrusted prompts.
