# Filament Harness — Architecture and SRS

Version 0.5.4.

## 1. Purpose

Filament is a small agentic coding harness for local and hosted LLMs. It gives
a model a real tool surface and a bounded context window, runs subagents in
their own sessions, and exposes the whole loop through a terminal UI, an
interactive CLI, and an Electron desktop app.

The design target is a laptop-class workflow: a 7B–32B local model, a project
directory, and a context budget that stays near 32k–120k tokens instead of
growing without bound.

## 2. System Overview

```text
┌──────────────────────────── UI ────────────────────────────┐
│ Electron renderer │ TUI / CLI │ Python API │ JSON backend  │
└───────────────┬────────────────────────────────────────────┘
                │ events + requests
┌───────────────▼────────────────────────────────────────────┐
│ ChatOrchestrator                                           │
│  · owns one Session                                        │
│  · builds an AgentLoop per turn                            │
└───────────────┬────────────────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────────────────┐
│ AgentLoop                                                  │
│  · messages, steps, budget, cancel event, event sink       │
│  · tool registry (workspace-scoped)                        │
│  · compaction before each model call                       │
│  · subagent factory                                        │
└───────┬───────────────────────────────┬────────────────────┘
        │                               │
┌───────▼────────┐              ┌───────▼────────────────────┐
│ LLMClient      │              │ ToolRegistry               │
│ chat + stream  │              │ file / search / shell /    │
│ tool calls     │              │ todo / ask / spawn / finish│
└────────────────┘              └────────────────────────────┘
```

Core modules:

| Module | Responsibility |
|---|---|
| `agent_loop.py` | The loop, events, cancel, stall guard, subagent creation |
| `toolkit.py` | Tool schemas and handlers, path confinement, output limits |
| `llm.py` | Provider adapters and the neutral chat message format |
| `stream.py` | SSE / NDJSON parsing, tool-call fragment assembly |
| `compaction.py` | Token estimates, tool-result pruning, turn-boundary summaries |
| `session.py` | JSON session persistence, parent links, atomic writes, corrupt-file normalization |
| `chat.py` | Persistent chat wrapper, mode handling, session load/new |
| `backend_server.py` | Newline-delimited JSON server with streaming notifications |
| `console.py` | Dependency-free terminal event renderer |
| `cli.py`, `tui.py` | Entry points and interactive surfaces |
| `orchestrator.py` | Legacy one-shot batch runner |

## 3. Agent Loop

Each turn follows the same state machine:

```text
user message
   │
   ▼
compact context if over threshold
   │
   ▼
LLM chat(messages, tools)
   │           │
   │           ├── text deltas ──────────────► UI
   │           └── tool calls
   ▼
assistant message recorded
   │
   ├── no tool calls ───────────────────────► done
   │
   └── execute tools in order
         ├── normal result ─────────────────► tool message, continue
         ├── finish ────────────────────────► done with summary
         ├── ask_user ──────────────────────► awaiting_input, pause
         └── error ─────────────────────────► tool error, continue
```

Guarantees:

- Every assistant tool call receives a tool message before the next model call.
  When the loop pauses for a question or finishes mid-batch, remaining calls get
  an explicit "not executed" result. A session reloaded after an interruption
  is repaired the same way before the next request.
- Four identical tool-call batches in a row stop the loop with `stalled` rather
  than burning tokens.
- `cancel()` is checked before each model call; a stop during a stream ends the
  turn as `cancelled` without executing tools from the partial response.
- `max_steps` bounds the main loop and each subagent independently.

## 4. Tools

Tools are declared once in `ToolRegistry` with a JSON Schema and a handler.
They are filtered by mode and depth:

- plan mode exposes only read-only tools (`read_file`, `list_dir`, `glob`,
  `grep`, `check_python`, `todo_write`, `ask_user`, read-only subagents,
  `finish`);
- subagents do not receive `ask_user` or `spawn_agent` when the depth limit is
  reached.

Path handling is centralized in `ToolRegistry._resolve`: null bytes, control
characters, drive-relative paths, and paths outside the workspace are rejected
unless `allow_outside_target` is set.

Tool results are clipped at `tool_output_limit`; the full text is saved under
`.filament/runs` and the excerpt names the saved file.

## 5. Subagents

`spawn_agent(task, system?)` creates a new `Session` with:

- `kind = "subagent"`;
- `parent_id = <parent session>`;
- a title from the delegated task;
- its own `AgentLoop` with `subagent_max_tokens` and `subagent_max_steps`.

The child runs synchronously from the parent's perspective but streams events
tagged with `subagent_id`. When it finishes, the parent's tool result is the
child's final summary. Child files are merged into the parent's change list.

`spawn_agents(tasks[])` runs up to six children in a thread pool; each child has
an independent session and event stream.

`max_depth` defaults to 1, so a subagent cannot spawn another by default.

## 6. Context Management

Token estimate: `chars / chars_per_token`, with a hard minimum of 1 per message.

Before each model call:

1. If the estimate is under `compact_threshold × budget`, continue.
2. Otherwise prune large tool results older than `keep_recent_messages`.
3. If still over, cut at a user-turn boundary. When a single long turn has no
   later user boundary, cut at the first non-tool message instead; an assistant
   call stays ahead of its own results. The cut never separates an assistant
   tool call from its tool results and never starts the recent window with a
   dangling tool message.
4. Replace older turns with one summary message:
   - deterministic digest of goals, files, commands, notes, and outcomes;
   - optionally rewritten by the model when `compact_with_llm` is true and the
     loop is the main agent.
5. If the result is still too large, drop the oldest whole turns, or one safe
   assistant/tool step inside a long turn, without leaving an orphan result.

Compaction stats and the current context usage are emitted as events, so the UI
can show the meter and a compaction marker.

## 7. Sessions

Sessions are JSON files at `.filament/sessions/<id>.json`:

```json
{
  "session_id": "20261111-223636-c86b6c45",
  "cwd": "D:/project",
  "mode": "build",
  "parent_id": null,
  "kind": "main",
  "title": "fix the failing tests",
  "status": "done",
  "messages": [],
  "todos": [],
  "pending_question": null,
  "child_sessions": [],
  "files_changed": [],
  "usage": {"steps": 5, "tokens": 541, "compactions": 0}
}
```

Writes use a temporary file and atomic replace. Old session files missing new
fields load with defaults; unknown fields are ignored. Loaded container fields
are normalized: non-object messages, unknown roles, tool results without an id,
and pending questions that cannot be answered are dropped, so one corrupt file
cannot break the session list or the next turn.

## 8. Providers and Streaming

Neutral message format:

```python
{"role": "assistant", "content": "...", "tool_calls": [{"id", "name", "arguments"}]}
{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
```

Adapters:

| Provider | Tool schema | Stream | Notes |
|---|---|---|---|
| `mock` | neutral | whole text | Deterministic scripts for tests |
| `ollama` | OpenAI function tools | `/api/chat` NDJSON | Local or hosted, optional key |
| `openai` | OpenAI function tools | SSE | Works with Groq, Together, vLLM, LM Studio |
| `anthropic` | `input_schema` | SSE | Tool results grouped into user messages |

If a local model returns no native tool calls but emits a single JSON object
such as `{"tool": "read_file", "arguments": {...}}`, the loop accepts it as one
action and continues. This is a fallback, not the primary path.

Responses are read through a bounded reader (8 MiB), and error bodies are
truncated to 64 KiB before being included in an exception.

## 9. Backend and UI

The Electron app spawns `python -m filament.backend_server` with the repository
root on `PYTHONPATH`. The protocol is newline-delimited JSON:

- request: `{"id", "method", "params"}`
- response: `{"id", "ok": true, "result"}` or `{"id", "ok": false, "error"}`
- notification: `{"event": "...", "session_id": "..."}`

`chat`, `answer`, and `skip_question` run on worker threads; the reader remains
free for `cancel`. A write lock serializes stdout across threads.

Methods: `ping`, `get_config`, `set_config`, `select_workspace`, `get_state`,
`chat`, `answer`, `skip_question`, `cancel`, `new_session`, `set_mode`,
`list_sessions`, `load_session`, `delete_session`, `get_messages`,
`compress_context`, `read_file`, `list_dir`.

The renderer is event-driven and holds no Node access. It renders assistant
deltas, tool cards, nested subagent streams, todos, context, questions, and a
final done summary. History is rebuilt from stored neutral messages.

## 10. Configuration

`HarnessConfig` validates every field on load. Important additions in 0.5:

| Field | Default | Meaning |
|---|---|---|
| `max_steps` | 40 | Main loop limit |
| `subagent_max_steps` | 24 | Child loop limit |
| `subagent_max_tokens` | 32000 | Child context budget |
| `tool_output_limit` | 12000 | Inline tool result chars |
| `keep_recent_messages` | 8 | Recent window for compaction |
| `compact_threshold` | 0.75 | Compaction trigger |
| `compact_with_llm` | true | Model-written summaries |
| `max_depth` | 1 | Subagent nesting |
| `llm.stream` | true | Streaming requests |
| `llm.max_output_tokens` | 4096 | Response output cap |

Environment overrides remain available through `FILAMENT_*` variables. A
project-local `.env` is read only for the active configuration location, and a
quoted `"false"` is rejected instead of acting like `true`.

## 11. Software Requirements Specification

### 11.1 Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | The system shall run an agent loop with native tool calling and streamed output. |
| FR-2 | The system shall expose read, write, exact edit, multi-edit, list, glob, grep, shell, syntax-check, todo, question, subagent, and finish tools. |
| FR-3 | The system shall confine every path to the workspace unless explicitly allowed. |
| FR-4 | The system shall keep context under the configured budget using pruning and turn-boundary compaction. |
| FR-5 | The system shall preserve the pairing of assistant tool calls and tool results across compaction. |
| FR-6 | The system shall give every subagent its own session, history, budget, and step limit. |
| FR-7 | The system shall return a subagent's final summary to the parent as a tool result. |
| FR-8 | The system shall pause on `ask_user` and resume with the user's answer. |
| FR-9 | The system shall stop on `finish`, cancellation, step limits, or repeated tool-call stalls. |
| FR-10 | The system shall persist session state atomically, including parent links and pending questions. |
| FR-11 | The system shall stream JSON notifications while a turn runs. |
| FR-12 | The system shall support build and read-only plan modes. |
| FR-13 | The system shall expose CLI, TUI, Python, and Electron surfaces. |
| FR-14 | The system shall retain the legacy batch runner. |

### 11.2 Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | Default main context shall remain at or below 120k estimated tokens, with subagents at 32k. |
| NFR-2 | Provider responses shall be read with a hard byte cap. |
| NFR-3 | Session writes shall be atomic and bounded. |
| NFR-4 | The Electron renderer shall run with context isolation and no Node integration. |
| NFR-5 | The terminal surfaces shall use only the Python standard library. |
| NFR-6 | A failed summary or failed subagent shall not abort the parent run. |
| NFR-7 | The mock provider shall support tests without network access. |

## 12. Error Handling

- Tool errors become `ERROR:` tool results the model can recover from.
- Provider failures retry once when no text has streamed, then end the turn with
  `status = "error"`. After a partial stream the turn ends immediately, so the
  UI is not shown the same text twice.
- A failed subagent becomes a tool error in the parent; other parallel children
  still return.
- A failed compaction summary falls back to a deterministic digest.
- A broken event sink is ignored so a UI problem cannot stop a run.
- Oversized or malformed backend request lines produce one error response and do
  not terminate the server.

## 13. Verification

- 193 unit tests cover the loop, tools, editing, subagents, question
  pause/resume, compaction, streaming parsers, backend protocol, configuration,
  and the legacy batch runner.
- The full HTTP streaming path was exercised against a protocol-faithful fake
  Ollama server: five steps, file creation, todos, a subagent with its own
  session, and a finish summary.
- The Electron app was exercised with that server: streaming text, tool cards,
  nested subagent output, question cards, session load, settings save, and both
  themes.
- CI runs the suite on Windows for Python 3.11 and 3.12, compiles the package,
  smoke-tests the legacy CLI, installs Electron, and syntax-checks the renderer
  and main process.

## 14. Legacy Compatibility

`filament run <tasks.json>` still uses `Orchestrator`, which spawns one focused
completion per task, appends extracted code in one batch, and optionally runs a
line-level review/fix pass. Its modules (`agent.py`, `tools.py`, `files.py`,
`review.py`, `tasks.py`, `context.py`) remain for existing scripts and tests.
New work should use the agent loop.

## 15. Known Limitations

- Token counts are estimates.
- Compaction is lossy and cannot preserve every detail.
- Tool-calling quality depends on the model; small local models may need the
  JSON fallback or a larger model.
- The harness is not a sandbox. Shell access is real; use an allowlist and a
  disposable workspace for untrusted prompts.
- Cancellation takes effect between model calls, not mid-request.

## 16. Future Work

- Exact tokenizer integration per provider.
- Optional approval gates for shell and destructive edits.
- Diff review before writes in a dedicated review mode.
- Richer file tree and diff viewer in the Electron app.
- Resumable background subagents across app restarts.
