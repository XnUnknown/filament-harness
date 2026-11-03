# Filament Electron App Plan

## Goal

Build a desktop GUI application for Filament Harness that feels like a lightweight
Claude Code / OpenCode-style coding agent, while reusing the existing Python
harness for planning, agent execution, review, and context management.

The GUI must not introduce repository-wide context, must preserve the 120k-token
budget, and must keep all file operations inside the selected working directory.

## Architecture

Frontend:
- Electron main process spawns and supervises the Python backend.
- Electron renderer is a plain HTML/CSS/JavaScript app to avoid heavy
  frontend build dependencies.
- IPC between renderer and Python uses newline-delimited JSON over stdio.

Backend:
- Add `filament/backend_server.py`, started as `python -m filament.backend_server`.
- The server receives one JSON request per line and returns one JSON response
  per line.
- The server reuses `HarnessConfig`, `ChatOrchestrator`, `Orchestrator`,
  `Session`, `Toolbox`, and context compression.

Persistent state:
- Sessions remain in `<workspace>/.filament/sessions`.
- Config remains `filament.toml` plus `.env`.
- No secrets are stored in the Electron frontend.

## Iteration Plan

### Feature 1: Electron scaffold
Deliverable:
- `electron/package.json`
- `electron/main.js`
- `electron/preload.js`
- `electron/renderer/index.html`
- basic window lifecycle, safe IPC bridge.

Acceptance:
- `npm install` works.
- `npm start` opens a desktop window.
- Main process can send a ping to renderer and back.

### Feature 2: Python backend bridge
Deliverable:
- `filament/backend_server.py` implementing JSON line protocol.
- Commands: `ping`, `load_config`, `save_config`, `list_sessions`, `load_session`.
- Electron main client with restart-on-crash and timeout handling.

Acceptance:
- Renderer can call `filament.ping()` and receive version/session id.
- Backend failures surface as structured errors instead of silent hangs.

### Feature 3: Workspace lock
Deliverable:
- Folder picker in Electron.
- Backend validates all requested paths against the selected workspace.
- Tool execution uses the selected workspace as root.

Acceptance:
- Files cannot escape the selected workspace.
- Switching workspaces clears in-memory session/task state unless loaded explicitly.

### Feature 4: Conversational agent UI
Deliverable:
- Chat transcript panel.
- Input box with Enter to send.
- Streaming-friendly status line for planning/execution/reflection.
- Small-talk responses do not spawn coding agents.

Acceptance:
- A message such as "hello" receives a normal response.
- A coding request produces plan, execution result, and generated-file display.

### Feature 5: Mode switches
Deliverable:
- Mode toggle: `auto`, `plan`, `review`, `accept`.
- Plan mode asks confirmation before execution.
- Review mode shows review findings and optional fixes.
- Auto mode runs plan and bounded follow-up loop without extra prompts.

Acceptance:
- Changing mode updates backend session state.
- Plan-mode execution waits for explicit user confirmation.

### Feature 6: Settings panel
Deliverable:
- Provider, model, base URL, API key, timeout.
- Context, workers, shell timeout, reasoning effort.
- Validation errors displayed inline.

Acceptance:
- Settings round-trip through backend.
- Invalid provider/key errors are shown without crashing the app.

### Feature 7: Session manager
Deliverable:
- New/load/list sessions.
- Session metadata display.
- Delete/rename optional.

Acceptance:
- Sessions persist after app restart.
- Loaded session restores transcript and changed-file list.

### Feature 8: Agent activity and task progress
Deliverable:
- Show current phase: planning, spawning, appending, reviewing, reflecting, done.
- Show task list with status and target files.
- Show token/context usage.

Acceptance:
- Each tool/agent operation emits progress events to renderer.
- Failed tasks are visible without hiding successful ones.

### Feature 9: Code viewer and diff
Deliverable:
- Read-only file viewer for generated/changed files.
- Simple before/after diff for line fixes.
- Copy/open-file actions.

Acceptance:
- Review fixes visibly change the displayed file.
- Large files are bounded before display.

### Feature 10: Autonomous review loop
Deliverable:
- After execution, orchestrator reads changed files and runs targeted checks.
- Follow-up plans are applied up to a configured limit.
- Loop stops on `done` or when no further plan is returned.

Acceptance:
- A clearly incomplete generated result triggers a follow-up without manual prompting.
- The loop respects the configured maximum number of passes.

### Feature 11: Context compression
Deliverable:
- Compression runs when nearing budget.
- Transcript indicators when compression happened.

Acceptance:
- Long sessions do not exceed the configured token budget.
- Important objectives and changed files survive compression.

### Feature 12: Security and containment
Deliverable:
- Backend validates all paths against workspace root.
- Shell commands run only in workspace root.
- Output truncation and timeouts enforced.

Acceptance:
- A request to read or write outside the workspace is rejected.
- Long shell output cannot overflow the context budget.

### Feature 13: Product polish and release
Deliverable:
- App icon, menu, keyboard shortcuts.
- Installable packaged builds for Windows.
- Release notes and smoke-test checklist.

Acceptance:
- Packaged app launches without a separate Python setup beyond bundled/runtime instructions.
- CI builds a distributable artifact.

## Backend JSON Protocol

Each request is one JSON object per line:

```json
{"id": "1", "method": "ping", "params": {}}
```

Response:

```json
{"id": "1", "ok": true, "result": {"version": "0.4.0"}}
```

Initial methods:
- `ping`
- `get_config`
- `set_config`
- `select_workspace`
- `list_sessions`
- `load_session`
- `new_session`
- `chat`
- `run_task`
- `read_file`
- `compress_context`

## Development Order

1. Electron scaffold.
2. Python backend bridge.
3. Workspace lock.
4. Conversational agent UI.
5. Mode switches.
6. Settings, sessions, progress.
7. Code viewer and review loop.
8. Context compression and security hardening.
9. Packaging and release.

## Non-Goals For Initial Release

- Multi-user collaboration.
- Remote/cloud-hosted backend.
- IDE-plugin protocol.
- Full DOM/browser automation.
- Uncontrolled full-repository indexing.
