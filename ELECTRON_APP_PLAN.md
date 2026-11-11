# Electron App

The Filament desktop app is a thin, event-driven renderer over the JSON backend.
It has no Node access, no bundler, and no external UI dependencies.

## Run

```bash
cd electron
npm install
npm start
```

Requirements:

- Python 3.11+ with the `filament` package installed (`pip install -e ..`), or
  `PYTHONPATH` pointing at the repository root;
- `FILAMENT_PYTHON` if the interpreter is not on `PATH` as `python`.

The app starts `python -m filament.backend_server` from the repository root and
loads `renderer/index.html`.

## Structure

```text
electron/
  main.js              window, backend process, IPC, event fan-out
  preload.js           contextBridge: command, selectWorkspace, openPath, onEvent
  renderer/
    index.html         layout: sidebar, transcript, inspector, settings modal
    app.js             state, event handling, markdown, history, questions
    styles.css         dark/light themes, cards, meter, responsive layout
```

## Request flow

1. Renderer calls `window.filament.command(method, params)`.
2. `main.js` assigns a request id and writes one line to the backend stdin.
3. The backend answers with `{"id", "ok", "result"}`.
4. While a turn runs, the backend writes notifications with an `event` key.
5. `main.js` forwards notifications to the renderer through
   `filament:event`, and the renderer updates the transcript incrementally.

Long methods (`chat`, `answer`, `skip_question`) have a 30-minute client timeout;
other methods time out after 60 seconds. The Stop button sends `cancel`, which
is handled by the backend reader thread while the agent worker continues.

## Event types rendered

- `session` — session id, title, mode
- `status` — thinking / cancelling
- `delta` — streamed assistant text
- `assistant_start`, `assistant_end` — turn boundaries and tool-call metadata
- `tool_start`, `tool_end` — collapsible tool cards with duration and result
- `subagent_start`, `subagent_end` — nested subagent cards
- `todos` — plan panel
- `context` — token meter
- `compaction` — context marker
- `question` — question card with options and free-text answer
- `heartbeat` — elapsed-time status while the model is thinking
- `error`, `done`, `backend_exit`

Child events carry `subagent_id`; the renderer routes them into the matching
subagent card instead of the main transcript.

## UI features

- Workspace picker and session list with status dots.
- Build / Plan segmented control; plan mode is read-only.
- Streaming assistant text with a compact markdown renderer (code fences,
  inline code, bold/italic, lists, headings, safe links).
- Collapsible tool cards with arguments and results; failures auto-expand.
- Subagent cards with their own tool cards and final report.
- Question cards with suggested options, free-text input, and a disabled send
  button until the question is answered.
- Context meter with warning and hot states.
- Plan, subagents, and changed files in the inspector.
- Settings modal for provider, model, base URL, key, timeouts, context budget,
  step limits, and streaming.
- Dark and light themes persisted in `localStorage`.
- Keyboard: Enter sends, Shift+Enter inserts a newline.

## History

`get_messages` returns the neutral message list. The renderer rebuilds the
transcript from it:

- user messages become bubbles;
- assistant text becomes markdown blocks;
- tool calls become cards filled from matching `tool_call_id` results;
- `finish` becomes a done summary instead of a tool card;
- `ask_user` results become "You answered: ..." lines;
- compaction summaries become a context marker.

Subagent cards are live-only; a reloaded session shows the parent's spawn tool
card with the child summary in its result.

## Security

- `contextIsolation: true`, `nodeIntegration: false`.
- The preload exposes only four functions.
- No remote content is loaded; CSP allows only self-hosted scripts and styles.
- The backend is a child process with the repository root on `PYTHONPATH`.

## Debugging

- `main.js` logs renderer `console-message`, `did-fail-load`, preload errors, and
  backend stderr to the terminal.
- Launch with `--remote-debugging-port=9222` and use the DevTools protocol if
  the window is occluded and OS screenshots look stale.
- Backend protocol lines can be tested directly:

```bash
printf '%s\n' '{"id":1,"method":"ping"}' | python -m filament.backend_server
```

## Packaging notes

The app currently runs from the repository. To package it, ship the `filament`
package (or a virtual environment) with the Electron resources, set
`FILAMENT_PYTHON` to that interpreter, and include `main.js`, `preload.js`, and
`renderer/`. The backend path in `main.js` assumes `electron/` sits next to the
Python package.
