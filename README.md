# Filament Harness

Lightweight coding harness for local LLMs. Keeps context small by spawning focused agents
per task, appending their output in batches, and reviewing only targeted file regions.

Python 3.11+.

## Quickstart

```bash
pip install -e .
filament run examples/demo_tasks.json --client mock
```

Example task file:

```json
{
  "tasks": [
    "Write a Python function named add(a, b) that returns a + b.",
    {
      "id": "math",
      "task": "Implement math helpers",
      "subtasks": [
        "Write a Python function named mul(a, b) that returns a * b.",
        "Write a Python function named sub(a, b) that returns a - b."
      ],
      "target": "examples/math.py"
    }
  ],
  "target": "examples/generated.py"
}
```

## Concepts

- Orchestrator: expands tasks/subtasks, enforces token budget, runs agents in parallel.
- Agent: receives one exact task; returns only requested code.
- Tools: `run_shell`, `spawn_agent(task)`.
- Context budget: enforced near 120k tokens (approx., 4 chars/token) with safety margin.
- Writes: append batch after all agents finish; fixes apply by exact line number.
- Security: target paths must stay inside the working root unless explicitly allowed.
- Tool arguments: `run_shell` accepts only positive timeouts (default `timeout_shell`, capped at 86400s) and `read_file` only positive `max_chars`; invalid values raise `ValueError` before reaching `subprocess` or slicing. `max_fixes` must be null or a non-negative integer.
- Provider transport: HTTP response bodies are capped at 8 MiB (`MAX_RESPONSE_BYTES`) and provider error text at 64 KiB; larger bodies raise `RuntimeError` before JSON decoding. `[llm].timeout` must be a whole number of seconds between 1 and 86400.

## Providers

Filament supports:

- `mock` for offline tests
- `ollama` for local Ollama
- `openai` for OpenAI and OpenAI-compatible endpoints (Groq, Together, vLLM, LM Studio, etc.)
- `anthropic` for Anthropic Messages API

Use `--client openai --model ...` with `FILAMENT_API_KEY`, or configure `[llm]` in TOML. Ollama cloud endpoints may use either `FILAMENT_API_KEY` or `OLLAMA_API_KEY`.

## TUI

Start the interactive terminal UI:

```bash
filament tui
# or
filament-tui
```

The TUI supports provider/settings/context/execution/path/run configuration, task
management, config save/load, and running the harness with live result output.

## Electron backend

`python -m filament.backend_server` is the newline-delimited JSON backend used by
the Electron app. Each request is one UTF-8 JSON object per line; each response is
one line, either `{"id": ..., "ok": true, "result": ...}` or
`{"id": ..., "ok": false, "error": "..."}`. Request lines are capped at 1 MiB
(`MAX_REQUEST_LINE_BYTES`); a malformed, non-object, or oversized request gets an
error response and the session keeps running.

`ping` returns the package version. `set_config` validates a copy of the live
config before publishing it: unknown keys and wrong JSON types are rejected, a
failed update leaves the previous config in place, and a successful update rebuilds
the LLM client and toolset so provider changes take effect.

## Configuration

Copy `filament.toml.example` to `filament.toml` or pass `--config`.

Important settings:

- `max_tokens`: total token budget (default 120000)
- `budget_margin`: reserved margin (default 0.05)
- `workers`: parallel agent count
- `timeout_shell`: shell timeout
- `allow_outside_target`: allow writing outside working root
- `atomic_writes`: write complete files via temp file and atomic replace when enabled
- `use_shell`: use shell mode for `run_shell`
- `offload_shell_output`: save oversized `run_shell` output under `.filament/runs` and return its path
- `command_allowlist`: optional command prefixes
- `[llm]`: `kind`, `model`, `base_url`, `timeout` (seconds, default 120)

Config values are type-checked when the file loads: numbers must be numbers, booleans
must be `true` or `false`, and `[llm]` must be a table. A quoted `"false"` is rejected
instead of acting like `true`, so a typo cannot silently enable `allow_outside_target`.
Every bad top-level key is listed in one `ValueError`, for example
`config error: invalid configuration: max_tokens must be an integer, got str '120000'`.
Whole floats such as `120000.0` are accepted for integer settings; fractional values
are rejected outside float settings such as `budget_margin`.

Environment overrides: `FILAMENT_CLIENT`, `FILAMENT_MODEL`, `FILAMENT_OLLAMA_URL`,
`FILAMENT_MAX_TOKENS`, `FILAMENT_WORKERS`, `FILAMENT_TIMEOUT_SHELL`, `FILAMENT_TIMEOUT`, `FILAMENT_USE_OLLAMA`.

## Testing

```bash
python -m unittest discover -s tests
```
