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

## Providers

Filament supports:

- `mock` for offline tests
- `ollama` for local Ollama
- `openai` for OpenAI and OpenAI-compatible endpoints (Groq, Together, vLLM, LM Studio, etc.)
- `anthropic` for Anthropic Messages API

Use `--client openai --model ...` with `FILAMENT_API_KEY`, or configure `[llm]` in TOML.

## TUI

Start the interactive terminal UI:

```bash
filament tui
# or
filament-tui
```

The TUI supports provider/settings/context/execution/path/run configuration, task
management, config save/load, and running the harness with live result output.

## Configuration

Copy `filament.toml.example` to `filament.toml` or pass `--config`.

Important settings:

- `max_tokens`: total token budget (default 120000)
- `budget_margin`: reserved margin (default 0.05)
- `workers`: parallel agent count
- `timeout_shell`: shell timeout
- `allow_outside_target`: allow writing outside working root
- `use_shell`: use shell mode for `run_shell`
- `command_allowlist`: optional command prefixes
- `[llm]`: `kind`, `model`, `base_url`

Environment overrides: `FILAMENT_CLIENT`, `FILAMENT_MODEL`, `FILAMENT_OLLAMA_URL`,
`FILAMENT_MAX_TOKENS`, `FILAMENT_WORKERS`, `FILAMENT_TIMEOUT_SHELL`, `FILAMENT_USE_OLLAMA`.

## Testing

```bash
python -m unittest discover -s tests
```
