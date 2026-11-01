# Filament Harness

Lightweight coding harness for local LLMs. Keeps context small by spawning focused agents per task and appending their output in batches.

Python 3.11+.

## Quickstart

```bash
pip install -e .
filament run examples/tasks.json
```

## Concepts

- Orchestrator: plans minimal tasks; does not hold large code.
- Agent: receives one exact task; returns only the requested code.
- Tools: `run_shell`, `spawn_agent(task)`.
- Context budget: enforced at 120k tokens (approx).
