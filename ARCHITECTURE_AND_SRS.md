# Filament Harness — Architecture and SRS

## 1. Purpose and Scope

Filament Harness is a lightweight coding harness for local LLMs running on laptops.
It keeps active context small by giving each coding agent exactly one task, collecting
results in parallel, and writing code in one batch after all agents complete.

This document describes the architecture that was implemented and the Software
Requirements Specification (SRS) for the current version.

## 2. System Overview

- Project name: `filament-harness`
- Repository path: `D:/Hermes Agent Workspace/filament-harness`
- Language/runtime: Python 3.11+
- LLM access: Ollama (optional) with deterministic mock client for offline tests
- Primary interfaces: Python API, CLI (`filament run`)
- Core constraint: active context stays at or below 120,000 tokens (approx.)

### 2.1 Core Concept

Instead of one long-lived agent that accumulates the whole codebase in context:

1. The orchestrator decomposes work into independent, exact task instructions.
2. Each task is given to one spawned agent with minimal or no repository context.
3. All agents run in parallel.
4. Their results are collected before any file write occurs.
5. All code is appended to the target file in one batch.
6. The resulting file is reviewed using a bounded prompt.
7. Line-specific fix tasks are spawned only for review findings.

## 3. Architecture

### 3.1 High-Level Components

| Component | Responsibility |
|---|---|
| `Orchestrator` | Plans task execution, spawns agents, batches writes, reviews, fixes |
| `CodingAgent` | Wraps one exact task instruction |
| `Toolset` | Provides `run_shell` and `spawn_agent(task)` |
| `LLMClient` | Ollama or mock implementation used by agents |
| `ContextBudget` | Approximates token usage and trims oldest context |
| `File helpers` | Code extraction, append writes, exact line replacement |
| `Review helper` | Bounded review and `FIX line: instruction` parsing |
| `CLI` | JSON task file execution |

### 3.2 Module Layout

```text
filament-harness/
  filament/
    __init__.py
    agent.py
    cli.py
    context.py
    files.py
    llm.py
    orchestrator.py
    review.py
    tools.py
  examples/
    demo_tasks.json
  tests/
    test_basic.py
  README.md
  pyproject.toml
  .gitignore
```

### 3.3 Data Flow

```text
Task list + target file
        |
        v
Orchestrator.run(tasks, target_file)
        |
        +--> CodingAgent per task
        |       |
        |       +--> spawn_agent(task) -> LLM -> code text
        |
        +--> Wait for all results
        |
        +--> Extract fenced code blocks
        |
        +--> Append all code in one batch -> target file
        |
        +--> run_shell tests (optional external)
        |
        +--> Review(target file) -> FIX lines
        |
        +--> Fix agents -> replace exact lines
        |
        +--> Rerun tests
```

### 3.4 Concurrency Model

- Agents are executed with a thread pool.
- Default worker count: 4.
- Agents do not write files directly.
- File writes happen only after all spawned agents complete.
- This prevents race conditions, interleaved code, and partial-write ambiguity.

### 3.5 Context Budget

- Approximation: 1 token = 4 characters.
- Default maximum: 120,000 tokens.
- Context is stored as chunks in `ContextBudget`.
- Each shell result, spawn prompt, and LLM result is added to the budget.
- If the budget is exceeded before a spawn, execution raises an error.
- When accumulation exceeds the cap, the oldest chunks are removed.
- Review input is capped to the remaining budget using the tail of the file.

## 4. Tools

### 4.1 `run_shell(command, timeout=60)`

- Executes a shell command in the configured working directory.
- Captures stdout and stderr.
- Records the command and output in the context budget.
- Used for tests, linting, build scripts, and other local commands.

### 4.2 `spawn_agent(task)`

Input:
- `task`: one exact coding instruction.

Behavior:
- No broad repository context is added.
- Uses a system prompt requiring one fenced code block and no prose.
- Returns raw generated text.
- Records prompt and result in the context budget.
- Refuses to run if the context budget is exceeded.

## 5. File Integration Rules

### 5.1 Code Extraction

- Extract the first fenced code block from LLM output.
- If no fence exists, fall back to raw text.
- Trim surrounding whitespace.

### 5.2 Batch Append

- Code is never appended until all agents complete.
- Each extracted code block is appended to the target file in completion order.
- Parent directories are created automatically.

### 5.3 Line Fixes

- Review findings use the format: `FIX <line_number>: <instruction>`.
- Fix agents return only the replacement line.
- Markdown fences are stripped before replacement.
- Replacement happens by exact 1-indexed line number.

## 6. Review Workflow

- Review runs after the batch append.
- Review input is capped to the remaining context budget.
- The review prompt is separate and bounded.
- Findings are parsed with a regex.
- Each finding triggers a fix agent.
- Fixes are applied and can be followed by targeted tests.

## 7. Environment and Configuration

| Setting | Default | Purpose |
|---|---|---|
| `FILAMENT_MODEL` | `llama3.2` | Ollama model name |
| `FILAMENT_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `FILAMENT_USE_OLLAMA` | unset | Set to `1` to use Ollama instead of mock |
| CLI `--client` | `auto` | Select `auto`, `mock`, or `ollama` |
| CLI `--workers` | `4` | Agent thread pool size |
| CLI `--shell-cwd` | `.` | Working directory for shell tools |

## 8. Software Requirements Specification

### 8.1 Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | The system shall accept a list of independent coding task instructions. |
| FR-2 | The system shall accept a target file path for generated code. |
| FR-3 | The system shall spawn one agent per task. |
| FR-4 | Each agent shall receive one exact task and no broad repository context. |
| FR-5 | The system shall support parallel agent execution. |
| FR-6 | The system shall wait for all agents to complete before file writes. |
| FR-7 | The system shall extract code from LLM output. |
| FR-8 | The system shall append all generated code in one batch. |
| FR-9 | The system shall track approximate token context usage. |
| FR-10 | The system shall refuse to exceed the configured token budget for a spawn. |
| FR-11 | The system shall provide a `run_shell` tool. |
| FR-12 | The system shall provide a `spawn_agent(task)` tool. |
| FR-13 | The system shall review the target file after the batch write. |
| FR-14 | The system shall identify line-specific fixes from review output. |
| FR-15 | The system shall spawn fix agents for review findings. |
| FR-16 | The system shall replace exact lines during fixes. |
| FR-17 | The system shall expose a CLI for JSON task files. |
| FR-18 | The system shall support mock and Ollama LLM clients. |

### 8.2 Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Active context shall stay at or below 120,000 tokens by default. |
| NFR-2 | Context approximation shall use 4 characters per token. |
| NFR-3 | The architecture shall avoid whole-repository prompts to coding agents. |
| NFR-4 | File writes shall be batched and deterministic. |
| NFR-5 | Parallel agents shall not write files directly. |
| NFR-6 | Review shall use bounded input based on remaining context. |
| NFR-7 | The system shall run without external network access when using the mock client. |
| NFR-8 | The codebase shall use Python 3.11+ standard library where practical. |
| NFR-9 | Operations shall be suitable for laptop-scale local LLMs. |

### 8.3 Constraints

- The token count is approximate, not exact.
- Agents depend on model compliance with the one-fenced-block instruction.
- Ollama must be installed and running for real local model execution.
- The mock client is for tests and does not represent real model behavior.
- Generated code quality depends on model quality.
- The harness targets lightweight coding workflows, not full-repository refactors.

## 9. Interfaces

### 9.1 Python API

```python
from filament.orchestrator import Orchestrator

orch = Orchestrator(client_kind="mock", max_tokens=120_000)
result = orch.run(
    [
        "Write a Python function named add(a, b) that returns a + b.",
        "Write a Python function named mul(a, b) that returns a * b.",
    ],
    "output.py",
)
```

### 9.2 CLI

```bash
filament run tasks.json
```

`tasks.json` format:

```json
{
  "tasks": [
    "Write a Python function named add(a, b) that returns a + b."
  ],
  "target": "output.py"
}
```

### 9.3 Task JSON Schema (informal)

- `tasks`: array of strings (required)
- `target`: string (optional, default `output.py`)
- `workers`: may be supplied through CLI, not JSON

## 10. Error Handling

- A spawn that would exceed the context budget raises `RuntimeError`.
- Missing target directories are created during append.
- Out-of-range line fixes are skipped instead of crashing the whole run.
- Ollama connection failures propagate as standard Python exceptions.
- Mock client is used by default for offline and test workflows.

## 11. Verification and Acceptance

The implementation was verified with:

- Unit test: `tests/test_basic.py`
  - Result: passes
  - Checks 2 appended functions
  - Checks context tokens ≤ 120,000
- CLI demo: `examples/demo_tasks.json`
  - Result: generated `add` and `mul` functions in one batch
  - Measured context: 95 tokens

Acceptance criteria:
- Target file exists after a run.
- Generated code is appended in batch order.
- Context usage remains within configured limit.
- Fixes are applied by exact line number.
- No parallel agent writes directly to files.

## 12. Known Limitations

- Token counts are character-based approximations.
- Real models may include prose despite system constraints.
- Automatic review depends on model adherence to `FIX <line>: <instruction>`.
- Line-based fixes can be fragile on large files.
- The harness is not intended for large refactors or security audits.

## 13. Future Improvements

- Optional exact tokenizer integration.
- Per-task target files and line ranges.
- Structured test runner integration.
- Stronger code extraction and validation.
- Optional Git diff review step.
- Persistent run artifacts and reproducible logs.
