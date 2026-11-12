"""Tool surface for the Filament agent loop.

The old harness exposed two blunt tools (`run_shell`, `spawn_agent`) and made
agents append code to a file.  That is why agents behaved lazily: they had no
way to look around, edit precisely, track work, or ask a question.  This module
gives them the same primitives a mature coding agent has - read, write, exact
edit, multi-edit, glob, grep, shell, todos, questions, subagents - while every
path stays confined to the workspace.
"""
from __future__ import annotations

import difflib
import fnmatch
import os
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .config import HarnessConfig, coerce_timeout
from .shell import prepare_command

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".filament",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

MAX_READ_LINES = 2000
MAX_READ_CHARS = 60_000
MAX_GREP_CHARS = 40_000
MAX_DIFF_CHARS = 4_000
MAX_SPAWN_DEPTH = 4


class ToolError(RuntimeError):
    """A recoverable tool failure; the message goes back to the model."""


class AskUser(Exception):
    """Signal that the loop must pause and wait for a human answer."""

    def __init__(self, question: str, options: list[str] | None = None):
        super().__init__(question)
        self.question = question
        self.options = options or []


class FinishTask(Exception):
    """Signal that the agent declared the task complete."""

    def __init__(self, summary: str):
        super().__init__(summary)
        self.summary = summary


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], str], str]
    read_only: bool = False
    parallel_safe: bool = False

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }


def _str_prop(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _int_prop(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


def _arg_str(args: dict[str, Any], key: str, required: bool = True, default: str = "") -> str:
    value = args.get(key, default)
    if value is None:
        value = default
    if not isinstance(value, str):
        value = str(value)
    if required and not value.strip():
        raise ToolError(f"{key} is required")
    return value


def _arg_int(args: dict[str, Any], key: str, default: int, minimum: int = 1, maximum: int = 1_000_000) -> int:
    value = args.get(key, default)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ToolError(f"{key} must be an integer")
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            raise ToolError(f"{key} must be an integer, got {value!r}") from None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ToolError(f"{key} must be an integer")
    if value < minimum:
        raise ToolError(f"{key} must be at least {minimum}")
    return min(value, maximum)


def _arg_bool(args: dict[str, Any], key: str, default: bool = False) -> bool:
    """Accept a real JSON boolean only.

    ``bool("false")`` is true, so treating strings as flags silently turned
    ``replace_all: "false"`` into a full-file replacement.
    """
    value = args.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ToolError(f"{key} must be true or false, got {type(value).__name__}")


def _truncate_middle(text: str, limit: int) -> str:
    if limit < 1:
        raise ValueError("limit must be positive")
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    marker = f"\n... [truncated {omitted} chars] ...\n"
    if limit <= len(marker):
        return text[-limit:]
    head = (limit - len(marker)) // 2
    tail = limit - len(marker) - head
    return text[:head] + marker + text[-tail:]


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))


def _count_occurrences(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    return haystack.count(needle)


def _match_lines(text: str, needle: str, limit: int = 6) -> list[int]:
    lines: list[int] = []
    for index, line in enumerate(text.splitlines(), 1):
        if needle.splitlines() and needle.splitlines()[0].strip() and needle.splitlines()[0].strip() in line:
            lines.append(index)
            if len(lines) >= limit:
                break
    return lines


def _unified_diff(path: str, before: str, after: str) -> str:
    diff = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
            n=2,
        )
    )
    if not diff:
        return ""
    text = "\n".join(diff)
    return _truncate_middle(text, MAX_DIFF_CHARS)


class ToolRegistry:
    """Workspace-scoped tools shared by the main agent and its subagents."""

    def __init__(
        self,
        workspace: str | Path,
        config: HarnessConfig,
        *,
        depth: int = 0,
        read_only: bool = False,
        allow_questions: bool = True,
        allow_subagents: bool = True,
        spawn_fn: Callable[..., dict[str, Any]] | None = None,
        on_file_change: Callable[[str], None] | None = None,
        on_todos: Callable[[list[dict[str, Any]]], None] | None = None,
        todos: list[dict[str, Any]] | None = None,
    ):
        self.workspace = Path(workspace).expanduser().resolve()
        self.config = config
        self.depth = max(0, depth)
        self.read_only = bool(read_only)
        self.allow_questions = bool(allow_questions)
        self.allow_subagents = bool(allow_subagents) and depth < min(config.max_depth, MAX_SPAWN_DEPTH)
        self.spawn_fn = spawn_fn
        self.on_file_change = on_file_change
        self.on_todos = on_todos
        self.todos: list[dict[str, Any]] = list(todos or [])
        self._tools = self._build_tools()

    # -- registry -----------------------------------------------------------
    def _build_tools(self) -> dict[str, Tool]:
        tools = [
            Tool(
                "read_file",
                "Read a text file with line numbers. Use offset/limit for large files; "
                "the footer tells you the next offset. Always read a file before editing it.",
                _obj(
                    {
                        "path": _str_prop("File path, relative to the workspace or absolute inside it."),
                        "offset": _int_prop("First line number to read (1-based). Default 1."),
                        "limit": _int_prop("Maximum lines to return. Default 400, max 2000."),
                    },
                    ["path"],
                ),
                self._read_file,
                read_only=True,
                parallel_safe=True,
            ),
            Tool(
                "list_dir",
                "List a directory. Directories end with '/'; files show their size.",
                _obj({"path": _str_prop("Directory path. Defaults to the workspace root.")}),
                self._list_dir,
                read_only=True,
                parallel_safe=True,
            ),
            Tool(
                "glob",
                "Find files by glob pattern (for example '**/*.py' or 'src/*.tsx'), newest first. "
                "Use this instead of guessing paths.",
                _obj(
                    {
                        "pattern": _str_prop("Glob pattern relative to the workspace."),
                        "path": _str_prop("Directory to search from. Default workspace root."),
                        "limit": _int_prop("Maximum results. Default 200."),
                    },
                    ["pattern"],
                ),
                self._glob,
                read_only=True,
                parallel_safe=True,
            ),
            Tool(
                "grep",
                "Search file contents with a regular expression. Returns path:line: text. "
                "Use it to find symbols, call sites, and error strings before editing.",
                _obj(
                    {
                        "pattern": _str_prop("Python regular expression."),
                        "path": _str_prop("File or directory to search. Default workspace root."),
                        "file_glob": _str_prop("Only search files matching this glob, e.g. '*.py'."),
                        "context": _int_prop("Lines of context around each match. Default 0."),
                        "max_results": _int_prop("Maximum matching lines. Default 100."),
                        "ignore_case": {"type": "boolean", "description": "Case-insensitive search."},
                    },
                    ["pattern"],
                ),
                self._grep,
                read_only=True,
                parallel_safe=True,
            ),
            Tool(
                "write_file",
                "Create a file or replace its entire contents. Prefer edit_file for changes to "
                "existing files so unrelated code is preserved.",
                _obj(
                    {
                        "path": _str_prop("File path inside the workspace."),
                        "content": _str_prop("Complete new file contents."),
                    },
                    ["path", "content"],
                ),
                self._write_file,
            ),
            Tool(
                "edit_file",
                "Replace an exact string in a file. old_string must match uniquely unless "
                "replace_all is true. Returns a diff. This is the normal way to change code.",
                _obj(
                    {
                        "path": _str_prop("File path inside the workspace."),
                        "old_string": _str_prop("Exact text to replace, including indentation."),
                        "new_string": _str_prop("Replacement text. Empty string deletes the match."),
                        "replace_all": {
                            "type": "boolean",
                            "description": "Replace every occurrence instead of requiring a unique match.",
                        },
                    },
                    ["path", "old_string", "new_string"],
                ),
                self._edit_file,
            ),
            Tool(
                "multi_edit",
                "Apply several exact-string edits to one file in a single atomic write. "
                "Use when a change touches multiple regions of the same file.",
                _obj(
                    {
                        "path": _str_prop("File path inside the workspace."),
                        "edits": {
                            "type": "array",
                            "description": "Edits applied in order.",
                            "items": _obj(
                                {
                                    "old_string": _str_prop("Exact text to replace."),
                                    "new_string": _str_prop("Replacement text."),
                                    "replace_all": {"type": "boolean"},
                                },
                                ["old_string", "new_string"],
                            ),
                        },
                    },
                    ["path", "edits"],
                ),
                self._multi_edit,
            ),
            Tool(
                "run_shell",
                "Run a shell command in the workspace and return exit code plus output. "
                "Use it to run tests, installs, linters, and builds. Long output is saved to a file.",
                _obj(
                    {
                        "command": _str_prop("Command line to run."),
                        "timeout": _int_prop("Timeout in seconds. Default from config."),
                        "cwd": _str_prop("Working directory inside the workspace. Default workspace root."),
                    },
                    ["command"],
                ),
                self._run_shell,
            ),
            Tool(
                "check_python",
                "Compile a Python file and report syntax errors without running it.",
                _obj({"path": _str_prop("Python file path.")}, ["path"]),
                self._check_python,
                read_only=True,
                parallel_safe=True,
            ),
            Tool(
                "todo_write",
                "Maintain the task list for multi-step work. Send the complete list every time; "
                "status is one of pending, in_progress, completed.",
                _obj(
                    {
                        "todos": {
                            "type": "array",
                            "items": _obj(
                                {
                                    "content": _str_prop("What needs to be done."),
                                    "status": _str_prop("pending | in_progress | completed"),
                                },
                                ["content", "status"],
                            ),
                        }
                    },
                    ["todos"],
                ),
                self._todo_write,
                read_only=True,
            ),
            Tool(
                "ask_user",
                "Ask the user a question and pause until they answer. Use this when requirements "
                "are ambiguous, a destructive action needs confirmation, or a choice changes the design.",
                _obj(
                    {
                        "question": _str_prop("The question to ask."),
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional suggested answers.",
                        },
                    },
                    ["question"],
                ),
                self._ask_user,
                read_only=True,
            ),
            Tool(
                "spawn_agent",
                "Delegate a self-contained task to a subagent. The subagent has its own context "
                "window and session, works independently, and returns a summary. Use for parallel "
                "research, isolated fixes, or work that would flood this context.",
                _obj(
                    {
                        "task": _str_prop("Complete, self-contained instructions for the subagent."),
                        "system": _str_prop("Optional extra system guidance for the subagent."),
                    },
                    ["task"],
                ),
                self._spawn_agent,
                parallel_safe=True,
            ),
            Tool(
                "spawn_agents",
                "Run several independent subagent tasks in parallel. Each has its own session "
                "and returns its own summary. Prefer this over repeated spawn_agent calls.",
                _obj(
                    {
                        "tasks": {
                            "type": "array",
                            "items": _obj({"task": _str_prop("Subagent instructions.")}, ["task"]),
                        }
                    },
                    ["tasks"],
                ),
                self._spawn_agents,
                parallel_safe=True,
            ),
            Tool(
                "finish",
                "Declare the task complete. Provide a short summary of what changed and how it "
                "was verified. Call this only when the work is actually done.",
                _obj({"summary": _str_prop("What was accomplished and verified.")}, ["summary"]),
                self._finish,
                read_only=True,
            ),
        ]
        return {tool.name: tool for tool in tools}

    def enabled_tools(self) -> list[Tool]:
        tools = []
        for tool in self._tools.values():
            if self.read_only and not tool.read_only:
                continue
            if not self.allow_questions and tool.name == "ask_user":
                continue
            if not self.allow_subagents and tool.name in {"spawn_agent", "spawn_agents"}:
                continue
            tools.append(tool)
        return tools

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.enabled_tools()]

    def execute(self, name: str, arguments: dict[str, Any], call_id: str = "") -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name}")
        if self.read_only and not tool.read_only:
            raise ToolError(f"tool {name} is not available in plan mode")
        if not self.allow_questions and name == "ask_user":
            raise ToolError("subagents cannot ask the user questions; make a reasonable assumption and continue")
        if not self.allow_subagents and name in {"spawn_agent", "spawn_agents"}:
            raise ToolError("subagent depth limit reached; complete the task yourself")
        if not isinstance(arguments, dict):
            raise ToolError("tool arguments must be an object")
        return tool.handler(arguments, call_id)

    # -- path helpers -------------------------------------------------------
    def _resolve(self, raw: Any, label: str = "path") -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ToolError(f"{label} is required")
        text = raw.strip()
        if "\x00" in text:
            raise ToolError("path contains a null byte")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
            raise ToolError("path contains a control character")
        candidate = Path(text).expanduser()
        if candidate.drive and not candidate.root:
            raise ToolError("path contains a drive-relative component")
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.workspace) and not self.config.allow_outside_target:
            raise ToolError(f"path escapes the workspace: {resolved}")
        return resolved

    def _rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.workspace).as_posix()
        except ValueError:
            return str(path)

    def _offload(self, kind: str, text: str) -> str | None:
        if not self.config.offload_shell_output:
            return None
        try:
            directory = self.workspace / ".filament" / "runs"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{kind}-{uuid.uuid4().hex[:12]}.log"
            path.write_text(text, encoding="utf-8", errors="replace")
            return self._rel(path)
        except (OSError, ValueError):
            return None

    def _note_change(self, path: Path) -> None:
        if self.on_file_change:
            try:
                self.on_file_change(self._rel(path))
            except Exception:  # noqa: BLE001
                pass

    # -- file tools ---------------------------------------------------------
    def _read_file(self, args: dict[str, Any], _call_id: str) -> str:
        target = self._resolve(args.get("path"))
        if not target.exists():
            raise ToolError(f"file not found: {self._rel(target)}")
        if target.is_dir():
            raise ToolError(f"not a file: {self._rel(target)} (use list_dir)")
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolError(f"could not read {self._rel(target)}: {exc}") from None
        offset = _arg_int(args, "offset", 1, 1)
        limit = _arg_int(args, "limit", 400, 1, MAX_READ_LINES)
        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            return f"{self._rel(target)} (empty file, 0 bytes)"
        start = offset - 1
        if start >= total:
            raise ToolError(f"offset {offset} is past the end of {self._rel(target)} ({total} lines)")
        chunk = lines[start : start + limit]
        numbered_lines = [
            f"{start + index + 1:>6}  {line}" for index, line in enumerate(chunk)
        ]
        shown = len(numbered_lines)
        if sum(len(item) + 1 for item in numbered_lines) > MAX_READ_CHARS:
            # Trim whole lines so the footer never claims lines that were cut
            # mid-text.  An agent that trusts a reported range must get it.
            marker = "\n... [read output truncated]"
            budget = max(1, MAX_READ_CHARS - len(marker))
            kept: list[str] = []
            used = 0
            for item in numbered_lines:
                extra = len(item) + (1 if kept else 0)
                if kept and used + extra > budget:
                    break
                if not kept and extra > budget:
                    kept.append(item[:budget])
                    break
                kept.append(item)
                used += extra
            shown = len(kept)
            numbered = "\n".join(kept) + marker
        else:
            numbered = "\n".join(numbered_lines)
        end = start + shown
        footer = f"\n[{self._rel(target)}: {total} lines total, showing {start + 1}-{end}]"
        if end < total:
            footer += f"; call read_file with offset={end + 1} for more"
        return f"{self._rel(target)}\n{numbered}{footer}"

    def _write_file(self, args: dict[str, Any], _call_id: str) -> str:
        target = self._resolve(args.get("path"))
        content = args.get("content")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        existed = target.exists()
        newline = self._newline_style(target) if existed else "\n"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._write_text(target, content, newline=newline)
        except OSError as exc:
            raise ToolError(f"could not write {self._rel(target)}: {exc}") from None
        self._note_change(target)
        action = "overwrote" if existed else "created"
        return f"{action} {self._rel(target)} ({len(content)} chars, {content.count(chr(10)) + 1} lines)"

    def _write_text(
        self,
        target: Path,
        content: str,
        *,
        newline: str = "\n",
        final_newline: bool = True,
    ) -> None:
        if final_newline and not content.endswith("\n"):
            content += "\n"
        if newline == "\r\n":
            # Keep the file's existing line endings: rewriting a CRLF file as
            # LF turns one edited line into a whole-file diff and breaks files
            # that require CRLF (.bat, .cmd, some Windows sources).
            content = content.replace("\r\n", "\n").replace("\n", "\r\n")
        if self.config.atomic_writes:
            import tempfile

            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                if target.exists():
                    try:
                        os.chmod(temp_name, target.stat().st_mode)
                    except OSError:
                        pass
                os.replace(Path(temp_name), target)
            finally:
                leftover = Path(temp_name)
                if leftover.exists():
                    leftover.unlink(missing_ok=True)
        else:
            with target.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)

    def _newline_style(self, target: Path) -> str:
        """Return the dominant line ending of an existing file."""
        try:
            with target.open("rb") as handle:
                sample = handle.read(65_536)
        except OSError:
            return "\n"
        crlf = sample.count(b"\r\n")
        lf = sample.count(b"\n") - crlf
        return "\r\n" if crlf > lf else "\n"

    def _replace(
        self,
        text: str,
        old: str,
        new: str,
        replace_all: bool,
        label: str,
    ) -> tuple[str, str]:
        if not old:
            raise ToolError(f"{label}: old_string must not be empty")
        if old == new:
            raise ToolError(f"{label}: old_string and new_string are identical")
        count = text.count(old)
        note = ""
        if count == 0:
            # Fuzzy fallback: models often drop trailing whitespace or CRLF.
            normalized_text = _normalize_text(text)
            normalized_old = _normalize_text(old)
            normalized_new = _normalize_text(new)
            fuzzy = _count_occurrences(normalized_text, normalized_old)
            if fuzzy == 0:
                raise ToolError(
                    f"{label}: old_string was not found; read the file and copy the exact text"
                )
            if fuzzy > 1 and not replace_all:
                lines = _match_lines(normalized_text, normalized_old)
                raise ToolError(
                    f"{label}: old_string matches {fuzzy} places (lines {lines}); "
                    "include more context or set replace_all"
                )
            text = normalized_text
            old, new = normalized_old, normalized_new
            count = fuzzy
            note = " (matched after normalizing line endings/whitespace)"
        if count > 1 and not replace_all:
            lines = _match_lines(text, old)
            raise ToolError(
                f"{label}: old_string matches {count} places (lines {lines}); "
                "include more context or set replace_all"
            )
        replaced = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        return replaced, note

    def _edit_file(self, args: dict[str, Any], _call_id: str) -> str:
        target = self._resolve(args.get("path"))
        if not target.exists():
            raise ToolError(f"file not found: {self._rel(target)}")
        old = args.get("old_string")
        new = args.get("new_string")
        if not isinstance(old, str) or not old:
            raise ToolError("old_string must be a non-empty string")
        if not isinstance(new, str):
            raise ToolError("new_string must be a string")
        replace_all = _arg_bool(args, "replace_all")
        before = target.read_text(encoding="utf-8", errors="replace")
        after, note = self._replace(before, old, new, replace_all, self._rel(target))
        if after == before:
            return f"no change to {self._rel(target)}"
        self._write_text(target, after, newline=self._newline_style(target), final_newline=False)
        self._note_change(target)
        diff = _unified_diff(self._rel(target), before, after)
        count = before.count(old) if old in before else 1
        return f"edited {self._rel(target)} ({count}{'+' if replace_all else ''} replacement{note})\n{diff}".rstrip()

    def _multi_edit(self, args: dict[str, Any], _call_id: str) -> str:
        target = self._resolve(args.get("path"))
        edits = args.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ToolError("edits must be a non-empty list")
        if len(edits) > 40:
            raise ToolError("too many edits in one call (max 40)")
        if not target.exists():
            raise ToolError(f"file not found: {self._rel(target)}")
        before = target.read_text(encoding="utf-8", errors="replace")
        working = before
        for index, edit in enumerate(edits, 1):
            if not isinstance(edit, dict):
                raise ToolError(f"edit {index} must be an object")
            old = edit.get("old_string")
            new = edit.get("new_string")
            if not isinstance(old, str) or not old:
                raise ToolError(f"edit {index}: old_string must be a non-empty string")
            if not isinstance(new, str):
                raise ToolError(f"edit {index}: new_string must be a string")
            working, _ = self._replace(
                working,
                old,
                new,
                _arg_bool(edit, "replace_all"),
                f"edit {index}",
            )
        if working == before:
            return f"no change to {self._rel(target)}"
        self._write_text(target, working, newline=self._newline_style(target), final_newline=False)
        self._note_change(target)
        diff = _unified_diff(self._rel(target), before, working)
        return f"applied {len(edits)} edits to {self._rel(target)}\n{diff}".rstrip()

    def _list_dir(self, args: dict[str, Any], _call_id: str) -> str:
        target = self._resolve(args.get("path", "."))
        if not target.exists():
            raise ToolError(f"directory not found: {self._rel(target)}")
        if not target.is_dir():
            raise ToolError(f"not a directory: {self._rel(target)}")
        try:
            entries = sorted(
                target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
            )
        except OSError as exc:
            raise ToolError(f"could not list {self._rel(target)}: {exc}") from None
        lines: list[str] = []
        for item in entries:
            if item.name in IGNORED_DIRS:
                continue
            if item.is_dir():
                lines.append(f"{item.name}/")
                continue
            try:
                size = item.stat().st_size
            except OSError:
                size = 0
            lines.append(f"{item.name}  ({size} bytes)")
        if len(lines) > 300:
            lines = lines[:300] + [f"... {len(lines) - 300} more entries"]
        return "\n".join(lines) if lines else "(empty directory)"

    def _glob(self, args: dict[str, Any], _call_id: str) -> str:
        pattern = _arg_str(args, "pattern").strip().replace("\\", "/")
        assert isinstance(pattern, str)
        if pattern.startswith("/") or re.match(r"^[A-Za-z]:", pattern):
            raise ToolError("pattern must be relative to the workspace")
        parts = PurePosixPath(pattern).parts
        if any(part in {"..", ""} for part in parts):
            raise ToolError("pattern must not contain '..'")
        root = self._resolve(args.get("path", "."))
        if not root.is_dir():
            raise ToolError(f"not a directory: {self._rel(root)}")
        limit = _arg_int(args, "limit", 200, 1, 1000)
        matches: list[Path] = []
        try:
            for item in root.glob(pattern):
                if not item.is_file():
                    continue
                # Compare against the search root, not the absolute path: a
                # workspace that lives under a directory named "build" or
                # "dist" must still return files.
                try:
                    relative = item.relative_to(root)
                except ValueError:
                    relative = item
                if any(part in IGNORED_DIRS for part in relative.parts[:-1]):
                    continue
                matches.append(item)
        except (OSError, ValueError) as exc:
            raise ToolError(f"invalid glob pattern: {exc}") from None
        if not matches:
            return f"no files match {pattern!r}"
        def mtime(item: Path) -> float:
            try:
                return item.stat().st_mtime
            except OSError:
                return 0.0

        matches.sort(key=mtime, reverse=True)
        shown = matches[:limit]
        lines = [self._rel(item) for item in shown]
        if len(matches) > len(shown):
            lines.append(f"... {len(matches) - len(shown)} more files")
        return "\n".join(lines)

    def _grep(self, args: dict[str, Any], _call_id: str) -> str:
        pattern = _arg_str(args, "pattern")
        flags = re.IGNORECASE if _arg_bool(args, "ignore_case") else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from None
        root = self._resolve(args.get("path", "."))
        file_glob = str(args.get("file_glob") or "").strip()
        context = _arg_int(args, "context", 0, 0, 10)
        max_results = _arg_int(args, "max_results", 100, 1, 1000)
        results: list[str] = []
        total_chars = 0
        truncated = False

        def scan_file(path: Path) -> bool:
            nonlocal total_chars, truncated
            if file_glob and not fnmatch.fnmatch(path.name, file_glob):
                return True
            try:
                if path.stat().st_size > 2 * 1024 * 1024:
                    return True
                raw = path.read_bytes()
            except OSError:
                return True
            if b"\x00" in raw[:4096]:
                return True
            text = raw.decode("utf-8", "replace")
            lines = text.splitlines()
            rel = self._rel(path)
            for index, line in enumerate(lines):
                if not regex.search(line):
                    continue
                start = max(0, index - context)
                end = min(len(lines), index + context + 1)
                for offset in range(start, end):
                    marker = ":" if offset == index else "-"
                    entry = f"{rel}:{offset + 1}{marker}{lines[offset]}"
                    results.append(entry)
                    total_chars += len(entry) + 1
                    if total_chars > MAX_GREP_CHARS:
                        truncated = True
                        return False
                if len(results) >= max_results:
                    truncated = True
                    return False
            return True

        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = []
            for current, dirs, names in os.walk(root):
                dirs[:] = [name for name in dirs if name not in IGNORED_DIRS]
                for name in names:
                    files.append(Path(current) / name)
        else:
            raise ToolError(f"path not found: {self._rel(root)}")
        for path in files:
            if not scan_file(path):
                break
        if not results:
            return f"no matches for {pattern!r}"
        if truncated:
            results.append("... [grep stopped at the output limit]")
        return "\n".join(results)

    # -- shell --------------------------------------------------------------
    def _run_shell(self, args: dict[str, Any], _call_id: str) -> str:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolError("command is required")
        if len(command) > 20_000:
            raise ToolError("command is too long")
        if "\x00" in command:
            raise ToolError("command contains a null byte")
        if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in command) or "\x7f" in command:
            raise ToolError("command contains a control character")
        if self.config.command_allowlist:
            stripped = command.lstrip()
            if not any(stripped.startswith(prefix) for prefix in self.config.command_allowlist):
                raise ToolError("command is not on the configured allowlist")
            if any(ch in command for ch in ("\n", "\r", ";", "|", "&", "`", "$", "<", ">", "(", ")", "{", "}")):
                raise ToolError("command contains shell metacharacters not permitted with an allowlist")
        cwd = self._resolve(args.get("cwd") or ".")
        if not cwd.exists() or not cwd.is_dir():
            raise ToolError(f"working directory does not exist: {self._rel(cwd)}")
        timeout = coerce_timeout(args.get("timeout"), self.config.timeout_shell, "shell timeout")
        use_shell = self.config.use_shell
        try:
            completed = subprocess.run(
                prepare_command(command, use_shell),
                shell=use_shell,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            partial = str(exc.stdout or "") + str(exc.stderr or "")
            saved = None
            if len(partial) > self.config.max_shell_output:
                saved = self._offload("shell", partial)
                partial = _truncate_middle(partial, self.config.max_shell_output)
            note = f"\n[full output saved: {saved}]" if saved else ""
            return f"exit_code: timeout after {int(timeout)}s\n{partial}{note}"
        except (OSError, ValueError) as exc:
            return f"exit_code: error\n{exc}"
        output = (completed.stdout or "") + (completed.stderr or "")
        saved = None
        if len(output) > self.config.max_shell_output:
            saved = self._offload("shell", output)
            output = _truncate_middle(output, self.config.max_shell_output)
        note = f"\n[full output saved: {saved}]" if saved else ""
        return f"exit_code: {completed.returncode}\n{output}{note}"

    def _check_python(self, args: dict[str, Any], _call_id: str) -> str:
        target = self._resolve(args.get("path"))
        if target.suffix != ".py":
            raise ToolError("not a Python file")
        if not target.exists():
            raise ToolError(f"file not found: {self._rel(target)}")
        try:
            compile(target.read_text(encoding="utf-8"), str(target), "exec")
        except SyntaxError as exc:
            return f"syntax error: {exc}"
        except OSError as exc:
            raise ToolError(f"could not read {self._rel(target)}: {exc}") from None
        return "syntax ok"

    # -- planning and interaction ------------------------------------------
    def _todo_write(self, args: dict[str, Any], _call_id: str) -> str:
        raw = args.get("todos")
        if not isinstance(raw, list) or not raw:
            raise ToolError("todos must be a non-empty list")
        if len(raw) > 20:
            raise ToolError("keep the todo list to 20 items or fewer")
        cleaned: list[dict[str, Any]] = []
        for index, item in enumerate(raw, 1):
            if not isinstance(item, dict):
                raise ToolError(f"todo {index} must be an object")
            content = str(item.get("content") or "").strip()
            if not content:
                raise ToolError(f"todo {index}: content is required")
            status = str(item.get("status") or "pending").strip().lower()
            if status not in {"pending", "in_progress", "completed"}:
                raise ToolError(f"todo {index}: status must be pending, in_progress, or completed")
            cleaned.append({"content": content[:300], "status": status})
        self.todos = cleaned
        if self.on_todos:
            try:
                self.on_todos(cleaned)
            except Exception:  # noqa: BLE001
                pass
        marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
        return "\n".join(f"{marks[item['status']]} {item['content']}" for item in cleaned)

    def _ask_user(self, args: dict[str, Any], _call_id: str) -> str:
        question = _arg_str(args, "question").strip()
        raw_options = args.get("options") or []
        options: list[str] = []
        if isinstance(raw_options, list):
            options = [str(item).strip() for item in raw_options if str(item).strip()][:6]
        raise AskUser(question, options)

    # -- subagents ----------------------------------------------------------
    def _format_child(self, result: dict[str, Any]) -> str:
        session_id = str(result.get("session_id") or "")
        status = str(result.get("status") or "unknown")
        steps = result.get("steps", 0)
        content = str(result.get("content") or "").strip()
        header = f"[subagent {session_id} · {status} · {steps} steps]"
        return f"{header}\n{content or '(no output)'}"

    def _spawn_agent(self, args: dict[str, Any], call_id: str) -> str:
        spawn = self.spawn_fn
        if not spawn:
            raise ToolError("subagents are not available in this run")
        task = _arg_str(args, "task").strip()
        if len(task) > 50_000:
            raise ToolError("task is too long")
        system = str(args.get("system") or "").strip() or None
        return self._format_child(spawn(task, system, call_id))

    def _spawn_agents(self, args: dict[str, Any], call_id: str) -> str:
        spawn = self.spawn_fn
        if not spawn:
            raise ToolError("subagents are not available in this run")
        raw = args.get("tasks")
        if not isinstance(raw, list) or not raw:
            raise ToolError("tasks must be a non-empty list")
        if len(raw) > 6:
            raise ToolError("at most 6 parallel subagents per call")
        tasks: list[tuple[str, str | None]] = []
        for index, item in enumerate(raw, 1):
            if isinstance(item, str):
                task, system = item.strip(), None
            elif isinstance(item, dict):
                task = str(item.get("task") or "").strip()
                system = str(item.get("system") or "").strip() or None
            else:
                raise ToolError(f"task {index} must be a string or object")
            if not task:
                raise ToolError(f"task {index} is empty")
            tasks.append((task[:50_000], system))

        def run(index: int, task: str, system: str | None) -> str:
            try:
                result = spawn(task, system, f"{call_id}.{index}" if call_id else "")
                return f"--- task {index + 1} ---\n{self._format_child(result)}"
            except Exception as exc:  # noqa: BLE001 - one child failing must not lose the others
                return f"--- task {index + 1} ---\n[subagent failed: {exc}]"

        with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as pool:
            futures = [pool.submit(run, index, task, system) for index, (task, system) in enumerate(tasks)]
            return "\n\n".join(future.result() for future in futures)

    def _finish(self, args: dict[str, Any], _call_id: str) -> str:
        summary = str(args.get("summary") or "").strip() or "Task complete."
        raise FinishTask(summary)
