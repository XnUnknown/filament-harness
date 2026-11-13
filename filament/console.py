"""Dependency-free terminal rendering for agent events.

Used by ``filament`` (the interactive CLI) and by the TUI chat screen.  The
renderer is a plain callable so it can be handed to ``ChatOrchestrator`` as its
event sink.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, TextIO


_TOOL_ICONS = {
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
    "multi_edit": "edit",
    "list_dir": "list",
    "glob": "find",
    "grep": "grep",
    "run_shell": "shell",
    "check_python": "check",
    "todo_write": "todo",
    "ask_user": "ask",
    "spawn_agent": "agent",
    "spawn_agents": "agent",
    "finish": "finish",
}


def tool_summary(name: str, args: dict[str, Any] | None) -> str:
    args = args or {}
    if name in {"read_file", "write_file", "edit_file", "multi_edit", "check_python"}:
        return str(args.get("path") or "")
    if name == "run_shell":
        return _one_line(str(args.get("command") or ""), 90)
    if name in {"grep", "glob"}:
        pattern = str(args.get("pattern") or "")
        path = str(args.get("path") or "")
        return f"{pattern}" + (f"  in {path}" if path and path != "." else "")
    if name == "list_dir":
        return str(args.get("path") or ".")
    if name == "spawn_agent":
        return _one_line(str(args.get("task") or ""), 90)
    if name == "spawn_agents":
        tasks = args.get("tasks") or []
        return f"{len(tasks)} parallel tasks"
    if name == "ask_user":
        return _one_line(str(args.get("question") or ""), 90)
    if name == "todo_write":
        return f"{len(args.get('todos') or [])} items"
    if name == "finish":
        return _one_line(str(args.get("summary") or ""), 90)
    return ""


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "..."


def _first_line(text: str, limit: int = 160) -> str:
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return _one_line(line, limit)
    return ""


class ConsoleRenderer:
    """Print agent events as compact terminal lines."""

    def __init__(self, stream: TextIO | None = None, color: bool | None = None):
        self.stream = stream or sys.stdout
        if color is None:
            color = bool(getattr(self.stream, "isatty", lambda: False)()) and os.environ.get("NO_COLOR") is None
        self.color = color
        self._delta_in_turn = False
        self._line_open = False
        self._prefix_pending = False
        self._turn_child = False

    # -- low level ----------------------------------------------------------
    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _end_line(self) -> None:
        if self._line_open:
            self._write("\n")
            self._line_open = False

    def _line(self, text: str) -> None:
        self._end_line()
        self._write(text + "\n")

    # -- sink ---------------------------------------------------------------
    def __call__(self, event: dict[str, Any]) -> None:
        self.handle(event)

    def handle(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type") or "")
        if kind == "assistant_start":
            self._delta_in_turn = False
            self._prefix_pending = True
            self._turn_child = bool(event.get("subagent_id"))
            return
        if kind == "delta":
            text = str(event.get("text") or "")
            if not text:
                return
            if self._prefix_pending:
                self._end_line()
                pad = "  | " if self._turn_child else ""
                self._write(pad + self._paint("filament", "33") + "  ")
                self._prefix_pending = False
                self._line_open = True
            self._write(text)
            self._delta_in_turn = True
            return
        if kind == "assistant_end":
            if self._delta_in_turn:
                self._end_line()
            elif str(event.get("content") or "").strip():
                pad = "  | " if event.get("subagent_id") else ""
                self._line(pad + self._paint("filament", "33") + "  " + str(event["content"]).strip())
            self._prefix_pending = False
            return
        if kind == "tool_start":
            name = str(event.get("name") or "")
            summary = tool_summary(name, event.get("args") or {})
            label = _TOOL_ICONS.get(name, name)
            pad = "  | " if event.get("subagent_id") else "  "
            self._line(self._paint(f"{pad}-> {label}", "36") + (f"  {summary}" if summary else ""))
            return
        if kind == "tool_end":
            name = str(event.get("name") or "")
            label = _TOOL_ICONS.get(name, name)
            ok = bool(event.get("ok", True))
            content = str(event.get("content") or "")
            duration = event.get("duration_ms")
            tail = f" ({duration} ms)" if isinstance(duration, int) else ""
            pad = "  | " if event.get("subagent_id") else "  "
            if ok:
                self._line(self._paint(f"{pad}<- {label} ok", "32") + tail)
            else:
                detail = _first_line(content)
                self._line(self._paint(f"{pad}<- {label} failed", "31") + (f": {detail}" if detail else "") + tail)
            return
        if kind == "subagent_start":
            task = _one_line(str(event.get("task") or ""), 100)
            self._line(self._paint("  [subagent start]", "33") + f" {task}")
            return
        if kind == "subagent_end":
            status = str(event.get("status") or "")
            steps = event.get("steps", 0)
            tokens = event.get("tokens", 0)
            self._line(self._paint("  [subagent end]", "33") + f" {status}, {steps} steps, {tokens} tokens")
            return
        if kind == "todos":
            todos = event.get("todos") or []
            if not todos:
                return
            self._line(self._paint("  plan:", "2"))
            for item in todos:
                mark = {"completed": "[x]", "in_progress": "[~]"}.get(str(item.get("status")), "[ ]")
                self._line(f"    {mark} {item.get('content', '')}")
            return
        if kind == "question":
            question = str(event.get("question") or "")
            self._line(self._paint("  ? " + question, "33"))
            for option in event.get("options") or []:
                self._line(f"      - {option}")
            return
        if kind == "compaction":
            before = event.get("before_tokens")
            after = event.get("after_tokens")
            if isinstance(before, int) and isinstance(after, int):
                self._line(self._paint(f"  [context compacted: {before} -> {after} tokens]", "2"))
            return
        if kind == "error":
            self._line(self._paint("  error: " + str(event.get("message") or ""), "31"))
            return
        if kind == "done":
            if event.get("subagent_id"):
                # The parent's subagent_end line and the finish argument already
                # carry the child's summary.
                return
            content = str(event.get("content") or "").strip()
            status = str(event.get("status") or "done")
            if content:
                if status == "done":
                    self._line(self._paint("  done: " + content, "32"))
                else:
                    self._line(self._paint(f"  [{status}] {content}", "33"))
            return
        # status, heartbeat, context, session: intentionally quiet


class AgentConsole:
    """Drive one chat session from a terminal prompt."""

    def __init__(
        self,
        chat,
        renderer: ConsoleRenderer | None = None,
        input_fn: Callable[[str], str] = input,
    ):
        self.chat = chat
        self.renderer = renderer or ConsoleRenderer()
        self.chat.set_event_sink(self.renderer)
        self.input_fn = input_fn

    def run_turn(self, message: str) -> dict[str, Any]:
        result = self.chat.send(message)
        # Keep answering until the turn actually ends.  A fixed number of
        # rounds left the session holding a pending question that this REPL
        # cannot answer, so every later message failed until /new.  Each round
        # waits for real user input, so this cannot spin on its own.
        while result.get("status") == "awaiting_input":
            question = result.get("question") or {}
            prompt = f"? {question.get('question', 'Question')}\n> "
            answer = self.input_fn(prompt).strip()
            if not answer:
                result = self.chat.skip_question(str(question.get("id") or ""))
            else:
                result = self.chat.answer(str(question.get("id") or ""), answer)
        return result

    def print_result(self, result: dict[str, Any]) -> None:
        status = result.get("status")
        if status in {"max_steps", "stalled", "cancelled", "error", "empty"}:
            content = str(result.get("content") or "").strip()
            if content:
                self.renderer._line("  " + content)
        tokens = result.get("tokens")
        steps = result.get("steps")
        if isinstance(tokens, int):
            self.renderer._line(self.renderer._paint(f"  [{steps} steps, ~{tokens} tokens]", "2"))
