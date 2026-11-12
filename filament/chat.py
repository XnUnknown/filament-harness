"""Conversational session wrapper around :class:`filament.agent_loop.AgentLoop`.

This replaces the old plan-then-batch orchestrator.  A request now starts a real
agent loop that keeps working until it calls ``finish``, asks the user a
question, or hits a limit.  The class name is kept so existing integrations and
tests keep importing the same object.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path, PurePosixPath
from typing import Any

from .agent_loop import AgentLoop, LoopResult
from .agent_tools import Toolbox
from .compaction import compact_messages, estimate_tokens
from .config import HarnessConfig
from .llm import LLMClient, build_client
from .session import (
    Session,
    create_session,
    delete_session,
    list_sessions as list_session_files,
    load_session as load_session_file,
    save_session,
)

MAX_PLAN_TASKS = 6

_MODE_MAP = {
    "auto": "build",
    "accept": "build",
    "review": "build",
    "build": "build",
    "plan": "plan",
}

COMPAT_SYSTEM = (
    "You are Filament's main orchestrator. Convert the user's coding request into a "
    "JSON plan for focused coding agents. Return ONLY valid JSON. "
    'Schema: {"summary": "short summary", "tasks": [{"id": "task1", "task": "...", '
    '"target": "relative/output.py"}]}. Maximum 6 tasks; relative paths only; no markdown.'
)


def normalize_mode(value: Any) -> str:
    text = str(value or "build").strip().lower()
    return _MODE_MAP.get(text, "build")


def _sanitize_target(value: Any) -> str:
    """Accept only safe relative paths from untrusted planner output."""
    target = str(value or "").strip().replace("\\", "/")
    if "\x00" in target or any(ord(char) < 32 or ord(char) == 127 for char in target):
        return "output.py"
    if not target:
        return "output.py"
    candidate = PurePosixPath(target)
    if (
        target.startswith("/")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or any(part in {"", ".", ".."} or ":" in part for part in candidate.parts)
    ):
        return "output.py"
    normalized = str(candidate)
    return normalized if normalized and normalized != "." else "output.py"


def _extract_json(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("planner response must be a string")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object found in planner response")
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        if not isinstance(obj, dict):
            raise ValueError("planner response must be a JSON object")
        return obj
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if not isinstance(parsed, dict):
                raise ValueError("planner response must be a JSON object")
            return parsed
        raise


_SMALL_TALK = {
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "who are you",
    "how are you",
    "good morning",
    "good evening",
    "what can you do",
    "help",
}


def is_small_talk(request: str) -> bool:
    cleaned = request.strip().lower().rstrip("!.? ")
    return cleaned in _SMALL_TALK


class ChatOrchestrator:
    """A persistent chat session backed by the agent loop."""

    def __init__(
        self,
        config: HarnessConfig,
        shell_cwd: str = ".",
        mode: str = "build",
        session: Session | None = None,
        client: LLMClient | None = None,
        event_sink: Any = None,
    ):
        self.config = config
        self.shell_cwd = str(Path(shell_cwd).expanduser().resolve())
        self.mode = normalize_mode(mode)
        self.client = client or build_client(config.llm)
        # Kept for compatibility with callers that read file previews directly.
        self.toolbox = Toolbox(self.shell_cwd, config)
        self.session = session or create_session(self.shell_cwd, self.mode)
        self.session.mode = self.mode
        self.messages: list[dict[str, Any]] = list(self.session.messages)
        self.history: list[str] = [
            str(message.get("content", ""))
            for message in self.messages
            if message.get("role") == "user"
        ]
        self.sink = event_sink
        self._cancel = threading.Event()
        self._active: AgentLoop | None = None

    # -- configuration ------------------------------------------------------
    def apply_config(self, config: HarnessConfig) -> None:
        """Adopt a validated config and rebuild the provider and file toolbox."""
        self.config = config
        self.client = build_client(config.llm)
        self.toolbox = Toolbox(self.shell_cwd, config)

    def set_event_sink(self, sink: Any) -> None:
        self.sink = sink

    def set_mode(self, mode: str) -> str:
        self.mode = normalize_mode(mode)
        self.session.mode = self.mode
        save_session(self.session, self.shell_cwd)
        return self.mode

    # -- sessions -----------------------------------------------------------
    def new_session(self, mode: str | None = None) -> Session:
        self.mode = normalize_mode(mode or self.mode)
        self.session = create_session(self.shell_cwd, self.mode)
        self.messages = []
        self.history = []
        save_session(self.session, self.shell_cwd)
        return self.session

    def load_session(self, session_id: str) -> None:
        session = load_session_file(session_id, self.shell_cwd)
        self.session = session
        self.mode = normalize_mode(session.mode)
        self.session.mode = self.mode
        self.messages = list(session.messages)
        self.history = [
            str(message.get("content", ""))
            for message in self.messages
            if message.get("role") == "user"
        ]

    def save(self) -> Path:
        self.session.messages = self.messages
        self.session.mode = self.mode
        return save_session(self.session, self.shell_cwd)

    @staticmethod
    def list_sessions(cwd: str | Path) -> list[Session]:
        return list_session_files(cwd, kind="main")

    @staticmethod
    def delete_session(session_id: str, cwd: str | Path) -> bool:
        return delete_session(session_id, cwd)

    # -- agent turns --------------------------------------------------------
    def _make_loop(self) -> AgentLoop:
        return AgentLoop(
            self.config,
            self.shell_cwd,
            self.session,
            client=self.client,
            event_sink=self.sink,
            read_only=self.mode == "plan",
            cancel_event=self._cancel,
        )

    def _adopt(self, loop: AgentLoop, result: LoopResult) -> dict[str, Any]:
        self.messages = list(loop.messages)
        self.session = loop.session
        self.session.messages = self.messages
        self._active = None
        try:
            save_session(self.session, self.shell_cwd)
        except (OSError, ValueError):
            pass
        return result.to_dict()

    def send(self, message: str) -> dict[str, Any]:
        message = str(message or "").strip()
        if not message:
            raise ValueError("message must not be empty")
        if self.session.pending_question:
            raise ValueError("a question is waiting for an answer; call answer() first")
        self._cancel.clear()
        loop = self._make_loop()
        self._active = loop
        try:
            result = loop.run(message)
        finally:
            self._active = None
        self.history.append(message)
        return self._adopt(loop, result)

    def answer(self, question_id: str, text: str) -> dict[str, Any]:
        pending = self.session.pending_question
        if not pending:
            raise ValueError("no pending question")
        if question_id and str(pending.get("id") or "") != str(question_id):
            raise ValueError("question id does not match the pending question")
        self._cancel.clear()
        loop = self._make_loop()
        self._active = loop
        try:
            result = loop.resume(str(text))
        finally:
            self._active = None
        return self._adopt(loop, result)

    def skip_question(self, question_id: str) -> dict[str, Any]:
        return self.answer(question_id, "(skipped; use your best judgment and continue)")

    def cancel(self) -> bool:
        self._cancel.set()
        active = self._active
        if active is not None:
            try:
                active.cancel()
            except Exception:  # noqa: BLE001
                pass
        elif self.session.pending_question:
            # Ctrl+C at a question prompt must not strand the session: clear the
            # question so the next turn can repair the unanswered tool call.
            self.session.pending_question = None
            self.session.status = "idle"
            try:
                save_session(self.session, self.shell_cwd)
            except (OSError, ValueError):
                pass
        return True

    # -- context ------------------------------------------------------------
    def compress(self) -> dict[str, Any]:
        before = estimate_tokens(self.messages, self.config.chars_per_token)
        messages, changed, stats = compact_messages(
            self.client if self.config.compact_with_llm else None,
            self.messages,
            self.config.effective_max_tokens,
            self.config.chars_per_token,
            keep_recent=self.config.keep_recent_messages,
            threshold=0.999,
            use_llm=self.config.compact_with_llm,
        )
        if changed:
            self.messages = messages
            self.session.messages = messages
            save_session(self.session, self.shell_cwd)
        stats["before_tokens"] = before
        return stats

    def state(self) -> dict[str, Any]:
        return {
            "session_id": self.session.session_id,
            "title": self.session.display_title,
            "mode": self.mode,
            "status": self.session.status,
            "tokens": estimate_tokens(self.messages, self.config.chars_per_token),
            "limit": self.config.effective_max_tokens,
            "messages": len(self.messages),
            "todos": list(self.session.todos),
            "pending_question": self.session.pending_question,
            "child_sessions": list(self.session.child_sessions),
            "files_changed": list(self.session.files_changed),
            "usage": dict(self.session.usage),
            "provider": {
                "kind": self.config.llm.kind,
                "model": self.config.llm.model,
                "stream": bool(self.config.llm.stream),
            },
        }

    def is_small_talk(self, request: str) -> bool:
        return is_small_talk(request)
