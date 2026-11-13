"""Lightweight session persistence for Filament coding chats."""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


def _coerce_timestamp(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _clean_messages(value: Any) -> list[dict[str, Any]]:
    """Keep only messages a provider can accept.

    Session files can be hand-edited, truncated by a crash, or written by an
    older version.  One junk entry must not crash ``list_sessions`` or turn
    into an invalid request on the next turn.
    """
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if not isinstance(role, str) or role not in _MESSAGE_ROLES:
            continue
        if role == "tool" and not str(item.get("tool_call_id") or ""):
            continue
        cleaned.append(item)
    return cleaned


def _clean_str_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str)]


def _clean_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _safe_session_id(session_id: Any) -> str:
    """Validate a session id so it cannot escape the session directory."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    candidate = session_id.strip()
    if candidate in {".", ".."}:
        raise ValueError("invalid session id")
    if "/" in candidate or "\\" in candidate or "\x00" in candidate:
        raise ValueError("session_id must not contain path separators or null bytes")
    if any(ord(char) < 32 or ord(char) == 127 for char in candidate):
        raise ValueError("session_id must not contain control characters")
    if ":" in candidate:
        raise ValueError("session_id must not contain a colon")
    if candidate != Path(candidate).name:
        raise ValueError("session_id must be a base file name")
    return candidate


@dataclass
class Session:
    session_id: str
    cwd: str
    mode: str = "build"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: list[dict[str, Any]] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    # Agent-loop metadata. Defaults keep sessions written by older versions
    # loadable without a migration step.
    parent_id: str | None = None
    kind: str = "main"
    title: str = ""
    status: str = "idle"
    todos: list[dict[str, Any]] = field(default_factory=list)
    pending_question: dict[str, Any] | None = None
    child_sessions: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fields loaded from disk are untrusted: normalize the containers so a
        # single malformed file cannot break listing, loading, or a new turn.
        self.messages = _clean_messages(self.messages)
        self.files_changed = _clean_str_list(self.files_changed)
        self.todos = _clean_dict_list(self.todos)
        self.child_sessions = _clean_str_list(self.child_sessions)
        if not isinstance(self.usage, dict):
            self.usage = {}
        pending_valid = (
            isinstance(self.pending_question, dict)
            and bool(str(self.pending_question.get("id") or ""))
            and bool(str(self.pending_question.get("tool_call_id") or ""))
        )
        if not pending_valid:
            # A question without an id cannot be answered and would leave the
            # UI with a disabled composer; drop it so the session can continue.
            if self.pending_question is not None and self.status == "awaiting_input":
                self.status = "idle"
            self.pending_question = None
        if not isinstance(self.title, str):
            self.title = str(self.title or "")
        now = time.time()
        self.created_at = _coerce_timestamp(self.created_at, now)
        self.updated_at = _coerce_timestamp(self.updated_at, now)

    def touch(self) -> None:
        self.updated_at = time.time()

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title
        for message in self.messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                text = str(message.get("content") or "").strip()
                if text:
                    return text.splitlines()[0][:80]
        return self.session_id


def session_dir(cwd: str | Path) -> Path:
    return Path(cwd).resolve() / ".filament" / "sessions"


def create_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def create_session(
    cwd: str | Path,
    mode: str = "build",
    parent_id: str | None = None,
    kind: str = "main",
    title: str = "",
) -> Session:
    return Session(
        session_id=create_session_id(),
        cwd=str(Path(cwd).resolve()),
        mode=mode,
        parent_id=parent_id,
        kind=kind,
        title=title,
    )


def save_session(session: Session, cwd: str | Path | None = None) -> Path:
    session_id = _safe_session_id(session.session_id)
    base = Path(cwd or session.cwd).resolve()
    directory = session_dir(base)
    directory.mkdir(parents=True, exist_ok=True)
    session.touch()
    path = directory / f"{session_id}.json"
    resolved = path.resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise ValueError("session path escapes session directory")
    session.session_id = session_id
    payload = json.dumps(asdict(session), indent=2, ensure_ascii=False) + "\n"
    # Atomic replace: the UI may poll session files while a loop is running.
    fd, temp_name = tempfile.mkstemp(prefix=f".{session_id}.", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(temp_name, resolved)
    finally:
        leftover = Path(temp_name)
        if leftover.exists():
            leftover.unlink(missing_ok=True)
    return resolved


def load_session(session_id: str, cwd: str | Path) -> Session:
    safe_id = _safe_session_id(session_id)
    directory = session_dir(cwd).resolve()
    path = (directory / f"{safe_id}.json").resolve()
    if not path.is_relative_to(directory):
        raise ValueError("session path escapes session directory")
    if not path.exists():
        raise FileNotFoundError(f"session not found: {safe_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("session file is not a JSON object")
    known = set(Session.__dataclass_fields__)
    payload = {key: value for key, value in data.items() if key in known}
    # The file name is authoritative.  A copied or hand-edited session whose
    # inner id disagrees with its path must not keep the old identity: saving
    # it would overwrite the file that name belongs to and leave the requested
    # session untouched.
    payload["session_id"] = safe_id
    return Session(**payload)


def list_sessions(
    cwd: str | Path,
    kind: str | None = "main",
    include_children: bool = False,
) -> list[Session]:
    directory = session_dir(cwd)
    if not directory.exists():
        return []
    sessions: list[Session] = []
    for path in directory.glob("*.json"):
        try:
            session_id = _safe_session_id(path.stem)
        except ValueError:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            known = set(Session.__dataclass_fields__)
            payload = {key: value for key, value in data.items() if key in known}
            payload["session_id"] = session_id
            session = Session(**payload)
        except (OSError, TypeError, ValueError):
            continue
        if kind is not None and session.kind != kind and not include_children:
            continue
        sessions.append(session)
    sessions.sort(key=lambda item: item.updated_at, reverse=True)
    return sessions


def delete_session(session_id: str, cwd: str | Path) -> bool:
    safe_id = _safe_session_id(session_id)
    directory = session_dir(cwd).resolve()
    path = (directory / f"{safe_id}.json").resolve()
    if not path.is_relative_to(directory):
        raise ValueError("session path escapes session directory")
    if path.exists():
        path.unlink()
        return True
    return False

