"""Lightweight session persistence for Filament coding chats."""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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

    def touch(self) -> None:
        self.updated_at = time.time()

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title
        for message in self.messages:
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
    return Session(**{key: value for key, value in data.items() if key in known})


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
            data = json.loads(path.read_text(encoding="utf-8"))
            known = set(Session.__dataclass_fields__)
            session = Session(**{key: value for key, value in data.items() if key in known})
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

