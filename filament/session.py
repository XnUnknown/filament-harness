"""Lightweight session persistence for Filament coding chats."""
from __future__ import annotations

import json
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
    mode: str = "auto"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: list[dict[str, Any]] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = time.time()


def session_dir(cwd: str | Path) -> Path:
    return Path(cwd).resolve() / ".filament" / "sessions"


def create_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def create_session(cwd: str | Path, mode: str = "auto") -> Session:
    return Session(session_id=create_session_id(), cwd=str(Path(cwd).resolve()), mode=mode)


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
    resolved.write_text(json.dumps(asdict(session), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
    return Session(**data)


def list_sessions(cwd: str | Path) -> list[Session]:
    directory = session_dir(cwd)
    if not directory.exists():
        return []
    sessions: list[Session] = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions.append(Session(**data))
        except (OSError, TypeError, ValueError):
            continue
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

