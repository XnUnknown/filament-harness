"""Lightweight session persistence for Filament coding chats."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
    base = Path(cwd or session.cwd).resolve()
    directory = session_dir(base)
    directory.mkdir(parents=True, exist_ok=True)
    session.touch()
    path = directory / f"{session.session_id}.json"
    path.write_text(json.dumps(asdict(session), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_session(session_id: str, cwd: str | Path) -> Session:
    path = session_dir(cwd) / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"session not found: {session_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Session(**data)


def list_sessions(cwd: str | Path) -> list[Session]:
    directory = session_dir(cwd)
    if not directory.exists():
        return []
    sessions: list[Session] = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            sessions.append(Session(**json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, TypeError, ValueError):
            continue
    return sessions


def delete_session(session_id: str, cwd: str | Path) -> bool:
    path = session_dir(cwd) / f"{session_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
