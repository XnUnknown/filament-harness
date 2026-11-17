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
        calls = item.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                # A scalar here used to survive loading and then crash token
                # estimation (loop construction / state()) and the renderer's
                # history rebuild before any provider could reject it.
                item = dict(item)
                item.pop("tool_calls", None)
            else:
                valid = [call for call in calls if isinstance(call, dict)]
                if len(valid) != len(calls):
                    item = dict(item)
                    if valid:
                        item["tool_calls"] = valid
                    else:
                        item.pop("tool_calls", None)
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
        pending = self.pending_question
        if (
            isinstance(pending, dict)
            and str(pending.get("id") or "")
            and str(pending.get("tool_call_id") or "")
        ):
            # The renderer iterates options and prints the question.  A
            # hand-edited session can carry a string or a number here; the
            # question card would throw on the first missing form control and
            # the UI would report the backend as unreachable.
            options = pending.get("options")
            if not isinstance(options, list):
                options = []
            pending["options"] = [
                str(item) for item in options if item is not None and str(item).strip()
            ][:6]
            if not isinstance(pending.get("question"), str):
                pending["question"] = str(pending.get("question") or "")
        else:
            # A question without an id cannot be answered and would leave the
            # UI with a disabled composer; drop it so the session can continue.
            if pending is not None and self.status == "awaiting_input":
                self.status = "idle"
            self.pending_question = None
        if self.status == "running":
            # The only writer of a "running" status is a live loop, so a value
            # read back from disk can only describe a turn whose process died.
            # Nothing can stop or join that run, and showing it as running in
            # the session list or status line is stale; report idle instead.
            self.status = "idle"
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


def _session_children(path: Path, directory: Path, parent_id: str) -> list[str]:
    """Return child session ids that still name ``parent_id`` as their parent.

    The parent link is checked before deleting anything: a copied or hand-edited
    session file must not be able to take an unrelated session down with it.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get("child_sessions")
    if not isinstance(raw, list):
        return []
    children: list[str] = []
    for item in raw:
        try:
            child_id = _safe_session_id(item)
        except ValueError:
            continue
        child_path = (directory / f"{child_id}.json").resolve()
        if not child_path.is_relative_to(directory) or not child_path.exists():
            continue
        try:
            child_data = json.loads(child_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(child_data, dict) and child_data.get("parent_id") == parent_id:
            children.append(child_id)
    return children


def delete_session(
    session_id: str,
    cwd: str | Path,
    *,
    _seen: set[str] | None = None,
) -> bool:
    """Delete a session and the subagent sessions it created.

    Subagent sessions are hidden from the session list, so deleting only the
    main file left them on disk with no way to remove them from the UI.  The
    walk is cycle-safe and only follows a child link whose file still points
    back at the session being deleted.
    """
    safe_id = _safe_session_id(session_id)
    directory = session_dir(cwd).resolve()
    path = (directory / f"{safe_id}.json").resolve()
    if not path.is_relative_to(directory):
        raise ValueError("session path escapes session directory")
    seen = _seen if _seen is not None else set()
    if safe_id in seen:
        return False
    seen.add(safe_id)
    if not path.exists():
        return False
    children = _session_children(path, directory, safe_id)
    path.unlink()
    for child_id in children:
        try:
            delete_session(child_id, cwd, _seen=seen)
        except (OSError, ValueError):
            # The parent is already gone; a child that cannot be removed must
            # not report the whole delete as failed.
            continue
    return True

