"""Newline-delimited JSON backend used by the Filament Electron app.

Protocol
--------
One JSON object per line on stdin.  Requests answer with::

    {"id": 1, "ok": true, "result": {...}}
    {"id": 1, "ok": false, "error": "..."}

Agent runs also stream notifications while they work::

    {"event": "delta", "session_id": "...", "text": "Hel"}

Notifications have no ``id`` and are dispatched by the Electron main process to
the renderer.  Long methods (``chat``, ``answer``) run on a worker thread so the
reader stays free to accept ``cancel``.
"""
from __future__ import annotations

import copy
import json
import sys
import threading
from pathlib import Path
from typing import Any, BinaryIO

from . import __version__
from .chat import ChatOrchestrator
from .config import LLMConfig, HarnessConfig, load_config
from .session import list_sessions as list_session_files

STATE: dict[str, Any] = {
    "workspace": None,
    "config": None,
    "chat": None,
    "busy": False,
}

# One request per line. Cap the frame so a runaway client cannot exhaust memory.
MAX_REQUEST_LINE_BYTES = 1_048_576
ASYNC_METHODS = {"chat", "answer", "skip_question"}

EVENT_STREAM: BinaryIO | None = None
_WRITE_LOCK = threading.Lock()
_STATE_LOCK = threading.RLock()


def _write_line(stream: BinaryIO, text: str) -> None:
    with _WRITE_LOCK:
        stream.write(text.encode("utf-8", "replace") + b"\n")
        stream.flush()


def _emit(event: dict[str, Any]) -> None:
    """Forward one agent event to the UI as a notification line."""
    stream = EVENT_STREAM
    if stream is None:
        return
    payload: dict[str, Any] = {"event": str(event.get("type") or "event")}
    for key, value in event.items():
        if key == "type":
            continue
        payload[key] = value
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return
    _write_line(stream, text)


def _resolve_workspace(value: Any) -> Path:
    if not value:
        return Path.cwd().resolve()
    return Path(str(value)).expanduser().resolve()


def _load_workspace(workspace: str | None) -> dict[str, Any]:
    root = _resolve_workspace(workspace)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace does not exist: {root}")
    cfg = load_config(None)
    candidate = root / "filament.toml"
    if candidate.exists():
        cfg = load_config(candidate)
    chat = ChatOrchestrator(cfg, str(root), mode="build", event_sink=_emit)
    with _STATE_LOCK:
        STATE["workspace"] = root
        STATE["config"] = cfg
        STATE["chat"] = chat
    _emit(
        {
            "type": "session",
            "session_id": chat.session.session_id,
            "title": chat.session.display_title,
            "mode": chat.mode,
        }
    )
    return {
        "workspace": str(root),
        "config": _config_to_dict(cfg),
        "session_id": chat.session.session_id,
        "mode": chat.mode,
    }


def _require_chat() -> ChatOrchestrator:
    chat = STATE["chat"]
    if chat is None:
        _load_workspace(str(Path.cwd()))
        chat = STATE["chat"]
    if not isinstance(chat, ChatOrchestrator):
        raise ValueError("chat session is not loaded")
    return chat


def _require_idle(action: str) -> None:
    """Reject a state replacement that a running turn would silently undo."""
    with _STATE_LOCK:
        if STATE["busy"]:
            raise ValueError(f"cannot {action} while an agent turn is running")


def _config_to_dict(config: HarnessConfig) -> dict[str, Any]:
    return {
        "max_tokens": config.max_tokens,
        "chars_per_token": config.chars_per_token,
        "budget_margin": config.budget_margin,
        "workers": config.workers,
        "max_tasks": config.max_tasks,
        "timeout_shell": config.timeout_shell,
        "max_shell_output": config.max_shell_output,
        "allow_outside_target": config.allow_outside_target,
        "atomic_writes": config.atomic_writes,
        "use_shell": config.use_shell,
        "offload_shell_output": config.offload_shell_output,
        "max_steps": config.max_steps,
        "subagent_max_steps": config.subagent_max_steps,
        "subagent_max_tokens": config.subagent_max_tokens,
        "tool_output_limit": config.tool_output_limit,
        "keep_recent_messages": config.keep_recent_messages,
        "compact_threshold": config.compact_threshold,
        "compact_with_llm": config.compact_with_llm,
        "max_depth": config.max_depth,
        "command_allowlist": config.command_allowlist,
        "llm": {
            "kind": config.llm.kind,
            "model": config.llm.model,
            "base_url": config.llm.base_url,
            "api_key": None if not config.llm.api_key else "***",
            "timeout": config.llm.timeout,
            "reasoning_effort": config.llm.reasoning_effort,
            "stream": config.llm.stream,
            "max_output_tokens": config.llm.max_output_tokens,
        },
    }


def _apply_config(config: HarnessConfig, updates: Any) -> HarnessConfig:
    """Return a validated copy; the caller's config is never mutated on failure."""
    if not isinstance(updates, dict):
        raise ValueError("config must be an object")
    candidate = copy.deepcopy(config)
    for key, value in updates.items():
        if value is None:
            continue
        if key == "llm":
            if not isinstance(value, dict):
                raise ValueError("llm must be an object")
            for inner_key, inner_value in value.items():
                if inner_key not in LLMConfig.__dataclass_fields__:
                    raise ValueError(f"unknown llm config key: {inner_key}")
                if inner_value is None:
                    continue
                setattr(candidate.llm, inner_key, inner_value)
            continue
        if key not in HarnessConfig.__dataclass_fields__:
            raise ValueError(f"unknown config key: {key}")
        setattr(candidate, key, value)
    candidate.validate()
    return candidate


def _session_payload(chat: ChatOrchestrator) -> dict[str, Any]:
    return {
        "session_id": chat.session.session_id,
        "title": chat.session.display_title,
        "mode": chat.mode,
        "kind": chat.session.kind,
        "parent_id": chat.session.parent_id,
        "status": chat.session.status,
    }


def _turn_payload(result: dict[str, Any]) -> dict[str, Any]:
    chat = STATE.get("chat")
    payload = dict(result)
    payload["type"] = "question" if result.get("status") == "awaiting_input" else "done"
    if isinstance(chat, ChatOrchestrator):
        payload["state"] = chat.state()
    return payload


def _list_dir_entries(path: str, limit: int = 500) -> list[dict[str, Any]]:
    root = Path(STATE["workspace"] or Path.cwd()).resolve()
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if not target.is_relative_to(root):
        raise ValueError("path escapes the workspace")
    if not target.is_dir():
        raise ValueError(f"not a directory: {target}")
    entries: list[dict[str, Any]] = []
    for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if item.name in {".git", ".filament", "__pycache__", "node_modules"}:
            continue
        try:
            size = item.stat().st_size
        except OSError:
            size = 0
        entries.append(
            {
                "name": item.name,
                "path": item.relative_to(root).as_posix(),
                "type": "dir" if item.is_dir() else "file",
                "size": size,
            }
        )
        if len(entries) >= limit:
            break
    return entries


def handle(method: str, params: dict[str, Any]) -> Any:
    if method == "ping":
        return {"version": __version__, "workspace": str(STATE["workspace"] or Path.cwd())}

    if method == "get_config":
        if STATE["config"] is None:
            _load_workspace(str(STATE["workspace"] or Path.cwd()))
        return _config_to_dict(STATE["config"])

    if method == "set_config":
        if STATE["config"] is None:
            _load_workspace(str(STATE["workspace"] or Path.cwd()))
        current = STATE["config"]
        if not isinstance(current, HarnessConfig):
            raise ValueError("configuration is not loaded")
        updates = params.get("config") or {}
        config = _apply_config(current, updates)
        with _STATE_LOCK:
            STATE["config"] = config
        chat = STATE.get("chat")
        if isinstance(chat, ChatOrchestrator):
            chat.apply_config(config)
        return _config_to_dict(config)

    if method == "select_workspace":
        _require_idle("switch workspaces")
        return _load_workspace(params.get("path"))

    if method == "get_state":
        chat = _require_chat()
        config = STATE["config"]
        if not isinstance(config, HarnessConfig):
            raise ValueError("configuration is not loaded")
        return {
            "workspace": str(STATE["workspace"]),
            "busy": bool(STATE["busy"]),
            "state": chat.state(),
            "config": _config_to_dict(config),
        }

    if method == "list_sessions":
        workspace = STATE["workspace"] or Path.cwd()
        include_children = bool(params.get("include_children"))
        sessions = list_session_files(
            workspace, kind=None if include_children else "main"
        )
        return [
            {
                "session_id": session.session_id,
                "title": session.display_title,
                "mode": session.mode,
                "kind": session.kind,
                "parent_id": session.parent_id,
                "status": session.status,
                "updated_at": session.updated_at,
                "tokens": session.usage.get("tokens", 0),
            }
            for session in sessions
        ]

    if method == "load_session":
        _require_idle("load a session")
        chat = _require_chat()
        chat.load_session(str(params.get("session_id", "")))
        _emit(
            {
                "type": "session",
                "session_id": chat.session.session_id,
                "title": chat.session.display_title,
                "mode": chat.mode,
            }
        )
        return {
            "session": _session_payload(chat),
            "messages": chat.messages,
            "state": chat.state(),
        }

    if method == "new_session":
        _require_idle("start a new session")
        chat = _require_chat()
        chat.new_session(str(params.get("mode", chat.mode)))
        _emit(
            {
                "type": "session",
                "session_id": chat.session.session_id,
                "title": chat.session.display_title,
                "mode": chat.mode,
            }
        )
        return {"session": _session_payload(chat), "state": chat.state()}

    if method == "set_mode":
        chat = _require_chat()
        mode = chat.set_mode(str(params.get("mode", "build")))
        _emit(
            {
                "type": "session",
                "session_id": chat.session.session_id,
                "title": chat.session.display_title,
                "mode": mode,
            }
        )
        return {"mode": mode, "state": chat.state()}

    if method == "delete_session":
        chat = _require_chat()
        return {
            "deleted": chat.delete_session(
                str(params.get("session_id", "")), str(STATE["workspace"])
            )
        }

    if method == "get_messages":
        chat = _require_chat()
        return {
            "session": _session_payload(chat),
            "messages": chat.messages,
            "state": chat.state(),
        }

    if method == "chat":
        chat = _require_chat()
        chat.set_event_sink(_emit)
        with _STATE_LOCK:
            if STATE["busy"]:
                raise ValueError("an agent turn is already running")
            STATE["busy"] = True
        try:
            mode = params.get("mode")
            if mode:
                chat.set_mode(str(mode))
            message = str(params.get("message", "")).strip()
            if not message:
                raise ValueError("message must not be empty")
            _emit(
                {
                    "type": "session",
                    "session_id": chat.session.session_id,
                    "title": chat.session.display_title,
                    "mode": chat.mode,
                }
            )
            result = chat.send(message)
        finally:
            STATE["busy"] = False
        return _turn_payload(result)

    if method == "answer":
        chat = _require_chat()
        chat.set_event_sink(_emit)
        with _STATE_LOCK:
            if STATE["busy"]:
                raise ValueError("an agent turn is already running")
            STATE["busy"] = True
        try:
            question_id = str(params.get("question_id", ""))
            text = str(params.get("text", "")).strip()
            if not text:
                raise ValueError("answer must not be empty")
            result = chat.answer(question_id, text)
        finally:
            STATE["busy"] = False
        return _turn_payload(result)

    if method == "skip_question":
        chat = _require_chat()
        chat.set_event_sink(_emit)
        with _STATE_LOCK:
            if STATE["busy"]:
                raise ValueError("an agent turn is already running")
            STATE["busy"] = True
        try:
            result = chat.skip_question(str(params.get("question_id", "")))
        finally:
            STATE["busy"] = False
        return _turn_payload(result)

    if method == "cancel":
        chat = STATE.get("chat")
        if isinstance(chat, ChatOrchestrator):
            chat.cancel()
        return {"cancelling": True}

    if method == "compress_context":
        chat = _require_chat()
        return chat.compress()

    if method == "read_file":
        chat = _require_chat()
        path = str(params.get("path", ""))
        content = chat.toolbox.read_file(path, params.get("max_chars", 12_000))
        return {"path": path, "content": content}

    if method == "list_dir":
        if STATE["workspace"] is None:
            _load_workspace(str(Path.cwd()))
        return {
            "entries": _list_dir_entries(str(params.get("path", "."))),
            "workspace": str(STATE["workspace"]),
        }

    raise ValueError(f"unknown method: {method}")


def _encode(payload: dict[str, Any]) -> str:
    # Strict on purpose: an unserializable result is a protocol error the client
    # must see, not something to paper over with repr().
    return json.dumps(payload, ensure_ascii=False)


def _error_response(request_id: Any, message: str) -> str:
    try:
        return _encode({"id": request_id, "ok": False, "error": message})
    except (TypeError, ValueError):
        return '{"id": null, "ok": false, "error": "response could not be serialized"}'


def respond(line: str) -> str:
    """Answer one protocol line; a bad request costs one response, not the session."""
    request_id = None
    try:
        request = json.loads(line)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        request_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("method must be a non-empty string")
        params = request.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        return _encode({"id": request_id, "ok": True, "result": handle(method, params)})
    except Exception as exc:  # noqa: BLE001
        return _error_response(request_id, str(exc))


def _request_method(line: str) -> str:
    try:
        request = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(request, dict):
        return ""
    method = request.get("method")
    return method if isinstance(method, str) else ""


def _discard_line(stream: Any) -> None:
    while True:
        chunk = stream.readline(MAX_REQUEST_LINE_BYTES + 1)
        if not chunk or chunk.endswith(b"\n"):
            return


def main() -> None:
    global EVENT_STREAM
    # Byte-level UTF-8 framing: redirected stdio on Windows defaults to the
    # locale codec, which cannot encode most model output.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    EVENT_STREAM = stdout
    while True:
        raw = stdin.readline(MAX_REQUEST_LINE_BYTES + 1)
        if not raw:
            break
        if len(raw) > MAX_REQUEST_LINE_BYTES and not raw.endswith(b"\n"):
            _discard_line(stdin)
            _write_line(
                stdout,
                _error_response(None, f"request line exceeds {MAX_REQUEST_LINE_BYTES} bytes"),
            )
            continue
        try:
            line = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            _write_line(stdout, _error_response(None, "request line is not valid UTF-8"))
            continue
        if not line:
            continue
        if _request_method(line) in ASYNC_METHODS:
            thread = threading.Thread(
                target=lambda payload, out=stdout: _write_line(out, respond(payload)),
                args=(line,),
                name="filament-request",
                daemon=True,
            )
            thread.start()
        else:
            _write_line(stdout, respond(line))


if __name__ == "__main__":
    main()
