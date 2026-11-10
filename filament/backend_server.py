"""Newline-delimited JSON backend used by the Filament Electron app."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, BinaryIO

from . import __version__
from .config import LLMConfig, HarnessConfig, load_config
from .chat import ChatOrchestrator
from .session import create_session

STATE: dict[str, Any] = {
    "workspace": None,
    "config": None,
    "chat": None,
}

# One request per line. Cap the frame so a runaway client cannot exhaust memory.
MAX_REQUEST_LINE_BYTES = 1_048_576

_INT_KEYS = {
    "max_tokens",
    "chars_per_token",
    "workers",
    "max_tasks",
    "timeout_shell",
    "max_shell_output",
}
_FLOAT_KEYS = {"budget_margin"}
_BOOL_KEYS = {
    "allow_outside_target",
    "atomic_writes",
    "use_shell",
    "offload_shell_output",
}


def _resolve_workspace(value: Any) -> Path:
    if not value:
        return Path.cwd().resolve()
    return Path(str(value)).expanduser().resolve()


def _load_workspace(workspace: str | None) -> dict[str, Any]:
    root = _resolve_workspace(workspace)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace does not exist: {root}")
    cfg = load_config(None)
    # Prefer workspace config if present.
    candidate = root / "filament.toml"
    if candidate.exists():
        cfg = load_config(candidate)
    chat = ChatOrchestrator(cfg, str(root), mode="auto")
    STATE["workspace"] = root
    STATE["config"] = cfg
    STATE["chat"] = chat
    return {
        "workspace": str(root),
        "config": _config_to_dict(cfg),
        "session_id": chat.session.session_id,
        "mode": chat.mode,
    }


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
        "command_allowlist": config.command_allowlist,
        "llm": {
            "kind": config.llm.kind,
            "model": config.llm.model,
            "base_url": config.llm.base_url,
            "api_key": None if not config.llm.api_key else "***",
            "timeout": config.llm.timeout,
            "reasoning_effort": config.llm.reasoning_effort,
        },
    }


def _coerce_config_value(key: str, value: Any) -> Any:
    """Reject wrong JSON types before they reach arithmetic or truthiness checks."""
    if key in _INT_KEYS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        return value
    if key in _FLOAT_KEYS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        return value
    if key in _BOOL_KEYS:
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
        return value
    if key == "command_allowlist":
        if not isinstance(value, list) or not all(isinstance(prefix, str) for prefix in value):
            raise ValueError("command_allowlist must be a list of strings")
        return list(value)
    return value


def _coerce_llm_value(key: str, value: Any) -> Any:
    if key == "timeout":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("llm.timeout must be an integer number of seconds")
        return value
    if key == "use_ollama_env":
        if not isinstance(value, bool):
            raise ValueError("llm.use_ollama_env must be a boolean")
        return value
    if not isinstance(value, str):
        raise ValueError(f"llm.{key} must be a string")
    return value


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
                setattr(candidate.llm, inner_key, _coerce_llm_value(inner_key, inner_value))
            continue
        if key not in HarnessConfig.__dataclass_fields__:
            raise ValueError(f"unknown config key: {key}")
        setattr(candidate, key, _coerce_config_value(key, value))
    candidate.validate()
    return candidate


def handle(method: str, params: dict[str, Any]) -> Any:
    if method == "ping":
        return {"version": __version__, "workspace": str(STATE["workspace"] or Path.cwd())}
    if method == "get_config":
        if STATE["config"] is None:
            _load_workspace(str(Path.cwd()))
        return _config_to_dict(STATE["config"])
    if method == "set_config":
        if STATE["config"] is None:
            _load_workspace(str(Path.cwd()))
        current = STATE["config"]
        if not isinstance(current, HarnessConfig):
            raise ValueError("configuration is not loaded")
        updates = params.get("config")
        if updates is None:
            updates = {}
        config = _apply_config(current, updates)
        STATE["config"] = config
        if STATE["chat"] is not None:
            STATE["chat"].apply_config(config)
        return _config_to_dict(config)
    if method == "select_workspace":
        return _load_workspace(params.get("path"))
    if method == "list_sessions":
        if STATE["chat"] is None:
            _load_workspace(str(Path.cwd()))
        return [{"session_id": s.session_id, "mode": s.mode, "updated_at": s.updated_at} for s in STATE["chat"].list_sessions(STATE["workspace"])]
    if method == "load_session":
        if STATE["chat"] is None:
            _load_workspace(str(Path.cwd()))
        STATE["chat"].load_session(str(params.get("session_id", "")))
        return {"session_id": STATE["chat"].session.session_id, "mode": STATE["chat"].mode}
    if method == "new_session":
        if STATE["chat"] is None:
            _load_workspace(str(Path.cwd()))
        STATE["chat"].session = create_session(STATE["workspace"], str(params.get("mode", "auto")))
        STATE["chat"].mode = STATE["chat"].session.mode
        STATE["chat"].messages = []
        return {"session_id": STATE["chat"].session.session_id, "mode": STATE["chat"].mode}
    if method == "chat":
        if STATE["chat"] is None:
            _load_workspace(str(Path.cwd()))
        request = str(params.get("message", "")).strip()
        if not request:
            raise ValueError("message must not be empty")
        chat = STATE["chat"]
        if chat.is_small_talk(request):
            content = chat.direct_chat(request)
            return {"type": "chat", "content": content, "session_id": chat.session.session_id}
        summary, tasks = chat.plan(request)
        asked = False
        mode = params.get("mode", chat.mode)
        if mode != "auto":
            asked = True
        result = {"type": "plan", "summary": summary, "tasks": tasks, "session_id": chat.session.session_id}
        if mode == "auto":
            run_result = chat.run_plan(tasks, review=(mode in {"review", "accept"}), fix=(mode in {"auto", "review", "accept"}))
            result["execution"] = run_result
        return result
    if method == "run_task":
        if STATE["chat"] is None:
            _load_workspace(str(Path.cwd()))
        tasks = params.get("tasks", [])
        if not isinstance(tasks, list):
            raise ValueError("tasks must be a list")
        chat = STATE["chat"]
        result = chat.run_plan(tasks, review=bool(params.get("review", True)), fix=bool(params.get("fix", True)), max_fixes=params.get("max_fixes"))
        return result
    if method == "read_file":
        if STATE["chat"] is None:
            _load_workspace(str(Path.cwd()))
        return {"content": STATE["chat"].toolbox.read_file(str(params.get("path", "")), params.get("max_chars", 12000))}
    if method == "compress_context":
        if STATE["chat"] is None:
            _load_workspace(str(Path.cwd()))
        changed = STATE["chat"].compress_if_needed()
        return {"compressed": changed}
    raise ValueError(f"unknown method: {method}")


def _encode(payload: dict[str, Any]) -> str:
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


def _write_line(stream: BinaryIO, text: str) -> None:
    stream.write(text.encode("utf-8", "replace") + b"\n")
    stream.flush()


def _discard_line(stream: BinaryIO) -> None:
    while True:
        chunk = stream.readline(MAX_REQUEST_LINE_BYTES + 1)
        if not chunk or chunk.endswith(b"\n"):
            return


def main() -> None:
    # Byte-level UTF-8 framing: redirected stdio on Windows defaults to the
    # locale codec, which cannot encode most model output.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
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
        _write_line(stdout, respond(line))


if __name__ == "__main__":
    main()
