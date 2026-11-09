"""Newline-delimited JSON backend used by the Filament Electron app."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .config import HarnessConfig, load_config
from .chat import ChatOrchestrator
from .session import create_session

STATE: dict[str, Any] = {
    "workspace": None,
    "config": None,
    "chat": None,
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


def _apply_config(config: HarnessConfig, updates: dict[str, Any]) -> HarnessConfig:
    for key, value in updates.items():
        if key == "llm" and isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if hasattr(config.llm, inner_key) and inner_value is not None:
                    setattr(config.llm, inner_key, inner_value)
        elif hasattr(config, key) and value is not None:
            setattr(config, key, value)
    config.validate()
    return config


def handle(method: str, params: dict[str, Any]) -> Any:
    if method == "ping":
        return {"version": "0.4.15", "workspace": str(STATE["workspace"] or Path.cwd())}
    if method == "get_config":
        if STATE["config"] is None:
            _load_workspace(str(Path.cwd()))
        return _config_to_dict(STATE["config"])
    if method == "set_config":
        if STATE["config"] is None:
            _load_workspace(str(Path.cwd()))
        config = _apply_config(STATE["config"], params.get("config", {}))
        STATE["config"] = config
        if STATE["chat"] is not None:
            STATE["chat"].config = config
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
        return {"content": STATE["chat"].toolbox.read_file(str(params.get("path", "")), int(params.get("max_chars", 12000)))}
    if method == "compress_context":
        if STATE["chat"] is None:
            _load_workspace(str(Path.cwd()))
        changed = STATE["chat"].compress_if_needed()
        return {"compressed": changed}
    raise ValueError(f"unknown method: {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            method = request.get("method", "")
            params = request.get("params", {}) or {}
            result = handle(method, params)
            response = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            response = {"id": request_id, "ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
