"""Tools available to the main Filament orchestrator."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .config import HarnessConfig


class Toolbox:
    def __init__(self, cwd: str | Path, config: HarnessConfig):
        self.root = Path(cwd).resolve()
        self.config = config

    def _safe(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"path escapes working directory: {resolved}")
        return resolved

    def read_file(self, path: str | Path, max_chars: int = 12000) -> str:
        target = self._safe(path)
        if not target.exists():
            raise FileNotFoundError(str(target))
        if target.is_dir():
            raise ValueError(f"not a file: {target}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"
        return text

    def write_file(self, path: str | Path, content: str, append: bool = False) -> str:
        target = self._safe(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return f"wrote {len(content)} chars to {target.relative_to(self.root)}"

    def list_dir(self, path: str | Path = ".") -> str:
        target = self._safe(path)
        if not target.is_dir():
            raise ValueError(f"not a directory: {target}")
        entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        lines = []
        for item in entries[:200]:
            suffix = "/" if item.is_dir() else ""
            lines.append(f"{item.name}{suffix}")
        if len(entries) > 200:
            lines.append(f"... {len(entries) - 200} more entries")
        return "\n".join(lines) if lines else "(empty)"

    def run_shell(self, command: str, timeout: int | None = None) -> str:
        if not command or not command.strip():
            raise ValueError("command must not be empty")
        timeout = timeout or self.config.timeout_shell
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            if len(output) > self.config.max_shell_output:
                output = output[: self.config.max_shell_output] + "\n... [truncated]"
            return f"exit_code: {completed.returncode}\n{output}"
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "") + (exc.stderr or "")
            return f"exit_code: timeout\n{partial[: self.config.max_shell_output]}"

    def check_python(self, path: str | Path) -> str:
        target = self._safe(path)
        if target.suffix != ".py":
            return "not a Python file"
        try:
            compile(target.read_text(encoding="utf-8"), str(target), "exec")
            return "syntax ok"
        except SyntaxError as exc:
            return f"syntax error: {exc}"

    def call(self, tool: str, args: dict | None = None) -> str:
        args = args or {}
        if tool == "read_file":
            return self.read_file(args.get("path", ""), int(args.get("max_chars", 12000)))
        if tool == "write_file":
            return self.write_file(args.get("path", ""), str(args.get("content", "")), bool(args.get("append", False)))
        if tool == "list_dir":
            return self.list_dir(args.get("path", "."))
        if tool == "run_shell":
            return self.run_shell(str(args.get("command", "")), args.get("timeout"))
        if tool == "check_python":
            return self.check_python(args.get("path", ""))
        raise ValueError(f"unknown tool: {tool}")
