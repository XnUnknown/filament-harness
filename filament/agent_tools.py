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
        if "\x00" in str(path):
            raise ValueError("path contains a null byte")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in str(path)):
            raise ValueError("path contains a control character")
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
        atomic = bool(getattr(self.config, "atomic_writes", False))

        if append:
            from .files import append_code
            append_code(target, content, atomic=atomic)
            return f"wrote {len(content)} chars to {target.relative_to(self.root)}"

        if atomic:
            import tempfile
            fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                if target.exists():
                    try:
                        os.chmod(temp_name, target.stat().st_mode)
                    except OSError:
                        pass
                os.replace(Path(temp_name), target)
            finally:
                leftover = Path(temp_name)
                if leftover.exists():
                    leftover.unlink(missing_ok=True)
        else:
            with target.open("w", encoding="utf-8", newline="\n") as handle:
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
        if len(command) > 20_000:
            raise ValueError("command is too long")
        if self.config.command_allowlist:
            stripped = command.lstrip()
            if not any(stripped.startswith(prefix) for prefix in self.config.command_allowlist):
                raise ValueError("command is not on the configured allowlist")
            if any(ch in command for ch in ("\n", "\r", ";", "|", "&", "`", "$", "<", ">", "(", ")", "{", "}")):
                raise ValueError("command contains shell metacharacters not permitted with an allowlist")

        timeout = timeout or self.config.timeout_shell
        use_shell = self.config.use_shell
        try:
            if use_shell:
                completed = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.root),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    errors="replace",
                )
            else:
                import shlex
                completed = subprocess.run(
                    shlex.split(command),
                    shell=False,
                    cwd=str(self.root),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    errors="replace",
                )
            output = (completed.stdout or "") + (completed.stderr or "")
            if len(output) > self.config.max_shell_output:
                output = _truncate(output, self.config.max_shell_output)
            return "exit_code: " + str(completed.returncode) + chr(10) + output
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "") + (exc.stderr or "")
            if len(partial) > self.config.max_shell_output:
                partial = _truncate(partial, self.config.max_shell_output)
            return "exit_code: timeout" + chr(10) + partial
        except (OSError, ValueError) as exc:
            return "exit_code: error" + chr(10) + str(exc)

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

def _truncate(text: str, limit: int) -> str:
    if limit < 1:
        raise ValueError("limit must be positive")
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    marker = chr(10) + "... [truncated " + str(omitted) + " chars] ..." + chr(10)
    if limit <= len(marker):
        return text[-limit:]
    head_budget = (limit - len(marker)) // 2
    tail_budget = limit - len(marker) - head_budget
    return text[:head_budget] + marker + text[-tail_budget:]
