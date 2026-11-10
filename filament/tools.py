"""Tools available to agents: run_shell and spawn_agent."""
from __future__ import annotations

import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable

from .config import HarnessConfig, coerce_timeout
from .context import ContextBudget
from .llm import LLMClient


CODING_SYSTEM = (
    "You are a focused coding agent. "
    "Return only the requested code in one fenced block. No prose."
)

_SHELL_LOCK = threading.Lock()


@dataclass
class Toolset:
    client: LLMClient
    budget: ContextBudget
    config: HarnessConfig
    shell_cwd: str = "."

    def run_shell(self, command: str, timeout: int | None = None) -> str:
        if not command or not command.strip():
            raise ValueError("command must not be empty")
        if len(command) > 20_000:
            raise ValueError("command is too long")
        if self.config.command_allowlist:
            stripped = command.lstrip()
            if not any(stripped.startswith(prefix) for prefix in self.config.command_allowlist):
                raise ValueError("command is not on the configured allowlist")
            # Prevent command chaining/redirection when allowlist is active.
            if any(ch in command for ch in ("\n", "\r", ";", "|", "&", "`", "$", "<", ">", "(", ")", "{", "}")):
                raise ValueError("command contains shell metacharacters not permitted with an allowlist")

        from pathlib import Path

        cwd_path = Path(self.shell_cwd).expanduser()
        if not cwd_path.exists() or not cwd_path.is_dir():
            raise ValueError(f"shell working directory does not exist: {cwd_path}")
        cwd_path = cwd_path.resolve()
        if chr(0) in command:
            raise ValueError("command contains a null byte")
        if any(ord(ch) < 32 and ch not in '\t\n\r' for ch in command) or chr(127) in command:
            raise ValueError("command contains a control character")
        resolved_timeout = coerce_timeout(timeout, self.config.timeout_shell)

        use_shell = self.config.use_shell
        try:
            if use_shell:
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(cwd_path),
                    capture_output=True,
                    text=True,
                    timeout=resolved_timeout,
                    errors="replace",
                )
            else:
                proc = subprocess.run(
                    shlex.split(command),
                    cwd=str(cwd_path),
                    capture_output=True,
                    text=True,
                    timeout=resolved_timeout,
                    errors="replace",
                )
        except subprocess.TimeoutExpired as exc:
            partial = str(exc.stdout or "") + str(exc.stderr or "")
            saved_path = None
            if len(partial) > self.config.max_shell_output:
                saved_path = self._offload_shell_output(partial)
                partial = _truncate(partial, self.config.max_shell_output)
            result = f"exit_code: timeout\n{partial}"
            if saved_path is not None:
                result += f"\n[full output saved: {saved_path}]"
            self.budget.add(f"shell:{command}\n{result}")
            return result
        except (OSError, ValueError) as exc:
            result = f"exit_code: error\n{exc}"
            self.budget.add(f"shell:{command}\n{result}")
            return result

        output = (proc.stdout or "") + (proc.stderr or "")
        saved_path = None
        if len(output) > self.config.max_shell_output:
            saved_path = self._offload_shell_output(output)
            output = _truncate(output, self.config.max_shell_output)
        result = f"exit_code: {proc.returncode}\n{output}"
        if saved_path is not None:
            result += f"\n[full output saved: {saved_path}]"
        self.budget.add(f"shell:{command}\n{result}")
        return result


    def _offload_shell_output(self, output: str) -> str | None:
        if not bool(getattr(self.config, "offload_shell_output", True)):
            return None
        try:
            import uuid
            from pathlib import Path

            root = Path(self.shell_cwd).expanduser().resolve()
            directory = root / ".filament" / "runs"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / ("shell-" + uuid.uuid4().hex[:12] + ".log")
            path.write_text(output, encoding="utf-8", errors="replace")
            return str(path.relative_to(root))
        except (OSError, ValueError):
            return None

    def spawn_agent(self, task: str, system: str | None = None) -> str:
        task = task.strip()
        if not task:
            raise ValueError("task must not be empty")
        if len(task) > 50_000:
            raise ValueError("task is too long")
        prompt = task
        reserve = max(512, self.budget.max_tokens // 20)
        with self.budget._lock:
            if self.budget.total_tokens() + self.budget.estimate(f"spawn:{prompt}") + reserve > self.budget.max_tokens:
                raise RuntimeError("context budget exceeded before spawn")
            added_prompt = self.budget.add(f"spawn:{prompt}")
            if not added_prompt:
                raise RuntimeError("context budget exceeded before spawn")
        result = self.client.generate(prompt, system=system or CODING_SYSTEM)
        self.budget.add(f"result:{result}")
        return result


def _truncate(text: str, limit: int) -> str:
    if limit < 1:
        raise ValueError("limit must be positive")
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    marker = f"\n... [truncated {omitted} chars] ...\n"
    if limit <= len(marker):
        return text[-limit:]
    head_budget = (limit - len(marker)) // 2
    tail_budget = limit - len(marker) - head_budget
    return text[:head_budget] + marker + text[-tail_budget:]



