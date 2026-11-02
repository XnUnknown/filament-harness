"""Tools available to agents: run_shell and spawn_agent."""
from __future__ import annotations

import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable

from .config import HarnessConfig
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

        from pathlib import Path

        cwd_path = Path(self.shell_cwd).expanduser()
        if not cwd_path.exists():
            raise ValueError(f"shell working directory does not exist: {cwd_path}")
        cwd_path = cwd_path.resolve()
        timeout = timeout or self.config.timeout_shell

        use_shell = self.config.use_shell
        try:
            if use_shell:
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(cwd_path),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    errors="replace",
                )
            else:
                proc = subprocess.run(
                    shlex.split(command),
                    cwd=str(cwd_path),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    errors="replace",
                )
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "") + (exc.stderr or "")
            text = _truncate(partial, self.config.max_shell_output)
            result = f"exit_code: timeout\n{text}"
            self.budget.add(f"shell:{command}\n{result}")
            return result
        except (OSError, ValueError) as exc:
            result = f"exit_code: error\n{exc}"
            self.budget.add(f"shell:{command}\n{result}")
            return result

        output = (proc.stdout or "") + (proc.stderr or "")
        output = _truncate(output, self.config.max_shell_output)
        result = f"exit_code: {proc.returncode}\n{output}"
        self.budget.add(f"shell:{command}\n{result}")
        return result

    def spawn_agent(self, task: str, system: str | None = None) -> str:
        task = task.strip()
        if not task:
            raise ValueError("task must not be empty")
        if len(task) > 50_000:
            raise ValueError("task is too long")
        prompt = task
        reserve = max(512, self.budget.max_tokens // 20)
        if self.budget.total_tokens() + self.budget.estimate(prompt) + reserve > self.budget.max_tokens:
            raise RuntimeError("context budget exceeded before spawn")
        added_prompt = self.budget.add(f"spawn:{prompt}")
        if not added_prompt:
            raise RuntimeError("context budget exceeded before spawn")
        result = self.client.generate(prompt, system=system or CODING_SYSTEM)
        self.budget.add(f"result:{result}")
        return result


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"



