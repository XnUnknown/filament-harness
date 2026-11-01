"""Tools available to agents: run_shell and spawn_agent."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

from .context import ContextBudget
from .llm import LLMClient


CODING_SYSTEM = (
    "You are a focused coding agent. "
    "Return only the requested code in one fenced block. No prose."
)


@dataclass
class Toolset:
    client: LLMClient
    budget: ContextBudget
    shell_cwd: str = "."

    def run_shell(self, command: str, timeout: int = 60) -> str:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=self.shell_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        self.budget.add(f"shell:{command}\n{out}")
        return out

    def spawn_agent(self, task: str, system: str | None = None) -> str:
        prompt = task.strip()
        self.budget.add(f"spawn:{prompt}")
        if self.budget.total_tokens() > self.budget.max_tokens:
            raise RuntimeError("context budget exceeded before spawn")
        result = self.client.generate(prompt, system=system or CODING_SYSTEM)
        self.budget.add(f"result:{result}")
        return result
