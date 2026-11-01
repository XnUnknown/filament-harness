"""Main orchestrator: spawns focused agents, batches appends, reviews, fixes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import CodingAgent
from .context import ContextBudget
from .files import append_code, extract_code, replace_line
from .llm import make_client
from .review import review_file
from .tools import Toolset


class Orchestrator:
    def __init__(self, client_kind: str = "auto", max_tokens: int = 120_000, shell_cwd: str = "."):
        self.budget = ContextBudget(max_tokens=max_tokens)
        self.client = make_client(client_kind)
        self.tools = Toolset(client=self.client, budget=self.budget, shell_cwd=shell_cwd)
        self.max_tokens = max_tokens

    def run(self, tasks: list[str], target_file: str, workers: int = 4, review: bool = True, fix: bool = True):
        agents = [CodingAgent(t) for t in tasks]
        results: list[str] = [None] * len(agents)  # type: ignore[list-item]

        # Spawn agents in parallel; each gets only its own task.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(agent.run, self.tools): i for i, agent in enumerate(agents)}
            for fut in as_completed(future_map):
                i = future_map[fut]
                results[i] = fut.result()

        # Waited for all; now append everyone's code in one batch.
        codes = [extract_code(r) for r in results]
        for code in codes:
            append_code(target_file, code)

        fixes_applied = []
        if review:
            fixes = review_file(self.client, target_file, self.budget)
            if fix:
                for line_no, instruction in fixes:
                    task = (
                        f"Fix line {line_no} in {target_file}: {instruction}. "
                        "Return only the replacement line, no fences."
                    )
                    replacement = self.tools.spawn_agent(task)
                    replacement = replacement.strip().strip("`")
                    try:
                        replace_line(target_file, line_no, replacement)
                        fixes_applied.append((line_no, instruction))
                    except (IndexError, OSError):
                        fixes_applied.append((line_no, f"skipped: {instruction}"))

        return {
            "target_file": str(Path(target_file).resolve()),
            "agents": len(agents),
            "appended": len(codes),
            "fixes": fixes_applied,
            "context_tokens": self.budget.total_tokens(),
        }
