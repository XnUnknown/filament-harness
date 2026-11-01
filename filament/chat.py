"""Conversational main orchestrator: plans exact agent tasks from natural requests."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import HarnessConfig, load_config
from .llm import build_client
from .orchestrator import Orchestrator


PLAN_SYSTEM = (
    "You are Filament's main orchestrator. Convert a user's coding request into "
    "a JSON plan for focused coding agents. Return ONLY valid JSON. "
    "Schema: {\"summary\": \"short summary\", \"tasks\": ["
    "{\"id\": \"task1\", \"task\": \"exact coding instruction with signature and behavior\", "
    "\"target\": \"relative/output.py\"}]}. "
    "Rules: maximum 6 tasks; keep tasks independent; each task must be exact; "
    "use relative paths only; no markdown."
)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


class ChatOrchestrator:
    def __init__(self, config: HarnessConfig, shell_cwd: str = "."):
        self.config = config
        self.shell_cwd = shell_cwd
        self.client = build_client(config.llm)
        self.history: list[str] = []

    def plan(self, request: str) -> tuple[str, list[dict[str, Any]]]:
        prompt = f"User request: {request}\n\nReturn the JSON plan now."
        raw = self.client.generate(prompt, system=PLAN_SYSTEM)
        try:
            data = _extract_json(raw)
            summary = str(data.get("summary") or "").strip()
            tasks = data.get("tasks") or []
            if not isinstance(tasks, list) or not tasks:
                raise ValueError("plan has no tasks")
            normalized = []
            for index, item in enumerate(tasks[: self.config.max_tasks], 1):
                if isinstance(item, str):
                    normalized.append({"id": f"task{index}", "task": item, "target": "output.py"})
                elif isinstance(item, dict):
                    task_text = str(item.get("task") or item.get("instruction") or "").strip()
                    if not task_text:
                        continue
                    target = str(item.get("target") or "output.py").strip()
                    normalized.append(
                        {
                            "id": str(item.get("id") or f"task{index}"),
                            "task": task_text,
                            "target": target or "output.py",
                        }
                    )
            if not normalized:
                raise ValueError("plan tasks were empty")
            return summary, normalized
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fall back to a single focused agent if the model cannot plan JSON.
            return "Fallback single-task plan", [
                {"id": "task1", "task": request, "target": "output.py"}
            ]

    def run_plan(
        self,
        tasks: list[dict[str, Any]],
        review: bool = True,
        fix: bool = True,
        max_fixes: int | None = None,
    ) -> dict:
        orchestrator = Orchestrator(
            config=self.config,
            shell_cwd=self.shell_cwd,
            allow_outside_target=self.config.allow_outside_target,
        )
        result = orchestrator.run(tasks, "output.py", review=review, fix=fix, max_fixes=max_fixes)
        result["plan_summary"] = ""
        return result

    def run_request(
        self,
        request: str,
        review: bool = True,
        fix: bool = True,
        max_fixes: int | None = None,
    ) -> dict:
        self.history.append(request)
        summary, tasks = self.plan(request)
        result = self.run_plan(tasks, review=review, fix=fix, max_fixes=max_fixes)
        result["plan_summary"] = summary
        return result
