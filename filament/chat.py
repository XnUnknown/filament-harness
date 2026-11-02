"""Conversational main orchestrator: plans exact agent tasks from natural requests."""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
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


MAX_PLAN_TASKS = 6


def _sanitize_target(value: Any) -> str:
    """Accept only safe relative paths from untrusted planner output."""
    target = str(value or "").strip().replace("\\", "/")
    if not target:
        return "output.py"
    candidate = PurePosixPath(target)
    if (
        target.startswith("/")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or any(part in {"", ".", ".."} or ":" in part for part in candidate.parts)
    ):
        return "output.py"
    normalized = str(candidate)
    return normalized if normalized and normalized != "." else "output.py"


def _extract_json(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("planner response must be a string")
    cleaned = text.strip()
    # Remove a surrounding markdown code fence if the model added one.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object found in planner response")
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        return obj
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


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
            task_cap = min(self.config.max_tasks, MAX_PLAN_TASKS)
            normalized = []
            for index, item in enumerate(tasks[:task_cap], 1):
                if isinstance(item, str):
                    normalized.append({"id": f"task{index}", "task": item.strip(), "target": "output.py"})
                elif isinstance(item, dict):
                    task_text = str(item.get("task") or item.get("instruction") or "").strip()
                    if not task_text:
                        continue
                    normalized.append(
                        {
                            "id": str(item.get("id") or f"task{index}").strip() or f"task{index}",
                            "task": task_text,
                            "target": _sanitize_target(item.get("target")),
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
