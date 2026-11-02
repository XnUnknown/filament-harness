"""Task specification and subtask expansion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskSpec:
    id: str
    instruction: str
    target: str | None = None
    subtasks: list["TaskSpec"] = field(default_factory=list)
    enabled: bool = True


def _coerce_task(raw: Any, index: int, parent: str = "", inherited_target: str | None = None) -> TaskSpec:
    if isinstance(raw, str):
        task_id = f"{parent}{index + 1}"
        return TaskSpec(id=task_id, instruction=raw.strip(), target=inherited_target)
    if not isinstance(raw, dict):
        raise ValueError(f"task {index + 1} must be a string or object")
    instruction = str(raw.get("task") or raw.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"task {index + 1} is missing a task/instruction")
    task_id = str(raw.get("id") or f"{parent}{index + 1}")
    target = raw.get("target")
    if target in (None, ""):
        target = inherited_target
    else:
        target = str(target).strip()
    enabled = bool(raw.get("enabled", True))
    children = []
    for child_index, child in enumerate(raw.get("subtasks", []) or []):
        children.append(_coerce_task(child, child_index, f"{task_id}.", inherited_target=target))
    return TaskSpec(id=task_id, instruction=instruction, target=target, subtasks=children, enabled=enabled)


def parse_tasks(raw_tasks: Any) -> list[TaskSpec]:
    if not isinstance(raw_tasks, list):
        raise ValueError("tasks must be a list")
    tasks: list[TaskSpec] = []
    for index, raw in enumerate(raw_tasks):
        task = _coerce_task(raw, index)
        if task.enabled:
            tasks.append(task)
    return tasks


def flatten_tasks(tasks: list[TaskSpec]) -> list[TaskSpec]:
    """Flatten a task tree into execution order while preserving IDs."""
    result: list[TaskSpec] = []

    def visit(task: TaskSpec) -> None:
        if not task.enabled:
            return
        if task.subtasks:
            for child in task.subtasks:
                visit(child)
        else:
            result.append(task)

    for task in tasks:
        visit(task)
    return result


def task_instruction(task: TaskSpec) -> str:
    return f"[task {task.id}] {task.instruction}"
