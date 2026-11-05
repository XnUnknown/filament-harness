"""Main orchestrator: spawns focused agents, batches appends, reviews, fixes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import CodingAgent
from .config import HarnessConfig, load_config
from .context import ContextBudget
from .files import append_code, ensure_target_within_root, extract_code, replace_line
from .llm import build_client
from .review import review_file
from .tasks import TaskSpec, flatten_tasks, parse_tasks, task_instruction
from .tools import Toolset


class Orchestrator:
    def __init__(
        self,
        client_kind: str | None = None,
        config: HarnessConfig | None = None,
        max_tokens: int | None = None,
        shell_cwd: str = ".",
        allow_outside_target: bool | None = None,
    ):
        if config is None:
            config = load_config()
        if max_tokens is not None:
            config.max_tokens = max_tokens
        if allow_outside_target is not None:
            config.allow_outside_target = allow_outside_target
        if client_kind is not None:
            config.llm.kind = client_kind
        config.validate()

        self.config = config
        self.shell_cwd = shell_cwd
        self.budget = ContextBudget(
            max_tokens=config.effective_max_tokens,
            chars_per_token=config.chars_per_token,
        )
        self.client = build_client(config.llm)
        self.tools = Toolset(
            client=self.client,
            budget=self.budget,
            config=config,
            shell_cwd=shell_cwd,
        )

    def _coerce_task_specs(self, tasks) -> list[TaskSpec]:
        if isinstance(tasks, (str, dict, TaskSpec)):
            tasks = [tasks]
        if not isinstance(tasks, list):
            raise ValueError("tasks must be a list")
        if all(isinstance(task, TaskSpec) for task in tasks):
            return list(tasks)
        return parse_tasks(tasks)

    def _one_line_replacement(self, text: str) -> str:
        raw = text.strip()
        try:
            code = extract_code(raw) if "```" in raw else raw
        except (TypeError, ValueError):
            code = raw
        lines = [line for line in code.splitlines() if line.strip()]
        if len(lines) != 1:
            return ""
        return lines[0].strip().strip("`").strip()

    def run(
        self,
        tasks,
        target_file: str,
        workers: int | None = None,
        review: bool = True,
        fix: bool = True,
        max_fixes: int | None = None,
    ):
        task_specs = self._coerce_task_specs(tasks)
        flattened = flatten_tasks(task_specs)
        if not flattened:
            raise ValueError("no enabled tasks to run")
        if len(flattened) > self.config.max_tasks:
            raise ValueError(f"task count {len(flattened)} exceeds max_tasks {self.config.max_tasks}")

        root = Path(self.shell_cwd).expanduser().resolve()
        plans = []
        for task in flattened:
            effective_target = task.target or target_file
            target_path = ensure_target_within_root(
                effective_target,
                root,
                self.config.allow_outside_target,
            )
            plans.append((task, target_path))

        worker_count = min(max(1, workers or self.config.workers), len(plans))
        raw_results: list[str | None] = [None] * len(plans)
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_index = {
                pool.submit(CodingAgent(task_instruction(task)).run, self.tools): index
                for index, (task, _) in enumerate(plans)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    raw_results[index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"task {plans[index][0].id}: {exc}")

        extracted: list[tuple[int, Path, str]] = []
        for index, (task, target_path) in enumerate(plans):
            raw = raw_results[index]
            if raw is None:
                continue
            try:
                code = extract_code(raw)
            except (TypeError, ValueError) as exc:
                errors.append(f"task {task.id}: {exc}")
                continue
            extracted.append((index, target_path, code))

        appended_counts: dict[str, int] = {}
        for _, target_path, code in extracted:
            append_code(target_path, code, atomic=self.config.atomic_writes)
            appended_counts[str(target_path)] = appended_counts.get(str(target_path), 0) + 1

        main_target = ensure_target_within_root(
            target_file,
            root,
            self.config.allow_outside_target,
        )

        syntax_ok = True
        if main_target.suffix == ".py" and main_target.exists():
            try:
                compile(main_target.read_text(encoding="utf-8"), str(main_target), "exec")
            except SyntaxError as exc:
                syntax_ok = False
                errors.append(f"syntax validation failed for {main_target}: {exc}")

        fixes_applied: list[dict] = []
        if review and main_target.exists():
            try:
                fixes = review_file(self.client, main_target, self.budget)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"review failed: {exc}")
                fixes = []
            if fix:
                limit = len(fixes) if max_fixes is None else min(len(fixes), max(0, max_fixes))
                for line_no, instruction in fixes[:limit]:
                    try:
                        line_count = len(main_target.read_text(encoding="utf-8").splitlines())
                    except OSError as exc:
                        errors.append(f"fix skipped for line {line_no}: {exc}")
                        continue
                    if line_no < 1 or line_no > line_count:
                        errors.append(f"review fix skipped (line {line_no} out of range)")
                        continue
                    fix_task = (
                        f"[fix line {line_no}] In {main_target.name}, replace line {line_no}: "
                        f"{instruction}. Return only the replacement line."
                    )
                    try:
                        replacement_raw = self.tools.spawn_agent(fix_task)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"fix agent failed for line {line_no}: {exc}")
                        continue
                    replacement = self._one_line_replacement(replacement_raw)
                    if not replacement:
                        errors.append(f"fix agent returned no single replacement for line {line_no}")
                        continue
                    try:
                        replace_line(main_target, line_no, replacement)
                        fixes_applied.append(
                            {"line": line_no, "instruction": instruction, "replacement": replacement}
                        )
                    except (IndexError, OSError, ValueError) as exc:
                        errors.append(f"fix apply failed for line {line_no}: {exc}")

        return {
            "target_file": str(main_target),
            "targets": appended_counts,
            "tasks": len(flattened),
            "source_tasks": len(task_specs),
            "appended": len(extracted),
            "failed_tasks": len(errors),
            "errors": errors,
            "fixes": fixes_applied,
            "context_tokens": self.budget.total_tokens(),
            "budget_limit": self.budget.max_tokens,
            "syntax_ok": syntax_ok,
        }
