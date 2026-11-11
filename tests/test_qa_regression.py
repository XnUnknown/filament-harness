"""Regression tests added during QA review."""
import sys
import tempfile
import unittest
from pathlib import Path

from filament.config import HarnessConfig
from filament.context import ContextBudget
from filament.files import append_code, replace_line
from filament.llm import MockLLM
from filament.review import review_file
from filament.session import _safe_session_id
from filament.tools import Toolset


class _CapturingFixClient:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.reply



class TestQARegressions(unittest.TestCase):
    def test_context_budget_preserves_context_when_marker_does_not_fit(self):
        budget = ContextBudget(max_tokens=8, chars_per_token=4)
        self.assertTrue(budget.add("a" * 4))
        before = budget.dump()
        self.assertFalse(budget.add("x" * 1000))
        self.assertEqual(budget.dump(), before)

    def test_append_code_rejects_drive_relative_path(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                append_code("C:evil.py", "x\n")

    def test_replace_line_rejects_drive_relative_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.py"
            path.write_text("old\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                replace_line("C:evil.py", 1, "new")

    def test_session_id_rejects_colon(self):
        with self.assertRaises(ValueError):
            _safe_session_id("session:evil")


    def test_task_whitespace_target_inherits_parent_target(self):
        from filament.tasks import flatten_tasks, parse_tasks

        tasks = parse_tasks(
            [
                {
                    "id": "parent",
                    "task": "parent work",
                    "target": "parent.py",
                    "subtasks": [{"id": "child", "task": "child work", "target": "   "}],
                }
            ]
        )
        flat = flatten_tasks(tasks)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0].target, "parent.py")

    def test_subtasks_must_be_a_list(self):
        from filament.tasks import parse_tasks

        for bad in ("build the thing", {"task": "nested"}, 5):
            with self.assertRaises(ValueError):
                parse_tasks([{"task": "parent", "subtasks": bad}])

    def test_review_truncation_reports_absolute_line_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "big.py"
            lines = [f"line_{i:04d} = {i}" for i in range(1, 1201)]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            client = _CapturingFixClient("FIX 1: replace this")
            fixes = review_file(
                client, path, ContextBudget(max_tokens=2048, chars_per_token=4)
            )
            self.assertEqual(len(fixes), 1)
            line_no, instruction = fixes[0]
            self.assertEqual(instruction, "replace this")
            self.assertGreater(line_no, 1)
            excerpt = client.prompts[0].split("```\n", 1)[1]
            first_excerpt_line = excerpt.splitlines()[0]
            self.assertEqual(lines[line_no - 1], first_excerpt_line)

    def test_toolset_without_shell_runs_unquoted_executable_path(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = HarnessConfig(use_shell=False)
            toolset = Toolset(
                client=MockLLM(),
                budget=ContextBudget(max_tokens=4096),
                config=cfg,
                shell_cwd=d,
            )
            # An unquoted native path exercises backslash handling on Windows;
            # POSIX-mode shlex would otherwise turn C:\dir into C:dir.
            command = f'{sys.executable} -c "print(424242)"'
            result = toolset.run_shell(command, timeout=60)
            self.assertIn("exit_code: 0", result)
            self.assertIn("424242", result)

if __name__ == "__main__":
    unittest.main()
